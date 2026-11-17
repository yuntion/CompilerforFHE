"""Generate restricted schedules, compile trusted C++, and validate candidates.

This driver never accepts source code from a provider. The native verifier and
code generator produce all executable code from the frozen compiler DAG.
"""
from __future__ import annotations

from collections import Counter
import heapq
from pathlib import Path
import subprocess
import time

from backend.execution import build_runner
from backend.toolchain import _native, command
from .artifacts import digest, verify_artifact, write_json
from .prompts import build_messages
from .protocol import canonical_bytes, parse_json, read_json, sha256, validate_candidate, verify_task
from .provider import ProviderError


def baseline_candidate(task_dir, strategy="depth-first"):
    """A deterministic competitor/replay fixture; this does not call a model."""
    task = verify_task(task_dir)
    dag = read_json(Path(task_dir)/"dag.json")
    if strategy == "reference":
        schedule = list(dag["reference_schedule"])
    elif strategy == "depth-first":
        nodes = dag["nodes"]
        order = {node["id"]: index for index, node in enumerate(nodes)}
        pending = {node["id"]: len(node["dependencies"]) for node in nodes}
        users = {node["id"]: [] for node in nodes}
        for node in nodes:
            for dependency in node["dependencies"]:
                users[dependency].append(node["id"])
        ready = [-order[node_id] for node_id, count in pending.items() if count == 0]
        heapq.heapify(ready)
        schedule = []
        while ready:
            node_id = nodes[-heapq.heappop(ready)]["id"]
            schedule.append(node_id)
            for user in users[node_id]:
                pending[user] -= 1
                if pending[user] == 0:
                    heapq.heappush(ready, -order[user])
    else:
        raise ValueError("baseline strategy must be reference or depth-first")
    candidate = {"schema_version": 1, "status": "proposal", "mode": "schedule-v1",
                 "task_hash": task["task_hash"], "schedule": schedule,
                 "rationale": [f"Deterministic {strategy} baseline; no model or performance claim."]}
    return validate_candidate(candidate, task, dag)


def schedule_metrics(dag, schedule):
    """Count live intermediate ciphertext values, not estimated RAM or GPU bytes."""
    nodes = {node["id"]: node for node in dag["nodes"]}
    cipher_nodes = {name for name, node in nodes.items() if "cipher" in node["result_type"].lower()}
    roots = set(dag["outputs"])
    remaining = Counter(operand for node in nodes.values() for operand in node["operands"] if operand in nodes)
    live, peak = set(), 0
    for node_id in schedule:
        if node_id in cipher_nodes:
            live.add(node_id)
        peak = max(peak, len(live))
        for operand in nodes[node_id]["operands"]:
            if operand in remaining:
                remaining[operand] -= 1
                if remaining[operand] == 0 and operand not in roots:
                    live.discard(operand)
        if remaining[node_id] == 0 and node_id not in roots:
            live.discard(node_id)
    return {"peak_live_intermediate_ciphertexts": peak,
            "total_intermediate_ciphertexts": len(cipher_nodes), "metric_is_bytes": False}


def run_candidates(task_dir, artifact, output, provider, config, *, cases=None):
    """Return an evidence report. A validated candidate never overwrites the reference."""
    from .validation import ValidationInfrastructureError, validate_encrypted

    task_dir, artifact, output = Path(task_dir).resolve(), Path(artifact).resolve(), Path(output).resolve()
    task = verify_task(task_dir)
    verify_artifact(artifact, task, config)
    if output == artifact or output.is_relative_to(artifact) or output == task_dir or output.is_relative_to(task_dir):
        raise ValueError("run output must be separate from the reference artifact and sealed task")
    if output.exists() and any(output.iterdir()):
        raise ValueError("run output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    dag = read_json(task_dir/"dag.json")
    # Independently reconstruct the graph with the pinned native tool. Never
    # validate a model against a graph supplied by that same model.
    translate = _native(config, "crossfhe-translate")
    command([translate, task_dir/"program.mlir", f"--backend={task['backend']}",
             "--emit-dag", "-o", output/"verified-dag.json"], config, log=output/"dag-verification.log")
    if canonical_bytes(read_json(output/"verified-dag.json")) != canonical_bytes(dag):
        raise ValueError("sealed DAG differs from the native compiler graph")
    command([translate, task_dir/"program.mlir", f"--backend={task['backend']}",
             "-o", output/"verified-reference.hpp"], config, log=output/"reference-verification.log")
    if digest(output/"verified-reference.hpp") != task["context_hashes"]["reference.hpp"]:
        raise ValueError("sealed reference differs from deterministic native codegen")
    api_context = read_json(task_dir/"api_context.json")
    runtime_name = {"seal-ckks": "seal_runtime.hpp", "openfhe-cross": "openfhe_runtime.hpp",
                    "phantom-ckks": "phantom_runtime.hpp"}[task["backend"]]
    api_context["runtime_header"] = {"path": runtime_name, "sha256": task["context_hashes"][runtime_name],
                                     "content": (task_dir/runtime_name).read_text(encoding="utf-8")}
    report = {"schema_version": 1, "task_hash": task["task_hash"], "backend": task["backend"],
              "status": "prepared", "attempts": [], "provider": type(provider).__name__,
              "validation_performed": False, "promotion_allowed": False, "selectable": False,
              "reference_artifact": str(artifact), "reference_retained": True,
              "execution_scope": "native-generated schedule code only; separate client/evaluator processes",
              "reference_with_last_use": schedule_metrics(dag, dag["reference_schedule"])}
    started = time.perf_counter()
    previous, feedback = None, None
    seen_schedules = set()
    for attempt_index in range(task["validation_policy"]["max_attempts"]):
        attempt = output/f"attempt-{attempt_index+1:02d}"
        attempt.mkdir()
        record = {"attempt": attempt_index+1, "status": "generating", "directory": str(attempt)}
        report["attempts"].append(record)
        stage = "generation"
        try:
            messages = build_messages(task, dag, api_context, previous=previous, feedback=feedback)
            write_json(attempt/"messages.json", messages)
            response = provider.generate(messages)
            raw = response["text"]
            (attempt/"response.txt").write_text(raw, encoding="utf-8")
            record["response_sha256"] = sha256(raw.encode("utf-8"))
            record["usage"] = response.get("usage", {})
            stage = "schema"
            previous = {"raw_response": raw}
            candidate = validate_candidate(parse_json(raw), task, dag)
            previous = candidate
            write_json(attempt/"candidate.json", candidate)
            if candidate["status"] == "no_candidate":
                record["status"] = report["status"] = "no_candidate"
                break
            schedule_hash = sha256(canonical_bytes(candidate["schedule"]))
            if schedule_hash in seen_schedules:
                raise ValueError("duplicate schedule: no new candidate; a failed schedule cannot be retried with fresh keys")
            seen_schedules.add(schedule_hash)
            record["schedule_sha256"] = schedule_hash
            stage = "native_verification"
            command([translate, task_dir/"program.mlir", f"--backend={task['backend']}",
                     f"--schedule={attempt/'candidate.json'}", "-o", attempt/"candidate.hpp"],
                    config, log=attempt/"codegen.log")
            record["status"] = "plan_verified"
            record["candidate_sha256"] = digest(attempt/"candidate.hpp")
            record["schedule_metrics"] = schedule_metrics(dag, candidate["schedule"])
            stage = "build"
            runner, record["build_seconds"] = build_runner(attempt/"candidate.hpp", attempt/"build",
                task["backend"], config, attempt/"logs", contract_header=task_dir/"reference.hpp")
            record["runner_sha256"] = digest(runner)
            record["status"] = "build_passed"
            stage = "functional"
            validation = validate_encrypted(artifact, runner, task, attempt/"validation", config, cases=cases)
            record["validation"] = validation
            report["validation_performed"] = validation["validation_performed"]
            record["status"] = report["status"] = "validated" if validation["policy_passed"] else "functional_only"
            report["candidate"] = str(attempt/"candidate.hpp")
            report["policy_passed"] = validation["policy_passed"]
            # First version archives evidence for explicit review. Timing and
            # finite regression tests alone do not authorize automatic promotion.
            break
        except ProviderError as error:
            record.update(status="provider_error", error=str(error))
            report["status"] = "provider_error"
            break
        except (ValidationInfrastructureError, OSError) as error:
            record.update(status="infrastructure_error", stage=stage, error=str(error)[-8000:])
            report["status"] = "infrastructure_error"
            break
        except subprocess.TimeoutExpired:
            record.update(status="infrastructure_error", stage=stage, error="backend command timed out")
            report["status"] = "infrastructure_error"
            break
        except (ValueError, RuntimeError) as error:
            record.update(status="rejected", stage=stage, error=str(error)[-8000:])
            validation_path = attempt/"validation/report.json"
            if stage == "functional" and validation_path.is_file():
                record["validation"] = read_json(validation_path)
                report["validation_performed"] = report["validation_performed"] or record["validation"]["validation_performed"]
            # Do not feed private inputs, decrypted results or filesystem paths
            # from client artifacts back into a model. Numerical diagnostics from
            # the verifier contain only case/output IDs and aggregate metrics.
            feedback = {"failed_stage": stage, "diagnostic": record["error"],
                        "attempt": attempt_index+1,
                        "remaining_attempts": task["validation_policy"]["max_attempts"]-attempt_index-1}
            write_json(attempt/"feedback.json", feedback)
            report["status"] = "rejected"
        finally:
            report["elapsed_seconds"] = time.perf_counter()-started
            write_json(output/"report.json", report)
    return report
