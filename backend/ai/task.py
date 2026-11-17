"""Prepare a sealed public scheduling task from a compiled compiler artifact."""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform

from backend.execution import BACKEND_DIRECTORIES
from backend.toolchain import ROOT
from .artifacts import json_bytes
from .context import load_documentation
from .protocol import CANDIDATE_SCHEMA, read_json, seal_task, validate_task
from .prompts import SYSTEM_PROMPT

PUBLIC_FILES = {"model.hpp": "reference.hpp", "04_optimized.mlir": "program.mlir",
                "resources_after.json": "contract.json", "dag.json": "dag.json"}
BACKEND_SOURCE_FILES = {
    backend: frozenset({
        "backend/CMakeLists.txt", f"backend/{directory}/CMakeLists.txt",
        f"backend/{directory}/{directory}_runtime.hpp",
        f"backend/{directory}/runner.{'cu' if directory == 'phantom' else 'cpp'}",
        *({"backend/phantom/phantom_decode.hpp"} if directory == "phantom" else set()),
    })
    for backend, directory in BACKEND_DIRECTORIES.items()
}


def prepare(artifact, output, *, documentation_root=None, atol=0.005, rtol=0.005,
            input_bound=0.5, seed=17, comparison_scale=1024.0, insecure_demo=False,
            max_attempts=10, repetitions=3, warmups=1):
    artifact, output = Path(artifact).resolve(), Path(output).resolve()
    compilation_status = read_json(artifact/"compilation_status.json")
    if compilation_status.get("state") != "ready":
        raise ValueError("artifact must be compiled and ready")
    manifest = read_json(artifact/"manifest.json")
    backend = manifest.get("backend")
    if not isinstance(backend, str) or backend not in BACKEND_DIRECTORIES:
        raise ValueError("AI scheduling supports seal-ckks, openfhe-cross and phantom-ckks; no standalone TFHE backend")
    if output == artifact or output.is_relative_to(artifact):
        raise ValueError("task output must be separate from the executable artifact")
    if output.exists() and any(output.iterdir()):
        raise ValueError("task output must be empty; use a new directory")
    profile = "INSECURE_TOY" if insecure_demo else (
        "HEStd_128_classic+STD128" if backend == "openfhe-cross" else "tc128")
    policy = {"security_profile": profile, "insecure_demo": insecure_demo,
              "comparison_scale": comparison_scale, "atol": atol, "rtol": rtol,
              "input_bound": input_bound, "seed": seed, "max_attempts": max_attempts,
              "warmups": warmups, "repetitions": repetitions}
    analysis = manifest.get("resource_analysis")
    contract = analysis.get("after") if isinstance(analysis, dict) else None
    if not isinstance(contract, dict) or not contract:
        raise ValueError("artifact has no native resource contract; compile again")
    files = {}
    hashes = manifest.get("artifact_hashes", {})
    for source, destination in PUBLIC_FILES.items():
        path = artifact/source
        content = path.read_bytes()
        if source not in hashes or hashlib.sha256(content).hexdigest() != hashes[source]:
            raise ValueError(f"artifact hash mismatch: {source}")
        files[destination] = content
    if read_json(artifact/"resources_after.json") != contract:
        raise ValueError("manifest and native resource contract differ")
    dag = read_json(artifact/"dag.json")
    directory = BACKEND_DIRECTORIES[backend]
    context = manifest.get("backend_context")
    if (not isinstance(context, dict) or not isinstance(context.get("library_revision"), str)
            or not context["library_revision"].strip()):
        raise ValueError("artifact lacks versioned backend context; compile with the current compiler")
    if context.get("backend") != backend or not isinstance(context.get("source_hashes"), dict):
        raise ValueError("artifact backend context mismatch")
    # Validate the complete set before reading any manifest-selected source. A
    # broad 'backend/' prefix would permit another backend or an arbitrary file.
    source_hashes = context["source_hashes"]
    if set(source_hashes) != BACKEND_SOURCE_FILES[backend]:
        raise ValueError("backend context source_files must match the fixed backend allowlist")
    runtime_name = f"{directory}_runtime.hpp"
    runtime_source = f"backend/{directory}/{runtime_name}"
    for name in sorted(BACKEND_SOURCE_FILES[backend]):
        expected, path = source_hashes[name], ROOT/name
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"backend source changed since compilation: {name}; compile again")
        if name == runtime_source:
            files[runtime_name] = content
    queries = ["rotate", "rescale", "multiply", "relinearize", "add"]
    if backend == "openfhe-cross":
        queries = ["EvalCKKStoFHEW", "EvalSign", "EvalFHEWtoCKKS", "EvalMult", "EvalAdd"]
    api_context = load_documentation(documentation_root, backend, queries)
    mapping = api_context["function_mapping"]
    if mapping["runtime_sha256"] != hashlib.sha256(files[runtime_name]).hexdigest():
        raise ValueError("function mapping runtime hash mismatch; update backend/api mapping for this runtime")
    mapping["runtime_hash_matches_task"] = True
    mapping["library_revision_matches_task"] = mapping["library_revision"] == context["library_revision"]
    files["api_context.json"] = json_bytes(api_context)
    files["candidate.schema.json"] = json_bytes(CANDIDATE_SCHEMA)
    files["PROMPT.md"] = (SYSTEM_PROMPT + "\n\nRead task.json, dag.json, contract.json, api_context.json, "
                          "reference.hpp and the pinned runtime header. A proposal is unvalidated.\n").encode("utf-8")
    task = {
        "schema_version": 2, "mode": "schedule-v1", "status": "prepared_unvalidated",
        "backend": backend, "target": "nvidia-cuda" if backend == "phantom-ckks" else platform.machine()+"-cpu",
        "model_sha256": manifest["model_sha256"], "library_revision": context["library_revision"],
        "backend_context": context, "optimization": manifest["optimization"],
        "approximation": manifest.get("approximation", {}), "contract": contract,
        "tensor_mapping": {"inputs": manifest.get("inputs", []), "outputs": manifest.get("outputs", [])},
        "validation_policy": policy,
        "objective": {"primary": "reduce live intermediate ciphertexts", "secondary": "measure evaluator latency",
                      "parameter_changes_allowed": False},
        "candidate_scope": "serial reordering of existing DAG nodes only",
        "immutable": ["DAG operations, operands, constants and outputs", "input/output ABI and tensor mapping",
                      "FHE algorithm, layout and switching encoding", "parameters, precision and approximation policy",
                      "key ownership and client/server boundary"],
        "required_validation": ["native schedule verifier", "real C++ compilation",
                                "same-input encrypted reference differential", "independent plaintext reference",
                                "fixed parameter profile", "repeated evaluator timing"],
        "context_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "validation_performed": False, "promotion_allowed": False,
    }
    validate_task(task, dag)
    task = seal_task(task)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (output/name).write_bytes(data)
    (output/"task.json").write_bytes(json_bytes(task))
    return task
