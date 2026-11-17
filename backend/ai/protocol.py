"""Public task integrity and the restricted, serial schedule candidate protocol.

Hashes identify the sealed task and its context; they are not signatures or an
authorization to run a candidate.  Acceptance remains the driver's decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


MAX_NODE_ID_LENGTH = 128
MAX_RATIONALE_ITEMS = 32
MAX_TEXT_LENGTH = 4096
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# Flat, public files only.  The manifest cannot authorize arbitrary path reads.
CONTEXT_FILES = frozenset({
    "program.mlir", "reference.hpp", "contract.json", "dag.json",
    "api_context.json", "candidate.schema.json", "PROMPT.md",
    "seal_runtime.hpp", "openfhe_runtime.hpp", "phantom_runtime.hpp",
})
REQUIRED_CONTEXT_FILES = frozenset({
    "program.mlir", "reference.hpp", "contract.json", "dag.json",
    "api_context.json", "candidate.schema.json", "PROMPT.md",
})
RUNTIME_FILES = {
    "seal-ckks": "seal_runtime.hpp",
    "openfhe-cross": "openfhe_runtime.hpp",
    "phantom-ckks": "phantom_runtime.hpp",
}

CANDIDATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CrossFHE serial schedule candidate",
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "status", "mode", "task_hash", "schedule", "rationale"],
            "properties": {
                "schema_version": {"type": "integer", "const": 1},
                "status": {"const": "proposal"},
                "mode": {"const": "schedule-v1"},
                "task_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "schedule": {
                    "type": "array", "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1, "maxLength": MAX_NODE_ID_LENGTH},
                },
                "rationale": {
                    "type": "array", "maxItems": MAX_RATIONALE_ITEMS,
                    "items": {"type": "string", "maxLength": MAX_TEXT_LENGTH},
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "status", "task_hash", "reason"],
            "properties": {
                "schema_version": {"type": "integer", "const": 1},
                "status": {"const": "no_candidate"},
                "task_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "reason": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_LENGTH},
            },
        },
    ],
}


def canonical_bytes(value: Any) -> bytes:
    """Encode finite JSON with sorted keys and one deterministic UTF-8 layout."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, RecursionError, UnicodeError) as error:
        raise ValueError(f"cannot encode canonical JSON: {error}") from error


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _nonfinite(value)
    return result


def parse_json(text: str) -> Any:
    """Parse one finite JSON value, with no prose, code fence, or duplicates.

    Network response limits belong to the provider that reads the response.
    """
    try:
        return json.loads(text, object_pairs_hook=_pairs,
                          parse_constant=_nonfinite, parse_float=_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"invalid JSON: {error}") from error


def read_json(path: str | Path) -> Any:
    """Read local UTF-8 JSON through the shared parser."""
    return parse_json(Path(path).read_text(encoding="utf-8"))


def seal_task(task: dict[str, Any]) -> dict[str, Any]:
    """Return a detached task with a hash over all fields except task_hash."""
    if not isinstance(task, dict):
        raise ValueError("task must be a JSON object")
    result = copy.deepcopy(task)
    result.pop("task_hash", None)
    result["task_hash"] = sha256(canonical_bytes(result))
    return result


def _integer(value: Any, lower: int, upper: int | None = None) -> bool:
    return type(value) is int and value >= lower and (upper is None or value <= upper)


def _finite(value: Any, *, positive: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    # math.isfinite converts ints to double and may overflow on adversarial JSON.
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def validate_policy(policy: dict[str, Any], backend: str) -> None:
    """Validate the single policy shared by task preparation and execution."""
    for name in ("atol", "rtol", "input_bound", "comparison_scale"):
        if not _finite(policy.get(name), positive=name in {"input_bound", "comparison_scale"}):
            raise ValueError(f"validation_policy {name} must be finite and "
                             + ("positive" if name in {"input_bound", "comparison_scale"} else "nonnegative"))
    limits = {"max_attempts": (1, 10), "repetitions": (1, 30),
              "warmups": (0, 10), "seed": (0, 2**32 - 1)}
    for name, (lower, upper) in limits.items():
        if not _integer(policy.get(name), lower, upper):
            raise ValueError(f"validation_policy {name} must be an integer between {lower} and {upper}")
    insecure = policy.get("insecure_demo")
    if type(insecure) is not bool or insecure and backend != "openfhe-cross":
        raise ValueError("validation_policy insecure_demo must be a boolean and is available only for OpenFHE")
    profile = "INSECURE_TOY" if insecure else (
        "HEStd_128_classic+STD128" if backend == "openfhe-cross" else "tc128")
    if policy.get("security_profile") != profile:
        raise ValueError("validation_policy security_profile disagrees with backend/insecure_demo")


def _validate_tensor_mapping(task: dict[str, Any], dag: dict[str, Any]) -> None:
    mapping = task.get("tensor_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("task tensor_mapping must be an object")
    for direction in ("inputs", "outputs"):
        tensors = mapping.get(direction)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f"tensor_mapping {direction} must be a nonempty array")
        offset, names = 0, set()
        for tensor in tensors:
            if not isinstance(tensor, dict):
                raise ValueError(f"invalid tensor_mapping {direction} entry")
            name, shape = tensor.get("name"), tensor.get("shape")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"invalid or duplicate tensor_mapping {direction} name")
            names.add(name)
            if not isinstance(shape, list) or any(not _integer(dim, 1) for dim in shape):
                raise ValueError(f"tensor_mapping {direction} shape must contain positive integers")
            count = tensor.get("count")
            if not _integer(count, 1) or count != math.prod(shape):
                raise ValueError(f"tensor_mapping {direction} count disagrees with shape")
            if not _integer(tensor.get("offset"), 0) or tensor["offset"] != offset:
                raise ValueError(f"tensor_mapping {direction} offsets must be contiguous")
            offset += count
        singular = "input" if direction == "inputs" else "output"
        contract = task["contract"]
        key = f"{singular}_elements"
        if key not in contract and contract.get("layout") != "periodic":
            key = f"{singular}_count"
        if key in contract and (not _integer(contract[key], 1) or contract[key] != offset):
            raise ValueError(f"tensor_mapping {direction} disagrees with contract {key}")
        count_key = f"{singular}_count"
        if count_key in contract and (not _integer(contract[count_key], 1)
                                      or contract[count_key] != len(dag[direction])):
            raise ValueError(f"DAG {direction} disagrees with contract {count_key}")


def _validate_contract(contract: dict[str, Any]) -> None:
    for name in ("multiplicative_depth", "logic_comparisons", "openfhe_depth"):
        if not _integer(contract.get(name), 0):
            raise ValueError(f"contract {name} must be a nonnegative integer")
    resources = contract.get("resources")
    if not isinstance(resources, dict) or any(not _integer(value, 0) for value in resources.values()):
        raise ValueError("contract resources must contain nonnegative integer counts")
    for name in ("input_count", "output_count", "input_elements", "output_elements", "packed_period"):
        if name in contract and not _integer(contract[name], 1):
            raise ValueError(f"contract {name} must be a positive integer")
    if "rotation_steps" in contract:
        steps = contract["rotation_steps"]
        if not isinstance(steps, list) or any(type(step) is not int for step in steps):
            raise ValueError("contract rotation_steps must contain integers")


def validate_task(task: dict[str, Any], dag: dict[str, Any]) -> None:
    """Check task semantics once at preparation or sealed-task loading."""
    if type(task.get("schema_version")) is not int or task["schema_version"] != 2:
        raise ValueError("unsupported task schema_version")
    if task.get("mode") != "schedule-v1":
        raise ValueError("unsupported task mode")
    for key in ("contract", "validation_policy"):
        if not isinstance(task.get(key), dict):
            raise ValueError(f"task {key} must be an object")
    validate_policy(task["validation_policy"], task["backend"])
    _validate_contract(task["contract"])
    if "approximation" in task and not isinstance(task["approximation"], dict):
        raise ValueError("task approximation must be an object")
    if (not isinstance(dag, dict) or type(dag.get("schema_version")) is not int
            or dag["schema_version"] != 1 or dag.get("backend") != task["backend"]):
        raise ValueError("task DAG backend/schema mismatch")
    _validate_schedule(dag.get("reference_schedule"), dag)
    _validate_tensor_mapping(task, dag)


def verify_task(task_dir: str | Path) -> dict[str, Any]:
    """Load local task files, checking their hashes and shared task semantics."""
    directory = Path(task_dir)
    task = read_json(directory / "task.json")
    if not isinstance(task, dict):
        raise ValueError("task must be a JSON object")
    backend = task.get("backend")
    if not isinstance(backend, str) or backend not in RUNTIME_FILES:
        raise ValueError("unsupported task backend")
    if seal_task(task)["task_hash"] != task.get("task_hash"):
        raise ValueError("task hash mismatch")
    hashes = task.get("context_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("context_hashes must be a nonempty object")
    for name, expected in hashes.items():
        if name not in CONTEXT_FILES:
            raise ValueError(f"context path is not allowlisted: {name}")
        if sha256((directory / name).read_bytes()) != expected:
            raise ValueError(f"context hash mismatch: {name}")
    required = REQUIRED_CONTEXT_FILES | {RUNTIME_FILES[backend]}
    if not required <= hashes.keys():
        raise ValueError(f"task is missing required context hashes: {sorted(required - hashes.keys())}")
    if (hashes.keys() & set(RUNTIME_FILES.values())) != {RUNTIME_FILES[backend]}:
        raise ValueError("task runtime context disagrees with backend")
    if canonical_bytes(read_json(directory / "contract.json")) != canonical_bytes(task["contract"]):
        raise ValueError("task contract differs from sealed contract.json")
    if canonical_bytes(read_json(directory / "candidate.schema.json")) != canonical_bytes(CANDIDATE_SCHEMA):
        raise ValueError("sealed candidate schema differs from schedule-v1 protocol")
    dag = read_json(directory / "dag.json")
    validate_task(task, dag)
    api = read_json(directory / "api_context.json")
    if not isinstance(api, dict) or api.get("backend") != backend:
        raise ValueError("task API context backend mismatch")
    return task


def _node_id(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= MAX_NODE_ID_LENGTH


def _validate_schedule(schedule: Any, dag: dict[str, Any]) -> None:
    if not isinstance(schedule, list) or not all(_node_id(item) for item in schedule):
        raise ValueError("schedule must be an array of node IDs")
    if not isinstance(dag, dict) or not isinstance(dag.get("nodes"), list):
        raise ValueError("DAG nodes must be an array")
    inputs = dag.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("DAG inputs must be an array")
    input_ids = []
    for item in inputs:
        if not isinstance(item, dict) or not _node_id(item.get("id")):
            raise ValueError("invalid DAG input ID")
        input_ids.append(item["id"])
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("duplicate DAG input ID")
    nodes = {}
    for node in dag["nodes"]:
        if not isinstance(node, dict) or not _node_id(node.get("id")):
            raise ValueError("invalid DAG node ID")
        node_id = node["id"]
        if node_id in nodes or node_id in input_ids:
            raise ValueError(f"duplicate DAG value ID: {node_id}")
        nodes[node_id] = node
    if len(schedule) != len(set(schedule)):
        raise ValueError("duplicate schedule node")
    if set(schedule) != set(nodes):
        missing, unknown = sorted(set(nodes) - set(schedule)), sorted(set(schedule) - set(nodes))
        raise ValueError(f"schedule node set mismatch: missing={missing}, unknown={unknown}")
    defined = set(input_ids)
    scheduled = set()
    for node_id in schedule:
        node = nodes[node_id]
        dependencies, operands = node.get("dependencies"), node.get("operands")
        if not isinstance(dependencies, list) or not all(_node_id(item) for item in dependencies):
            raise ValueError(f"invalid DAG dependencies: {node_id}")
        if not isinstance(operands, list) or not all(_node_id(item) for item in operands):
            raise ValueError(f"invalid DAG operands: {node_id}")
        if len(set(dependencies)) != len(dependencies) or any(item not in nodes for item in dependencies):
            raise ValueError(f"invalid DAG dependency node: {node_id}")
        if any(item in nodes and item not in dependencies for item in operands):
            raise ValueError(f"DAG omits data dependency: {node_id}")
        if any(item not in scheduled for item in dependencies):
            raise ValueError(f"dependency used before definition: {node_id}")
        if any(item not in defined for item in operands):
            raise ValueError(f"operand used before definition: {node_id}")
        scheduled.add(node_id)
        defined.add(node_id)
    if not isinstance(dag.get("outputs"), list) or any(
            not _node_id(item) or item not in defined for item in dag["outputs"]):
        raise ValueError("invalid DAG output value")


def validate_candidate(candidate: Any, task: dict[str, Any], dag: dict[str, Any]) -> dict[str, Any]:
    """Reject every edit outside a complete legal serial node permutation."""
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a JSON object")
    if type(candidate.get("schema_version")) is not int or candidate["schema_version"] != 1:
        raise ValueError("unsupported candidate schema_version")
    status = candidate.get("status")
    if status == "proposal":
        expected = {"schema_version", "status", "mode", "task_hash", "schedule", "rationale"}
    elif status == "no_candidate":
        expected = {"schema_version", "status", "task_hash", "reason"}
    else:
        raise ValueError("unsupported candidate status")
    if set(candidate) != expected:
        raise ValueError(f"candidate fields mismatch: missing={sorted(expected - set(candidate))}, "
                         f"unknown={sorted(set(candidate) - expected)}")
    digest = candidate["task_hash"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or digest != task.get("task_hash"):
        raise ValueError("candidate task hash mismatch")
    if status == "no_candidate":
        reason = candidate["reason"]
        if not isinstance(reason, str) or not 0 < len(reason) <= MAX_TEXT_LENGTH:
            raise ValueError("no_candidate reason must be a bounded nonempty string")
    else:
        if candidate["mode"] != "schedule-v1":
            raise ValueError("unsupported candidate mode")
        rationale = candidate["rationale"]
        if (not isinstance(rationale, list) or len(rationale) > MAX_RATIONALE_ITEMS
                or any(not isinstance(item, str) or len(item) > MAX_TEXT_LENGTH for item in rationale)):
            raise ValueError("rationale must be a bounded array of strings")
        if (not isinstance(dag, dict) or type(dag.get("schema_version")) is not int
                or dag["schema_version"] != 1 or dag.get("backend") != task.get("backend")):
            raise ValueError("candidate DAG backend/schema mismatch")
        _validate_schedule(candidate["schedule"], dag)
    return copy.deepcopy(candidate)
