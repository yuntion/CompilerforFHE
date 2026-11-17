"""Shared artifact verification and deterministic report output."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from backend.execution import backend_context
from .protocol import canonical_bytes, read_json


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    Path(path).write_bytes(json_bytes(value))


def verify_artifact(artifact, task, config):
    """Check a reference before generation and immediately before execution."""
    if read_json(artifact/"compilation_status.json").get("state") != "ready":
        raise ValueError("reference artifact is not ready")
    manifest = read_json(artifact/"manifest.json")
    suffix = ".exe" if os.name == "nt" else ""
    allowed = {"model.onnx", "model.hpp", "03_fhe.mlir", "04_optimized.mlir", "dag.json",
               "resources_before.json", "resources_after.json", f"build/bin/model-runner{suffix}"}
    hashes = manifest.get("artifact_hashes", {})
    if not isinstance(hashes, dict) or set(hashes) != allowed:
        raise ValueError("reference artifact lacks the current complete hash manifest; compile again")
    for name, expected in hashes.items():
        path = artifact/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(artifact) or digest(path) != expected:
            raise ValueError(f"reference artifact integrity failure: {name}")
    if (manifest["backend"] != task["backend"] or manifest["model_sha256"] != task["model_sha256"]
            or manifest.get("backend_context") != task["backend_context"]):
        raise ValueError("task does not belong to this reference artifact")
    if (canonical_bytes(task["tensor_mapping"]) != canonical_bytes(
            {"inputs": manifest["inputs"], "outputs": manifest["outputs"]})
            or canonical_bytes(task.get("approximation", {})) != canonical_bytes(manifest.get("approximation", {}))
            or task["optimization"] != manifest["optimization"]
            or task["model_sha256"] != hashes["model.onnx"]):
        raise ValueError("task tensor mapping, approximation or model metadata differ from the reference")
    if manifest.get("resource_analysis", {}).get("after") != task["contract"]:
        raise ValueError("reference resource contract differs from the task")
    if backend_context(task["backend"], config) != task["backend_context"]:
        raise ValueError("compiler, backend sources, toolchain or libraries changed; compile and prepare again")
    for source, exported in (("model.hpp", "reference.hpp"), ("04_optimized.mlir", "program.mlir"),
                             ("resources_after.json", "contract.json"), ("dag.json", "dag.json")):
        if hashes[source] != task["context_hashes"][exported]:
            raise ValueError(f"task and reference context differ: {source}")
    return manifest
