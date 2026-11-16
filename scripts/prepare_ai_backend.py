"""Prepare public, hash-checked compiler context for an AI backend candidate.

This command neither contacts a model nor executes generated code. A task remains
prepared_unvalidated until a separate real compiler and differential verifier run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = ("model.hpp", "04_optimized.mlir", "resources_after.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(artifact: Path, output: Path) -> dict:
    artifact, output = artifact.resolve(), output.resolve()
    if not (artifact/"compilation_status.json").is_file() or json.loads(
            (artifact/"compilation_status.json").read_text()).get("state") != "ready":
        raise ValueError("artifact must be compiled and ready")
    manifest = json.loads((artifact/"manifest.json").read_text())
    backend = manifest.get("backend")
    if backend not in ("seal-ckks", "openfhe-cross"):
        raise ValueError("only the implemented CPU backends are supported")
    contract = manifest.get("resource_analysis", {}).get("after")
    if not contract:
        raise ValueError("artifact has no native resource contract; compile with the current compiler")
    hashes = manifest.get("artifact_hashes", {})
    # An allowlist prevents exporting client inputs, private keys or arbitrary
    # paths mentioned by a modified manifest. No recursive artifact copying.
    for name in PUBLIC_FILES:
        if (artifact/name).is_symlink() or not (artifact/name).resolve().is_relative_to(artifact):
            raise ValueError(f"public context must be a regular artifact file: {name}")
        if name not in hashes or digest(artifact/name) != hashes[name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    if json.loads((artifact/"resources_after.json").read_text()) != contract:
        raise ValueError("manifest and native resource contract differ")
    if output == artifact or output.is_relative_to(artifact):
        raise ValueError("task output must be separate from the executable artifact")
    if output.exists() and any(output.iterdir()):
        raise ValueError("task output must be empty; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    runtime_name = "seal_runtime.hpp" if backend == "seal-ckks" else "openfhe_runtime.hpp"
    runtime = ROOT/"backend"/("seal" if backend=="seal-ckks" else "openfhe")/runtime_name
    files = {"reference.hpp": (artifact/"model.hpp").read_bytes(),
             "program.mlir": (artifact/"04_optimized.mlir").read_bytes(),
             "contract.json": (artifact/"resources_after.json").read_bytes(),
             runtime_name: runtime.read_bytes()}
    task = {
        "schema_version": 1,
        "status": "prepared_unvalidated",
        "backend": backend,
        "target": "x86-cpu",
        "model_sha256": manifest["model_sha256"],
        "library_revision": manifest.get("upstream_revisions", {}).get("seal" if backend=="seal-ckks" else "openfhe"),
        "optimization": manifest["optimization"],
        "approximation": manifest.get("approximation", {}),
        "contract": contract,
        "tensor_mapping": {"inputs": manifest.get("inputs", []), "outputs": manifest.get("outputs", [])},
        "max_repair_rounds": 10,
        "candidate_scope": "evaluator scheduling and target runtime API adaptation",
        "immutable": ["input/output ABI and tensor mapping", "FHE algorithm and switch encoding",
                      "depth/scale/security configuration", "key ownership and client/server boundary"],
        "required_validation": ["isolated real C++ compilation", "native reference differential tests",
            "encrypted boundary/random tests with independent client decryption",
            "unchanged security profile and precision", "artifact integrity",
            "recorded timing and memory; compare at fixed parameters"],
        "context_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "validation_performed": False,
        "promotion_allowed": False,
    }
    for name, data in files.items():
        (output/name).write_bytes(data)
    (output/"task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2)+"\n")
    (output/"PROMPT.md").write_text(
        "# Backend candidate task\n\n"
        "Read task.json, contract.json, program.mlir, reference.hpp and the pinned runtime header.\n"
        "Propose a candidate evaluator using the same crossfhe_generated ABI. Preserve the typed FHE algorithm, "
        "scheme switching encoding, output order, security parameters and key ownership. "
        "Do not introduce decryption, parameter downgrades, secret-dependent branches or new library capabilities.\n\n"
        "Return candidate.hpp and rationale.json listing changed scheduling/API choices, referenced API signatures, "
        "expected resource changes and assumptions. All performance claims require measurements. "
        "A proposal is unvalidated; only the independent compiler/test runner can accept it. "
        "Validation failures feed at most 10 repair rounds. Keep the deterministic reference if any required check fails.\n")
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        task = prepare(args.artifact, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot prepare AI backend task: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": task["status"],
                      "validation_performed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
