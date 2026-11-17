"""Public task preparation and sealing, without native builds or execution."""
import hashlib
import json

import pytest

from backend.ai import context as context_module
from backend.ai import task as task_module
from backend.ai.artifacts import digest
from backend.ai.protocol import verify_task


def artifact(tmp_path):
    path = tmp_path / "artifact"
    path.mkdir()
    contract = {"multiplicative_depth": 1, "logic_comparisons": 0, "openfhe_depth": 1,
                "resources": {"operations": 1}}
    # This is public compiler-context fixture data. It does not claim that the
    # evaluator was compiled or that an encrypted candidate was executed.
    dag = {
        "schema_version": 1, "backend": "seal-ckks",
        "inputs": [{"id": "arg0", "type": "!heir.rlwecipher<1 x f64>"}],
        "nodes": [{"id": "n0", "op": "heir.rlweadd", "operands": ["arg0", "arg0"],
                   "dependencies": [], "effect": "pure", "type": "!heir.rlwecipher<1 x f64>"}],
        "outputs": ["n0"], "reference_schedule": ["n0"],
    }
    files = {"model.hpp": "// public evaluator fixture\n", "04_optimized.mlir": "module {}\n",
             "resources_after.json": json.dumps(contract), "dag.json": json.dumps(dag)}
    hashes = {}
    for name, text in files.items():
        (path / name).write_text(text)
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()
    sources = ["backend/CMakeLists.txt", "backend/seal/CMakeLists.txt",
               "backend/seal/seal_runtime.hpp", "backend/seal/runner.cpp"]
    context = {
        "backend": "seal-ckks", "library_revision": "test", "build_type": "Release",
        "source_hashes": {name: digest(task_module.ROOT / name) for name in sources},
    }
    (path / "compilation_status.json").write_text('{"state":"ready"}')
    (path / "manifest.json").write_text(json.dumps({
        "backend": "seal-ckks", "model_sha256": hashlib.sha256(b"fixture-model").hexdigest(),
        "optimization": "scalar", "resource_analysis": {"after": contract}, "artifact_hashes": hashes,
        "backend_context": context,
        "inputs": [{"name": "x", "shape": [1], "offset": 0, "count": 1}],
        "outputs": [{"name": "y", "shape": [1], "offset": 0, "count": 1}],
    }))
    (path / "secret.bin").write_text("must not be exported")
    return path


def rewrite_manifest(source, change):
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


def test_task_is_public_and_never_claims_execution(tmp_path):
    source, dest = artifact(tmp_path), tmp_path / "task"
    task = task_module.prepare(source, dest)
    assert task["schema_version"] == 2 and task["mode"] == "schedule-v1"
    assert task["status"] == "prepared_unvalidated"
    assert not task["promotion_allowed"] and not task["validation_performed"]
    assert "max_repair_rounds" not in task and "max_attempts" not in task
    assert task["validation_policy"]["max_attempts"] == 10
    assert task["library_revision"] == "test"
    assert task["validation_policy"]["security_profile"] == "tc128"
    assert not (dest / "secret.bin").exists()
    assert set(task["context_hashes"]) == {
        "reference.hpp", "program.mlir", "contract.json", "dag.json", "seal_runtime.hpp",
        "api_context.json", "candidate.schema.json", "PROMPT.md",
    }
    for name, digest in task["context_hashes"].items():
        assert hashlib.sha256((dest / name).read_bytes()).hexdigest() == digest
    api_context = json.loads((dest / "api_context.json").read_text())
    mapping = api_context["function_mapping"]
    assert mapping["authority"] == "runtime_mapping"
    assert mapping["runtime_header"] == "backend/seal/seal_runtime.hpp"
    assert mapping["runtime_sha256"] == task["context_hashes"]["seal_runtime.hpp"]
    assert mapping["runtime_hash_matches_task"] is True
    assert mapping["library_revision"] != "test"
    assert mapping["library_revision_matches_task"] is False
    assert any(item["operation"] == "rotate" for item in mapping["mappings"])
    assert verify_task(dest) == task


def test_changed_public_context_is_rejected(tmp_path):
    source = artifact(tmp_path)
    (source / "model.hpp").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        task_module.prepare(source, tmp_path / "task")


def test_task_cannot_overwrite_artifact(tmp_path):
    source = artifact(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        task_module.prepare(source, source / "ai")


def test_task_cannot_overwrite_existing_output(tmp_path):
    source, dest = artifact(tmp_path), tmp_path / "task"
    dest.mkdir()
    (dest / "keep").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        task_module.prepare(source, dest)
    assert (dest / "keep").read_text() == "keep"


@pytest.mark.parametrize("change,diagnostic", [
    (lambda manifest: manifest["backend_context"].update(backend="openfhe-cross"), "backend context mismatch"),
    (lambda manifest: manifest["backend_context"]["source_hashes"].update(
        {"backend/seal/seal_runtime.hpp": "0" * 64}), "backend source changed"),
    (lambda manifest: manifest.update(backend="tfhe"), "no standalone TFHE backend"),
])
def test_backend_and_runtime_must_match_the_compiled_context(tmp_path, change, diagnostic):
    source = artifact(tmp_path)
    rewrite_manifest(source, change)
    with pytest.raises(ValueError, match=diagnostic):
        task_module.prepare(source, tmp_path / "task")
    assert not (tmp_path / "task").exists()


@pytest.mark.parametrize("extra_path", [
    "backend/openfhe/openfhe_runtime.hpp", "backend/seal/private-key.bin", "../../secret.bin",
])
def test_source_allowlist_rejects_extra_paths(tmp_path, extra_path):
    source = artifact(tmp_path)
    rewrite_manifest(source, lambda manifest: manifest["backend_context"]["source_hashes"].update(
        {extra_path: "0" * 64}))
    with pytest.raises(ValueError, match="fixed backend allowlist"):
        task_module.prepare(source, tmp_path / "task")
    assert not (tmp_path / "task").exists()


@pytest.mark.parametrize("policy,diagnostic", [
    ({"atol": float("nan")}, "atol must be finite"),
    ({"rtol": -0.1}, "rtol must be finite"),
    ({"input_bound": 0}, "input_bound must be finite and positive"),
    ({"max_attempts": 11}, "max_attempts must be an integer"),
    ({"insecure_demo": True}, "insecure_demo .*available only for OpenFHE"),
])
def test_invalid_validation_policy_does_not_create_a_task(tmp_path, policy, diagnostic):
    source = artifact(tmp_path)
    with pytest.raises(ValueError, match=diagnostic):
        task_module.prepare(source, tmp_path / "task", **policy)
    assert not (tmp_path / "task").exists()


@pytest.mark.parametrize("target", ["task.json", "dag.json", "api_context.json"])
def test_sealed_task_dag_and_function_mapping_tampering_is_rejected(tmp_path, target):
    source, dest = artifact(tmp_path), tmp_path / "task"
    task_module.prepare(source, dest)
    path = dest / target
    document = json.loads(path.read_text())
    if target == "task.json":
        document["validation_policy"]["atol"] = 123.0
    elif target == "dag.json":
        document["outputs"] = ["arg0"]
    else:
        document["function_mapping"]["sha256"] = "0" * 64
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="task hash mismatch|context hash mismatch: (dag|api_context).json"):
        verify_task(dest)


def test_function_mapping_for_another_runtime_is_rejected_before_task_export(tmp_path, monkeypatch):
    source, dest = artifact(tmp_path), tmp_path / "task"
    original_load = context_module.load_function_mapping

    def changed_mapping(backend, **kwargs):
        mapping = original_load(backend, **kwargs)
        mapping["runtime_sha256"] = "0" * 64
        return mapping

    # This isolates public-context configuration validation. No native compiler,
    # encrypted runner or decryption result is replaced with a simulated pass.
    monkeypatch.setattr(context_module, "load_function_mapping", changed_mapping)
    with pytest.raises(ValueError, match="function mapping runtime hash mismatch"):
        task_module.prepare(source, dest)
    assert not dest.exists()
