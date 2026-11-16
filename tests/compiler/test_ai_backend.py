import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("prepare_ai_backend",Path(__file__).parents[2]/"scripts/prepare_ai_backend.py")
prepare_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_module)


def artifact(tmp_path):
    path=tmp_path/"artifact";path.mkdir()
    contract={"multiplicative_depth":1,"logic_comparisons":0,"openfhe_depth":1,"resources":{}}
    files={"model.hpp":"// public evaluator\n","04_optimized.mlir":"module {}\n","resources_after.json":json.dumps(contract)}
    hashes={}
    for name,text in files.items():
        (path/name).write_text(text);hashes[name]=hashlib.sha256(text.encode()).hexdigest()
    (path/"compilation_status.json").write_text('{"state":"ready"}')
    (path/"manifest.json").write_text(json.dumps({"backend":"seal-ckks","model_sha256":"test",
        "optimization":"scalar","resource_analysis":{"after":contract},"artifact_hashes":hashes}))
    (path/"secret.bin").write_text("must not be exported")
    return path


def test_task_is_public_and_never_claims_execution(tmp_path):
    source=artifact(tmp_path);dest=tmp_path/"task"
    task=prepare_module.prepare(source,dest)
    assert task["status"]=="prepared_unvalidated"
    assert not task["promotion_allowed"] and not task["validation_performed"]
    assert task["max_repair_rounds"]==10
    assert not (dest/"secret.bin").exists()
    for name,digest in task["context_hashes"].items():
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==digest


def test_changed_public_context_is_rejected(tmp_path):
    source=artifact(tmp_path);(source/"model.hpp").write_text("tampered")
    with pytest.raises(ValueError,match="hash mismatch"):
        prepare_module.prepare(source,tmp_path/"task")


def test_task_cannot_overwrite_artifact(tmp_path):
    source=artifact(tmp_path)
    with pytest.raises(ValueError,match="separate"):
        prepare_module.prepare(source,source/"ai")


def test_task_cannot_overwrite_existing_output(tmp_path):
    source=artifact(tmp_path);dest=tmp_path/"task";dest.mkdir();(dest/"keep").write_text("keep")
    with pytest.raises(ValueError,match="empty"):
        prepare_module.prepare(source,dest)
    assert (dest/"keep").read_text()=="keep"
