import json
from pathlib import Path

import numpy as np
import pytest
from crossfhe.demos import make_demo
from crossfhe.pipeline import compile_torch,read_config,run_encrypted,command,environment

pytestmark=pytest.mark.fhe


@pytest.mark.parametrize("name,backend",[
    ("affine","seal-ckks"),("square-mlp","seal-ckks"),("tiny-cnn","seal-ckks"),
    ("poly-attention","seal-ckks"),("affine","openfhe-cross"),
    ("relu","openfhe-cross"),("relu-mlp","openfhe-cross"),("maxpool","openfhe-cross"),
])
def test_models_match_pytorch_with_actual_fhe(tmp_path,name,backend):
    config=read_config()
    model,inputs=make_demo(name)
    artifact,arrays,reference=compile_torch(model,inputs,tmp_path/name,config=config,backend=backend)
    _,report=run_encrypted(artifact,arrays,reference=reference,config=config,
                          insecure_demo=backend=="openfhe-cross",atol=0.001,rtol=0.001)
    assert report["validation"]["passed"]
    assert report["server_has_no_secret_key"]
    assert (artifact/"03_fhe.mlir").read_bytes()==(artifact/"04_optimized.mlir").read_bytes()
    manifest=json.loads((artifact/"manifest.json").read_text())
    if manifest["logic_comparisons"]:
        runtime=report["phases"]["evaluate"]["runtime"]
        assert runtime["ckks_to_logic"]==runtime["logic_bootstrap"]==runtime["logic_to_ckks"]==manifest["logic_comparisons"]
        assert "heir.depack" in (artifact/"03_fhe.mlir").read_text()
    if backend=="openfhe-cross":
        assert report["phases"]["keygen"]["runtime"]["security_profile"]=="INSECURE_TOY"
