import json
from pathlib import Path

import numpy as np
import pytest
from crossfhe.demos import make_demo
from crossfhe.pipeline import compile_torch,run_encrypted,command,environment

pytestmark = [pytest.mark.fhe, pytest.mark.native, pytest.mark.torch]


@pytest.mark.parametrize("name,backend",[
    pytest.param("affine", "seal-ckks", marks=pytest.mark.seal),
    pytest.param("square-mlp", "seal-ckks", marks=pytest.mark.seal),
    pytest.param("tiny-cnn", "seal-ckks", marks=pytest.mark.seal),
    pytest.param("poly-attention", "seal-ckks", marks=pytest.mark.seal),
    pytest.param("affine", "openfhe-cross", marks=pytest.mark.openfhe),
    pytest.param("relu", "openfhe-cross", marks=pytest.mark.openfhe),
    pytest.param("relu-mlp", "openfhe-cross", marks=pytest.mark.openfhe),
    pytest.param("maxpool", "openfhe-cross", marks=pytest.mark.openfhe),
])
def test_models_match_pytorch_with_actual_fhe(tmp_path,config,name,backend):
    model,inputs=make_demo(name)
    artifact,arrays,reference=compile_torch(model,inputs,tmp_path/name,config=config,backend=backend)
    _,report=run_encrypted(artifact,arrays,reference=reference,config=config,
                          insecure_demo=backend=="openfhe-cross",atol=0.001,rtol=0.001)
    assert report["validation"]["passed"]
    assert report["server_has_no_secret_key"]
    manifest=json.loads((artifact/"manifest.json").read_text())
    before,after=(manifest["resource_analysis"][name] for name in ("before","after"))
    assert after["multiplicative_depth"]<=before["multiplicative_depth"]
    assert after["logic_comparisons"]<=before["logic_comparisons"]
    assert manifest["optimization"]=="scalar"
    if manifest["logic_comparisons"]:
        runtime=report["phases"]["evaluate"]["runtime"]
        assert runtime["ckks_to_logic"]==runtime["logic_bootstrap"]==runtime["logic_to_ckks"]==manifest["logic_comparisons"]
        assert "heir.depack" in (artifact/"03_fhe.mlir").read_text()
    if backend=="openfhe-cross":
        assert report["phases"]["keygen"]["runtime"]["security_profile"]=="INSECURE_TOY"
