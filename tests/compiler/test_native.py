import os
from pathlib import Path

import pytest
from crossfhe.pipeline import compile_onnx,compile_torch,command,run_encrypted,export_torch
from crossfhe.demos import make_demo


@pytest.mark.native
def test_real_registered_dialects(config):
    suffix=".exe" if os.name=="nt" else ""
    tool=Path(config["native_build"])/"bin"/("crossfhe-opt"+suffix)
    text,_=command([tool,"--show-dialects"],config)
    assert "heir" in text and "scfhe" in text and "arith" in text


@pytest.mark.native
@pytest.mark.torch
def test_seal_rejects_logic_before_execution(tmp_path,config):
    model,inputs=make_demo("relu")
    with pytest.raises(ValueError,match="model needs logic FHE"):
        compile_torch(model,inputs,tmp_path/"artifact",config=config,backend="seal-ckks")


@pytest.mark.native
def test_native_invalid_mlir_is_rejected(tmp_path,config):
    suffix=".exe" if os.name=="nt" else ""
    tool=Path(config["native_build"])/"bin"/("crossfhe-opt"+suffix)
    path=tmp_path/"bad.mlir"
    path.write_text("module { func.func @model(%x: f64) -> f64 { return %missing : f64 } }")
    with pytest.raises(RuntimeError,match="undeclared SSA value"):
        command([tool,path,"--ppml-to-scfhe"],config)


@pytest.mark.parametrize("backend,diagnostic",[
    ("seal-ckks","SEAL has no logic FHE"),
    ("phantom-ckks","Phantom has no public logic FHE"),
    ("cuda","unsupported backend"),
])
@pytest.mark.native
def test_standalone_codegen_rejects_incompatible_backend(tmp_path,config,backend,diagnostic):
    suffix=".exe" if os.name=="nt" else ""
    tools=Path(config["native_build"])/"bin"
    source=Path(__file__).parent/"inputs/relu.mlir"
    lowered=tmp_path/"fhe.mlir"
    header=tmp_path/"model.hpp"
    command([tools/("crossfhe-opt"+suffix),source,"--ppml-to-scfhe",
             "--ppml-to-heir","--verify-each","-o",lowered],config)
    with pytest.raises(RuntimeError,match=diagnostic):
        command([tools/("crossfhe-translate"+suffix),lowered,
                 f"--backend={backend}","-o",header],config)
    assert not header.exists()


@pytest.mark.native
def test_incomplete_artifact_cannot_run(tmp_path,config):
    with pytest.raises(RuntimeError,match="artifact is not ready"):
        run_encrypted(tmp_path,{},config=config)


@pytest.mark.native
def test_phantom_requires_optional_configuration(tmp_path,config):
    cpu_config = {k:v for k,v in config.items() if not k.startswith("phantom_")}
    with pytest.raises(ValueError,match="bootstrap.py --with-phantom"):
        compile_onnx("unused.onnx",tmp_path/"model",config=cpu_config,backend="phantom-ckks")


@pytest.mark.parametrize("mode",["branch","item"])
@pytest.mark.torch
def test_data_dependent_python_trace_is_rejected(tmp_path,mode):
    import torch
    from torch import nn
    class Model(nn.Module):
        def forward(self,x):
            if mode=="branch":
                if x.sum()>0: return x
                return -x
            return x*x.sum().item()
    with pytest.raises(ValueError,match="would be frozen by ONNX tracing"):
        export_torch(Model(),(torch.tensor([0.25]),),tmp_path/"model.onnx")
