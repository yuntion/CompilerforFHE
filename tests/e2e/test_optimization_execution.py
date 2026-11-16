"""FHE A/B evidence for redundancy removal, depth reduction and bounded approximation."""
import json
from pathlib import Path

import numpy as np
import pytest
from crossfhe.pipeline import compile_torch, compile_onnx, run_encrypted

pytestmark=pytest.mark.fhe


def test_cse_reduces_cipher_multiplications_without_changing_outputs(tmp_path):
    import torch
    from torch import nn
    import onnx
    from onnx import helper, TensorProto
    # Keep duplicate nodes explicitly: PyTorch's exporter already CSEs x*y+x*y.
    source=tmp_path/"repeated.onnx"
    graph=helper.make_graph([
        helper.make_node("Mul",["input_0","input_1"],["a"]),
        helper.make_node("Mul",["input_0","input_1"],["b"]),
        helper.make_node("Add",["a","b"],["output_0"])],"repeated",[
        helper.make_tensor_value_info(n,TensorProto.FLOAT,[2]) for n in ("input_0","input_1")],
        [helper.make_tensor_value_info("output_0",TensorProto.FLOAT,[2])])
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)]),source)
    x,y=torch.tensor([0.25,-0.5]),torch.tensor([0.75,0.125])
    arrays={"input_0":x.numpy(),"input_1":y.numpy()}
    reference={"output_0":(x*y+x*y).numpy()}
    records={}
    for mode in ("none","scalar"):
        path=compile_onnx(source,tmp_path/mode,backend="seal-ckks",optimization=mode)
        outputs,report=run_encrypted(path,arrays,reference=reference,atol=0.001,rtol=0.001)
        records[mode]=(json.loads((path/"manifest.json").read_text()),outputs,report)
    before,after=(records[m][0]["resource_analysis"]["after"] for m in ("none","scalar"))
    assert before["resources"]["ciphertext_multiplications"]==4
    assert after["resources"]["ciphertext_multiplications"]==2
    assert records["none"][2]["phases"]["evaluate"]["runtime"]["multiply_cipher"]==4
    assert records["scalar"][2]["phases"]["evaluate"]["runtime"]["multiply_cipher"]==2
    np.testing.assert_allclose(records["none"][1]["output_0"],records["scalar"][1]["output_0"],atol=0.001,rtol=0.001)


def test_reassociation_reduces_levels_with_real_ckks(tmp_path):
    import torch
    from torch import nn
    class Model(nn.Module):
        def forward(self,*args):
            value=args[0]
            for arg in args[1:]: value=value*arg
            return value
    inputs=tuple(torch.tensor([0.8+i*0.02]) for i in range(8))
    depths=[]
    for mode in ("none","depth"):
        path,arrays,reference=compile_torch(Model(),inputs,tmp_path/mode,backend="seal-ckks",optimization=mode)
        _,report=run_encrypted(path,arrays,reference=reference,atol=0.001,rtol=0.001)
        assert report["validation"]["passed"]
        depths.append(json.loads((path/"manifest.json").read_text())["multiplicative_depth"])
    assert depths==[7,3]


def test_approximate_relu_runs_on_seal_with_the_declared_local_budget(tmp_path):
    import torch
    from torch import nn
    class Model(nn.Module):
        def forward(self,x): return torch.relu(x)
    # The extrema of the analytic uniform error are included explicitly.
    x=torch.tensor([-1.0,-0.5,0.0,0.5,1.0])
    path,arrays,reference=compile_torch(Model(),(x,),tmp_path/"relu",backend="seal-ckks",
        relu_bound=1.0,relu_max_error=0.0625)
    outputs,report=run_encrypted(path,arrays,reference=reference,atol=0.063,rtol=0)
    manifest=json.loads((path/"manifest.json").read_text())
    assert manifest["logic_comparisons"]==0
    assert manifest["approximation"]["relu_local_error_bound"]==0.0625
    assert report["validation"]["passed"]
    np.testing.assert_allclose(outputs["output_0"],x.numpy()/2+x.numpy()**2/2+1/16,atol=0.0001,rtol=0)


def test_repeated_relu_masks_are_shared_in_real_openfhe(tmp_path):
    import torch
    from torch import nn
    import onnx
    from onnx import helper, TensorProto
    source=tmp_path/"repeated-relu.onnx"
    graph=helper.make_graph([
        helper.make_node("Relu",["input_0"],["a"]),
        helper.make_node("Relu",["input_0"],["b"]),
        helper.make_node("Add",["a","b"],["output_0"])],"repeated-relu",
        [helper.make_tensor_value_info("input_0",TensorProto.FLOAT,[2])],
        [helper.make_tensor_value_info("output_0",TensorProto.FLOAT,[2])])
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)]),source)
    x=torch.tensor([-0.5,0.75])
    arrays={"input_0":x.numpy()};reference={"output_0":(2*torch.relu(x)).numpy()}
    records=[]
    for mode in ("none","scalar"):
        path=compile_onnx(source,tmp_path/mode,backend="openfhe-cross",optimization=mode)
        _,report=run_encrypted(path,arrays,reference=reference,insecure_demo=True,atol=0.001,rtol=0.001)
        manifest=json.loads((path/"manifest.json").read_text())
        runtime=report["phases"]["evaluate"]["runtime"]
        assert runtime["logic_bootstrap"]==manifest["logic_comparisons"]
        records.append(runtime["logic_bootstrap"])
    assert records==[4,2]
