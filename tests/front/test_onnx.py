import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper
import pytest

from crossfhe.frontend import FrontendError,import_onnx


def write_model(tmp_path,nodes,inputs,outputs,initializers=()):
    model=helper.make_model(helper.make_graph(nodes,"test",inputs,outputs,list(initializers)),
                            opset_imports=[helper.make_opsetid("",17)])
    path=tmp_path/"model.onnx"; onnx.save(model,path)
    return path


def vi(name,shape,dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name,dtype,shape)


@pytest.mark.parametrize("op",["Softmax","Sigmoid","Tanh","Exp","Sqrt"])
def test_unsupported_secret_nonlinear_has_diagnostic(tmp_path,op):
    path=write_model(tmp_path,[helper.make_node(op,["x"],["y"])],[vi("x",[1,2])],[vi("y",[1,2])])
    with pytest.raises(FrontendError,match=f"{op} has no exact implemented FHE lowering"):
        import_onnx(path)


def test_encrypted_division_rejected(tmp_path):
    path=write_model(tmp_path,[helper.make_node("Div",["x","z"],["y"])],
                     [vi("x",[2]),vi("z",[2])],[vi("y",[2])])
    with pytest.raises(FrontendError,match="Div denominator must be public"):
        import_onnx(path)


@pytest.mark.parametrize("shape",[["batch",2],[0,2]])
def test_dynamic_and_zero_input_shapes_rejected(tmp_path,shape):
    path=write_model(tmp_path,[helper.make_node("Identity",["x"],["y"])],[vi("x",shape)],[vi("y",shape)])
    with pytest.raises(FrontendError,match="positive static shape"):
        import_onnx(path)


def test_input_dtype_policy(tmp_path):
    path=write_model(tmp_path,[helper.make_node("Identity",["x"],["y"])],
                     [vi("x",[2],TensorProto.INT64)],[vi("y",[2],TensorProto.INT64)])
    with pytest.raises(FrontendError,match="float32 or float64"): import_onnx(path)


def test_relu_preserves_comparison_and_select(tmp_path):
    path=write_model(tmp_path,[helper.make_node("Relu",["x"],["y"])],[vi("x",[2])],[vi("y",[2])])
    mlir,manifest=import_onnx(path)
    assert mlir.count("arith.cmpf olt")==2
    assert mlir.count("arith.select")==2
    assert manifest["logic_comparisons"]==2
    assert manifest["outputs"][0]["shape"]==[2]
    assert "CKKS" not in manifest["numeric_semantics"]
    assert "layout" not in manifest


def test_scalar_limit_rejected_before_large_lowering(tmp_path):
    path=write_model(tmp_path,[helper.make_node("Mul",["x","x"],["y"])],[vi("x",[4])],[vi("y",[4])])
    with pytest.raises(FrontendError,match="operation limit"): import_onnx(path,max_operations=2)


def test_public_shape_chain_and_tensor_offsets(tmp_path):
    nodes=[helper.make_node("Shape",["x"],["shape"]),helper.make_node("Reshape",["x","shape"],["y"])]
    path=write_model(tmp_path,nodes,[vi("x",[2,3])],[vi("y",[2,3])])
    _,manifest=import_onnx(path)
    assert manifest["inputs"][0]["count"]==6
    assert manifest["outputs"][0]["count"]==6
    assert manifest["scalar_operations"]==0


def test_nonfinite_coefficient_rejected(tmp_path):
    constant=numpy_helper.from_array(np.array([np.inf],dtype=np.float32),"w")
    path=write_model(tmp_path,[helper.make_node("Mul",["x","w"],["y"])],
                     [vi("x",[1])],[vi("y",[1])],[constant])
    with pytest.raises(FrontendError,match="NaN/Inf"): import_onnx(path)


def test_unsupported_custom_domain_rejected(tmp_path):
    model=helper.make_model(helper.make_graph([helper.make_node("Custom",["x"],["y"],domain="custom")],
        "test",[vi("x",[1])],[vi("y",[1])]),opset_imports=[helper.make_opsetid("",17),helper.make_opsetid("custom",1)])
    path=tmp_path/"custom.onnx"; onnx.save(model,path)
    with pytest.raises(FrontendError,match="custom ONNX domain"): import_onnx(path)


def test_scalar_reduction_keeps_scalar_output(tmp_path):
    path=write_model(tmp_path,[helper.make_node("ReduceMean",["x"],["y"],keepdims=0)],
                     [vi("x",[])],[vi("y",[])])
    _,manifest=import_onnx(path)
    assert manifest["outputs"][0]["shape"]==[]
    assert manifest["outputs"][0]["count"]==1
