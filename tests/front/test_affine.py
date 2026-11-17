"""Affine parsing preserves model semantics independently of execution layouts."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from crossfhe.affine import import_affine_onnx


def save_affine_model(tmp_path, matrix, bias, dtype):
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "x"], ["product"]),
         helper.make_node("Add", ["product", "b"], ["y"])],
        "public-affine",
        [helper.make_tensor_value_info("x", dtype, [matrix.shape[1]])],
        [helper.make_tensor_value_info("y", dtype, [matrix.shape[0]])],
        [numpy_helper.from_array(matrix, "A"), numpy_helper.from_array(bias, "b")],
    )
    path = tmp_path / "affine.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    return path


def test_affine_import_preserves_coefficients_and_model_metadata_without_layout(tmp_path):
    path = save_affine_model(
        tmp_path, np.array([[.5, -1, 2], [1.5, .25, -.5]], dtype=np.float32),
        np.array([.25, -.75], dtype=np.float32), TensorProto.FLOAT,
    )
    matrix, bias, manifest = import_affine_onnx(path)
    np.testing.assert_array_equal(matrix, np.array([[.5, -1, 2], [1.5, .25, -.5]], dtype=np.float64))
    np.testing.assert_array_equal(bias, np.array([.25, -.75], dtype=np.float64))
    np.testing.assert_array_equal(matrix @ np.array([2, -1, .5]) + bias, np.array([3.25, 1.75]))
    assert matrix.dtype == bias.dtype == np.dtype(np.float64)
    assert manifest["inputs"] == [{"name": "x", "shape": [3], "offset": 0,
                                   "count": 3, "onnx_dtype": TensorProto.FLOAT}]
    assert manifest["outputs"] == [{"name": "y", "shape": [2], "offset": 0, "count": 2}]
    assert manifest["scalar_operations"] == manifest["logic_comparisons"] == 0
    assert manifest["model_parameters"] == "public"
    assert not any("packed" in name or "layout" in name for name in manifest)


def test_affine_import_accepts_model_dimensions_above_execution_slot_limits(tmp_path):
    weights = np.ones((1, 16385), dtype=np.float64)
    path = save_affine_model(tmp_path, weights, np.zeros(1), TensorProto.DOUBLE)
    matrix, bias, manifest = import_affine_onnx(path, max_elements=20000)
    np.testing.assert_array_equal(matrix, weights)
    np.testing.assert_array_equal(bias, np.zeros(1))
    assert manifest["inputs"] == [{"name": "x", "shape": [16385], "offset": 0,
                                   "count": 16385, "onnx_dtype": TensorProto.DOUBLE}]
    assert manifest["outputs"] == [{"name": "y", "shape": [1], "offset": 0, "count": 1}]
    assert not any("packed" in name or "layout" in name for name in manifest)
