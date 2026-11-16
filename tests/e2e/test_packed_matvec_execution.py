"""Actual SEAL CPU SIMD against NumPy and the existing scalar compiler path."""
import json

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n, TensorProto as T
import pytest

from crossfhe.pipeline import compile_onnx, compile_torch, run_encrypted


pytestmark = pytest.mark.fhe


def make_model(path, matrix, bias, *, side="left", column=False):
    rows, cols = matrix.shape
    if side == "left":
        input_shape, output_shape = ([cols, 1], [rows, 1]) if column else ([cols], [rows])
        operands, weight = ["A", "x"], matrix
    else:
        input_shape, output_shape = [cols], [rows]
        operands, weight = ["x", "A"], matrix.T
    nodes = [h.make_node("MatMul", operands, ["v"]), h.make_node("Add", ["v", "b"], ["y"])]
    graph = h.make_graph(nodes, "packed-matvec", [h.make_tensor_value_info("x", T.DOUBLE, input_shape)],
                         [h.make_tensor_value_info("y", T.DOUBLE, output_shape)],
                         [n.from_array(weight, "A"), n.from_array(bias.reshape(output_shape), "b")])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), path)
    return input_shape, output_shape


def execute(source, directory, packing, x, expected):
    path = compile_onnx(source, directory, backend="seal-ckks", packing=packing)
    outputs, report = run_encrypted(path, {"x": x}, reference={"y": expected}, atol=1e-4, rtol=1e-5)
    assert report["validation"]["passed"]
    assert report["server_has_no_secret_key"]
    manifest = json.loads((path / "manifest.json").read_text())
    analysis = manifest["resource_analysis"]["after"]
    runtime = report["phases"]["evaluate"]["runtime"]
    resources = analysis["resources"]
    if packing != "none":
        assert runtime["input_ciphertexts"] == runtime["output_ciphertexts"] == 1
        for actual, planned in (("rotations", "rotations"), ("multiply_plain", "plaintext_multiplications"),
                                ("multiply_cipher", "ciphertext_multiplications"), ("add", "additions"),
                                ("add_plain", "plaintext_additions"), ("materializations", "materializations")):
            assert runtime[actual] == resources[planned]
        assert runtime["rescale"] == resources["plaintext_multiplications"]
        assert report["phases"]["keygen"]["runtime"]["rotation_keys"] == len(analysis["rotation_steps"])
    record = {"packing": packing, "analysis": analysis, "report": report}
    (path / "case.json").write_text(json.dumps(record, indent=2) + "\n")
    return outputs["y"], record


def test_dense_scalar_diagonal_bsgs_and_auto_ab(tmp_path):
    rng = np.random.default_rng(20261116)
    matrix = rng.uniform(-0.5, 0.5, (16, 16))
    bias = rng.uniform(-0.2, 0.2, 16)
    x = rng.uniform(-0.5, 0.5, 16)
    source = tmp_path / "dense.onnx"
    make_model(source, matrix, bias)
    records = {}
    values = {}
    for mode in ("none", "diagonal", "bsgs", "auto"):
        values[mode], records[mode] = execute(source, tmp_path / mode, mode, x, matrix @ x + bias)
    assert records["none"]["report"]["phases"]["evaluate"]["runtime"]["input_ciphertexts"] == 16
    assert records["none"]["analysis"]["resources"]["plaintext_multiplications"] == 256
    assert [records[m]["analysis"]["resources"]["rotations"] for m in ("diagonal", "bsgs", "auto")] == [15, 6, 6]
    assert all(records[m]["analysis"]["resources"]["plaintext_multiplications"] == 16 for m in ("diagonal", "bsgs", "auto"))
    for value in values.values():
        np.testing.assert_allclose(value, values["none"], atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("packing,expected_rotations", [("diagonal", 5), ("bsgs", 3), ("auto", 3)])
def test_periodic_slot_wraparound_with_repeated_encryption(tmp_path, packing, expected_rotations):
    # Five logical values use an eight-slot period inside a larger physical
    # ciphertext. In particular, row 1 must read x[0] through rotation 7.
    # Merely padding the first physical slots with zeros would lose that value.
    matrix = np.array([
        [0.0, 0.5, 0.0, -0.25, 2.0],
        [-0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.25, 0.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0, 0.0, 0.0],
        [0.75, 0.0, 0.0, -0.125, 0.0],
    ])
    bias = np.array([0.0625, -0.125, 0.25, -0.375, 0.5])
    source = tmp_path / "wraparound.onnx"
    make_model(source, matrix, bias)
    artifact = compile_onnx(source, tmp_path / "model", backend="seal-ckks", packing=packing)
    analysis = json.loads((artifact / "resources_after.json").read_text())
    assert analysis["packed_period"] == 8
    assert analysis["nonzero_diagonals"] == 5
    assert analysis["resources"]["rotations"] == expected_rotations
    assert analysis["multiplicative_depth"] == 1

    # Both boundary basis vectors isolate wraparound mistakes; the signed
    # vector also checks accumulation. Reuse the compiled model with new keys
    # and ciphertexts to ensure each run consumes its current secret input.
    inputs = {
        "first_slot": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "last_slot": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        "signed_vector": np.array([0.875, -0.375, 0.25, -0.625, 0.5]),
    }
    checks = []
    for name, x in inputs.items():
        expected = matrix @ x + bias
        outputs, report = run_encrypted(artifact, {"x": x}, reference={"y": expected}, atol=1e-5, rtol=0)
        np.testing.assert_allclose(outputs["y"], expected, atol=1e-5, rtol=0)
        runtime = report["phases"]["evaluate"]["runtime"]
        keygen = report["phases"]["keygen"]["runtime"]
        assert keygen["security_bits"] == 128
        assert keygen["slots"] > analysis["packed_period"]
        assert keygen["slots"] % analysis["packed_period"] == 0
        assert keygen["rotation_keys"] == len(analysis["rotation_steps"])
        assert runtime["input_ciphertexts"] == runtime["output_ciphertexts"] == 1
        assert runtime["rotations"] == expected_rotations
        assert runtime["multiply_plain"] == analysis["resources"]["plaintext_multiplications"] == 5
        assert runtime["multiply_cipher"] == 0
        assert report["server_has_no_secret_key"]
        checks.append({"input_case": name, "expected": expected.tolist(), "actual": outputs["y"].tolist(),
                       "report": report})
    (tmp_path / "verification.json").write_text(json.dumps({"packing": packing,
        "analysis": analysis, "checks": checks}, indent=2) + "\n")
    maximum = max(check["report"]["validation"]["outputs"]["y"]["max_abs_error"] for check in checks)
    print(f"{packing}: 3 encrypted inputs, rotations={expected_rotations}, max_abs_error={maximum:.3e}")


@pytest.mark.parametrize("rows,cols,packing,side,column", [
    (3, 5, "diagonal", "left", False), (3, 5, "bsgs", "left", False),
    (7, 3, "bsgs", "left", True), (5, 7, "auto", "right", False),
    (1, 9, "bsgs", "left", False), (9, 1, "bsgs", "left", False),
    (1, 1, "auto", "left", False),
])
def test_rectangular_padding_wrap_and_vector_orientation(tmp_path, rows, cols, packing, side, column):
    rng = np.random.default_rng(19 + rows + cols)
    matrix = rng.uniform(-0.5, 0.5, (rows, cols))
    bias = rng.uniform(-0.1, 0.1, rows)
    x = np.linspace(-0.75, 0.625, cols)
    source = tmp_path / "rectangular.onnx"
    input_shape, output_shape = make_model(source, matrix, bias, side=side, column=column)
    execute(source, tmp_path / "packed", packing, x.reshape(input_shape), (matrix @ x + bias).reshape(output_shape))


@pytest.mark.parametrize("zero,bias_enabled", [(False, True), (True, True), (True, False)])
def test_sparse_and_zero_matrices(tmp_path, zero, bias_enabled):
    matrix = np.zeros((4, 7), dtype=np.float64)
    if not zero:
        matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[3, 3] = 0.5, -0.25, 0.125, -0.75
        matrix[0, 3], matrix[1, 4], matrix[2, 5], matrix[3, 6] = -0.5, 0.25, -0.125, 0.75
    bias = np.array([0.125, -0.25, 0.5, -0.75]) if bias_enabled else np.zeros(4)
    x = np.linspace(-0.5, 0.5, 7)
    source = tmp_path / "sparse.onnx"
    make_model(source, matrix, bias)
    _, record = execute(source, tmp_path / "packed", "auto", x, matrix @ x + bias)
    result = record["analysis"]
    assert result["nonzero_diagonals"] == (0 if zero else 2)
    assert result["resources"]["rotations"] == (0 if zero else 1)
    assert result["multiplicative_depth"] == (0 if zero else 1)
    assert result["resources"]["materializations"] == (1 if zero else 0)


def test_torch_linear_gemm_export(tmp_path):
    import torch
    torch.manual_seed(217)
    model = torch.nn.Linear(5, 3).double()
    x = torch.tensor([[0.5, -0.25, 0.125, -0.75, 0.375]], dtype=torch.float64)
    path, inputs, expected = compile_torch(model, x, tmp_path / "torch", backend="seal-ckks", packing="auto")
    _, report = run_encrypted(path, inputs, reference=expected, atol=1e-4, rtol=1e-5)
    assert report["validation"]["passed"]
    (path / "case.json").write_text(json.dumps({"packing": "auto", "report": report,
        "analysis": json.loads((path / "resources_after.json").read_text())}, indent=2) + "\n")


def test_gemm_alpha_beta_and_scalar_bias(tmp_path):
    matrix = np.arange(15, dtype=np.float64).reshape(5, 3) / 16 - 0.375
    x = np.array([[0.5, -0.25, 0.125, -0.75, 0.375]])
    graph = h.make_graph([h.make_node("Gemm", ["x", "A", "b"], ["y"], alpha=-0.5, beta=2.0)],
                         "scaled-gemm", [h.make_tensor_value_info("x", T.DOUBLE, [1, 5])],
                         [h.make_tensor_value_info("y", T.DOUBLE, [1, 3])],
                         [n.from_array(matrix, "A"), n.from_array(np.array(0.125), "b")])
    source = tmp_path / "gemm.onnx"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), source)
    execute(source, tmp_path / "packed", "bsgs", x, -0.5 * (x @ matrix) + 0.25)
