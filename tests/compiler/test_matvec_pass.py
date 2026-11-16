"""Native lowering, rotation planning and malformed packed IR contracts."""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n, TensorProto as T
import pytest

from crossfhe.frontend import FrontendError
from crossfhe.packed import import_packed_onnx
from crossfhe.pipeline import command, read_config, compile_onnx


FIXTURE = Path(__file__).parent / "inputs/matvec.mlir"


def lower(tmp_path, text, strategy="auto"):
    config = read_config()
    source = tmp_path / "input.mlir"
    source.write_text(text)
    target = tmp_path / "lowered.mlir"
    command([Path(config["native_build"]) / "bin/crossfhe-opt", source,
             f"--ppml-fhe-matvec=strategy={strategy}", "--ppml-fhe-optimize", "--verify-each", "-o", target], config)
    contract = tmp_path / "contract.json"
    command([Path(config["native_build"]) / "bin/crossfhe-translate", target,
             "--backend=seal-ckks", "--emit-contract", "-o", contract], config)
    return target, json.loads(contract.read_text())


@pytest.mark.parametrize("strategy,rotations,baby", [("diagonal", 3, 4), ("bsgs", 2, 2), ("auto", 2, 2)])
def test_native_selects_rotation_plan(tmp_path, strategy, rotations, baby):
    path, result = lower(tmp_path, FIXTURE.read_text(), strategy)
    assert "heir.matvec" not in path.read_text()
    assert result["resources"]["rotations"] == rotations
    # Uniform coefficients let CSE share equal giant-group products in BSGS.
    assert result["resources"]["plaintext_multiplications"] == (4 if strategy == "diagonal" else 2)
    assert result["matvec_baby_step"] == baby
    assert result["multiplicative_depth"] == 1
    assert len(result["rotation_steps"]) == rotations
    assert result["input_elements"] == result["output_elements"] == 4


@pytest.mark.parametrize("old,new,message", [
    ("<4 x f64>", "<3 x f64>", "power of two"),
    ("tensor<4x4xf64>", "tensor<5x4xf64>", "dimensions must fit"),
    ("tensor<4xf64>", "tensor<3xf64>", "bias must match"),
    ("dense<1.0>", "dense<0x7FF0000000000000>", "finite public"),
])
def test_native_rejects_invalid_shapes_and_coefficients(tmp_path, old, new, message):
    with pytest.raises(RuntimeError, match=message):
        lower(tmp_path, FIXTURE.read_text().replace(old, new))


def test_zero_matrix_has_no_rotations_or_plain_multiplications(tmp_path):
    path, result = lower(tmp_path, FIXTURE.read_text().replace("dense<1.0>", "dense<0.0>"))
    assert "heir.packed_zero" in path.read_text()
    assert result["nonzero_diagonals"] == result["multiplicative_depth"] == 0
    assert result["rotation_steps"] == []
    assert result["resources"]["materializations"] == 1
    assert result["resources"]["plaintext_multiplications"] == 0


@pytest.mark.parametrize("mutation,message", [
    (lambda s: s.replace('ppml.multiplicative_depth = 1', 'ppml.multiplicative_depth = 0'), "depth contract is stale"),
    (lambda s: s.replace('ppml.input_elements = 4', 'ppml.input_elements = 5'), "shape contract"),
    (lambda s: s.replace('step = 1 : i64', 'step = 4 : i64'), "rotation step"),
    (lambda s: s.replace('values = dense<1.000000e+00>', 'values = dense<0.000000e+00>'), "zero plaintext"),
])
def test_codegen_rejects_malformed_packed_contracts(tmp_path, mutation, message):
    path, _ = lower(tmp_path, FIXTURE.read_text())
    original = path.read_text()
    changed = mutation(original)
    assert changed != original
    path.write_text(changed)
    config = read_config()
    with pytest.raises(RuntimeError, match=message):
        command([Path(config["native_build"]) / "bin/crossfhe-translate", path,
                 "--backend=seal-ckks", "-o", tmp_path / "out.hpp"], config)


def test_openfhe_packed_path_is_rejected(tmp_path):
    path, _ = lower(tmp_path, FIXTURE.read_text())
    config = read_config()
    with pytest.raises(RuntimeError, match="require seal-ckks"):
        command([Path(config["native_build"]) / "bin/crossfhe-translate", path,
                 "--backend=openfhe-cross", "-o", tmp_path / "out.hpp"], config)
    with pytest.raises(ValueError, match="requires seal-ckks"):
        compile_onnx("unused.onnx", tmp_path / "unused", backend="openfhe-cross", packing="auto")


def test_unmarked_function_is_not_transformed(tmp_path):
    # Use opt directly; codegen correctly rejects the remaining high-level op.
    text = FIXTURE.read_text().replace("ppml.fhe_enabled = true", "ppml.fhe_enabled = false")
    source = tmp_path / "disabled.mlir"
    source.write_text(text)
    config = read_config()
    output, _ = command([Path(config["native_build"]) / "bin/crossfhe-opt", source, "--ppml-fhe-matvec"], config)
    assert "heir.matvec" in output and "heir.packed_mul_plain" not in output


def test_frontend_does_not_silently_pack_nonlinear_models(tmp_path):
    graph = h.make_graph([h.make_node("MatMul", ["A", "x"], ["z"]),
                          h.make_node("Relu", ["z"], ["y"])], "nonlinear",
                         [h.make_tensor_value_info("x", T.FLOAT, [3])],
                         [h.make_tensor_value_info("y", T.FLOAT, [2])],
                         [n.from_array(np.ones((2, 3), dtype=np.float32), "A")])
    source = tmp_path / "nonlinear.onnx"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), source)
    with pytest.raises(FrontendError, match="unsupported packed operation Relu"):
        import_packed_onnx(source)
