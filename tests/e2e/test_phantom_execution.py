"""Optional real CUDA integration: compiler-generated models and separate client/server processes."""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as n, TensorProto as T
import pytest

from crossfhe.pipeline import compile_onnx, compile_torch, run_encrypted, command

pytestmark = [pytest.mark.fhe, pytest.mark.native, pytest.mark.gpu]


def matvec(path, matrix, bias):
    rows, cols = matrix.shape
    graph = h.make_graph([h.make_node("MatMul", ["A", "x"], ["v"]),
                          h.make_node("Add", ["v", "b"], ["y"])], "phantom-matvec",
                         [h.make_tensor_value_info("x", T.DOUBLE, [cols])],
                         [h.make_tensor_value_info("y", T.DOUBLE, [rows])],
                         [n.from_array(matrix, "A"), n.from_array(bias, "b")])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), path)


@pytest.mark.parametrize("packing,rotations", [("diagonal", 5), ("bsgs", 3), ("auto", 3)])
def test_compiler_generated_periodic_matvec(tmp_path, config, packing, rotations):
    matrix = np.array([[0,.5,0,-.25,2],[-.5,0,0,0,0],[0,.25,0,0,0],
                       [-2,0,0,0,0],[.75,0,0,-.125,0]])
    bias = np.array([.0625,-.125,.25,-.375,.5])
    source = tmp_path/"model.onnx"
    matvec(source, matrix, bias)
    artifact = compile_onnx(source, tmp_path/"model", config=config, backend="phantom-ckks", packing=packing)
    for x in (np.eye(5)[0], np.eye(5)[-1], np.array([.875,-.375,.25,-.625,.5])):
        _, report = run_encrypted(artifact, {"x":x}, reference={"y":matrix@x+bias},
                                  config=config, atol=1e-5, rtol=0)
        assert report["server_has_no_secret_key"]
        runtime = report["phases"]["evaluate"]["runtime"]
        assert runtime["rotations"] == rotations
        assert runtime["multiply_plain"] == runtime["rescale"] == 5
        assert runtime["input_ciphertexts"] == runtime["output_ciphertexts"] == 1
        keygen = report["phases"]["keygen"]["runtime"]
        assert keygen["security_bits"] == 128
        assert keygen["slots"] > 8 and keygen["slots"] % 8 == 0
        server = Path(report["run_directory"])/"server"
        assert not any("secret" in file.name for file in server.iterdir())
        assert report["evaluation_bundle_bytes"] == sum(f.stat().st_size for f in server.glob("*.phantom"))

    # The evaluate command has only a server directory and ciphertexts. Missing public keys fail explicitly.
    (server/"galois.phantom").unlink()
    with pytest.raises(RuntimeError, match="missing or oversized artifact"):
        command([artifact/"build/bin/model-runner", "evaluate", server, server/"input.ct", server/"bad.ct"], config)


@pytest.mark.parametrize("zero", [False, True])
def test_rectangular_and_zero_matrix(tmp_path, config, zero):
    rng = np.random.default_rng(17)
    matrix = np.zeros((3,7)) if zero else rng.uniform(-.5,.5,(3,7))
    bias = np.array([.125,-.25,.5])
    x = np.linspace(-.5,.75,7)
    source = tmp_path/"rect.onnx"
    matvec(source, matrix, bias)
    artifact = compile_onnx(source, tmp_path/"model", config=config, backend="phantom-ckks", packing="auto")
    _, report = run_encrypted(artifact, {"x":x}, reference={"y":matrix@x+bias}, config=config, atol=1e-5, rtol=0)
    resources = json.loads((artifact/"resources_after.json").read_text())["resources"]
    runtime = report["phases"]["evaluate"]["runtime"]
    for actual, planned in (("rotations","rotations"), ("multiply_plain","plaintext_multiplications"),
                            ("add","additions"), ("materializations","materializations")):
        assert runtime[actual] == resources[planned]
    server = Path(report["run_directory"])/"server"
    runner = artifact/"build/bin/model-runner"
    original = (server/"input.ct").read_bytes()
    model_id = original.split(b"\n", 1)[0].split()[2]
    for corrupt, diagnostic in ((b"SEAL"+original[4:], "backend/version/model/kind mismatch"),
                                (original.replace(model_id,b"0"*64,1), "backend/version/model/kind mismatch"),
                                (original[:-32], "truncated ciphertext")):
        (server/"bad.ct").write_bytes(corrupt)
        with pytest.raises(RuntimeError, match=diagnostic):
            command([runner,"evaluate",server,server/"bad.ct",server/"bad-output.ct"],config)


@pytest.mark.torch
def test_multilevel_scalar_arithmetic_and_public_materialization(tmp_path, config):
    import torch
    class Model(torch.nn.Module):
        def forward(self, x):
            square = x*x
            return ((square+1.0)*(x-.25) + (.5-x), square*square,
                    square*0.0, x-x, torch.tensor([.25,-.125],dtype=torch.float64))
    x = torch.tensor([.5,-.25,.125],dtype=torch.float64)
    artifact, inputs, reference = compile_torch(Model(), x, tmp_path/"scalar", config=config,
                                               backend="phantom-ckks", optimization="none")
    # Fresh keys/encryption expose the pinned library's intermittent multiword
    # CRT decode failure on initial-level zeros and signed public constants.
    for _ in range(3):
        _, report = run_encrypted(artifact, inputs, reference=reference, config=config, atol=1e-5, rtol=0)
        runtime = report["phases"]["evaluate"]["runtime"]
        assert runtime["multiply_cipher"] > 0
        assert runtime["relinearize"] == runtime["multiply_cipher"]
        assert runtime["mod_switch"] > 0
        assert report["server_has_no_secret_key"]


@pytest.mark.torch
def test_depth_beyond_parameter_policy_is_rejected(tmp_path, config):
    import torch
    class Deep(torch.nn.Module):
        def forward(self, x):
            for _ in range(20): x = x*x
            return x
    artifact, inputs, _ = compile_torch(Deep(),torch.tensor([.5],dtype=torch.float64),
                                       tmp_path/"deep",config=config,backend="phantom-ckks")
    with pytest.raises(RuntimeError,match="model depth exceeds supported 128-bit CKKS policy"):
        run_encrypted(artifact,inputs,config=config)


@pytest.mark.torch
def test_logic_is_rejected_and_bounded_approximation_runs(tmp_path, config):
    import torch
    class Relu(torch.nn.Module):
        def forward(self, x): return torch.relu(x)
    x = torch.tensor([-.75, .25],dtype=torch.float64)
    with pytest.raises(ValueError, match="model needs logic FHE"):
        compile_torch(Relu(),x,tmp_path/"exact",config=config,backend="phantom-ckks")
    artifact, inputs, _ = compile_torch(Relu(),x,tmp_path/"approx",config=config,backend="phantom-ckks",
                                       relu_bound=1.0,relu_max_error=.0625)
    outputs, report = run_encrypted(artifact,inputs,config=config)
    expected = torch.relu(x).numpy()
    np.testing.assert_allclose(outputs["output_0"],expected,atol=.0625+1e-5,rtol=0)
    assert report["server_has_no_secret_key"]
