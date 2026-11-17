"""Restricted proposals reach real builds and same-ciphertext encrypted verification."""
import json

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as n
import pytest

from backend.ai.driver import baseline_candidate, run_candidates
from backend.ai.provider import OfflineProvider
from backend.ai.artifacts import digest
from backend.ai.task import prepare
from crossfhe.pipeline import compile_onnx


def branch_model(path):
    graph = h.make_graph([
        h.make_node("Mul", ["x", "scale"], ["a"]),
        h.make_node("Add", ["y", "bias"], ["b"]),
        h.make_node("Add", ["a", "b"], ["z"]),
    ], "independent-branches", [h.make_tensor_value_info(name, T.DOUBLE, [2]) for name in ("x", "y")],
        [h.make_tensor_value_info("z", T.DOUBLE, [2])],
        [n.from_array(np.array(.5), "scale"), n.from_array(np.array(.25), "bias")])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), path)


@pytest.mark.native
@pytest.mark.fhe
@pytest.mark.parametrize("backend", [
    pytest.param("seal-ckks", marks=pytest.mark.seal),
    pytest.param("openfhe-cross", marks=pytest.mark.openfhe),
    pytest.param("phantom-ckks", marks=pytest.mark.gpu),
])
def test_ai_candidate_builds_and_uses_trusted_client_on_same_ciphertext(tmp_path, config, backend):
    source = tmp_path/"branches.onnx"
    branch_model(source)
    artifact = compile_onnx(source, tmp_path/"reference", config=config, backend=backend, optimization="none")
    manifest_hash = digest(artifact/"manifest.json")
    header_hash = digest(artifact/"model.hpp")
    task = prepare(artifact, tmp_path/"task", max_attempts=2, repetitions=2, warmups=0,
                   atol=1e-4, rtol=1e-5)
    candidate = baseline_candidate(tmp_path/"task")
    invalid = {**candidate, "task_hash": "0"*64}
    report = run_candidates(tmp_path/"task", artifact, tmp_path/"run",
                            OfflineProvider([invalid, candidate]), config)
    assert report["status"] == "validated", report
    assert report["validation_performed"] and not report["promotion_allowed"]
    assert report["attempts"][0]["status"] == "rejected"
    assert report["attempts"][0]["stage"] == "schema"
    successful = report["attempts"][1]
    validation = successful["validation"]
    assert validation["functional_passed"] and validation["policy_passed"]
    assert validation["measured"] and validation["cases"]
    assert all(case["counts_match"] for case in validation["cases"])
    assert validation["measurement"]["repetitions"] == 2
    assert manifest_hash == digest(artifact/"manifest.json")
    assert header_hash == digest(artifact/"model.hpp")
    assert report["task_hash"] == task["task_hash"]
    saved = json.loads((tmp_path/"run/report.json").read_text())
    assert saved["status"] == "validated"
    repair_messages = json.loads((tmp_path/"run/attempt-02/messages.json").read_text())
    assert len(repair_messages) == 4 and "failed_stage" in repair_messages[-1]["content"]
    assert not any("secret" in path.name.lower() for path in (tmp_path/"run/attempt-02/validation/server").iterdir())


@pytest.mark.native
@pytest.mark.fhe
@pytest.mark.openfhe
def test_ai_openfhe_logic_toy_is_only_functional_validation(tmp_path, config):
    graph = h.make_graph([h.make_node("Relu", ["x"], ["y"])], "logic",
                        [h.make_tensor_value_info("x", T.DOUBLE, [1])],
                        [h.make_tensor_value_info("y", T.DOUBLE, [1])])
    source = tmp_path/"relu.onnx"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), source)
    artifact = compile_onnx(source, tmp_path/"reference", config=config, backend="openfhe-cross")
    prepare(artifact, tmp_path/"task", insecure_demo=True, repetitions=1, warmups=0, max_attempts=1)
    candidate = baseline_candidate(tmp_path/"task")
    report = run_candidates(tmp_path/"task", artifact, tmp_path/"run", OfflineProvider([candidate]), config,
                            cases=[{"x": np.array([-.5])}, {"x": np.array([.5])}])
    assert report["status"] == "functional_only", report
    validation = report["attempts"][0]["validation"]
    assert validation["functional_passed"] and not validation["policy_passed"]
    assert not report["selectable"] and not report["promotion_allowed"]


@pytest.mark.native
@pytest.mark.fhe
@pytest.mark.parametrize("backend", [
    pytest.param("seal-ckks", marks=pytest.mark.seal),
    pytest.param("phantom-ckks", marks=pytest.mark.gpu),
])
def test_ai_packed_candidate_preserves_period_and_rotation_keys(tmp_path, config, backend):
    matrix = np.array([[.5,0,-.25,.125,0], [0,.25,.5,0,-.125], [-.25,0,0,.5,.25]])
    graph = h.make_graph([h.make_node("MatMul", ["A", "x"], ["y"])], "packed",
                        [h.make_tensor_value_info("x", T.DOUBLE, [5])],
                        [h.make_tensor_value_info("y", T.DOUBLE, [3])], [n.from_array(matrix, "A")])
    source = tmp_path/"packed.onnx"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), source)
    artifact = compile_onnx(source, tmp_path/"reference", config=config, backend=backend, packing="auto")
    prepare(artifact, tmp_path/"task", repetitions=1, warmups=0, max_attempts=1, atol=1e-4, rtol=1e-5)
    candidate = baseline_candidate(tmp_path/"task")
    report = run_candidates(tmp_path/"task", artifact, tmp_path/"run", OfflineProvider([candidate]), config,
                            cases=[{"x": np.eye(5)[0]/2}, {"x": np.eye(5)[-1]/2}])
    assert report["status"] == "validated", report
    validation = report["attempts"][0]["validation"]
    assert validation["functional_passed"] and validation["policy_passed"]
    assert all(case["counts_match"] for case in validation["cases"])


@pytest.mark.native
@pytest.mark.fhe
@pytest.mark.seal
def test_ai_failed_schedule_cannot_retry_with_fresh_keys(tmp_path, config):
    graph = h.make_graph([h.make_node("Relu", ["x"], ["y"])], "approximation-rejection",
                        [h.make_tensor_value_info("x", T.DOUBLE, [1])],
                        [h.make_tensor_value_info("y", T.DOUBLE, [1])])
    source = tmp_path/"relu.onnx"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), source)
    artifact = compile_onnx(source, tmp_path/"reference", config=config, backend="seal-ckks",
                            relu_bound=1.0, relu_max_error=.0625)
    # p(0)=1/16 differs from ReLU(0)=0 by a public, known amount. Changing
    # encryption randomness cannot repair this deliberately incompatible budget.
    prepare(artifact, tmp_path/"task", max_attempts=2, atol=1e-6, rtol=0, repetitions=1, warmups=0)
    candidate = baseline_candidate(tmp_path/"task")
    repeated = {**candidate, "rationale": ["Retry the same schedule with a different explanation."]}
    report = run_candidates(tmp_path/"task", artifact, tmp_path/"run",
                            OfflineProvider([candidate, repeated]), config,
                            cases=[{"x": np.array([0.0])}])
    assert report["status"] == "rejected" and report["validation_performed"]
    assert report["attempts"][0]["stage"] == "functional"
    assert "duplicate schedule" in report["attempts"][1]["error"]
    assert not (tmp_path/"run/attempt-02/build").exists()
