"""Integrity, candidate capability limits, and public repair-message contracts."""

import copy
import json

import pytest

from backend.ai.prompts import build_messages
from backend.ai.protocol import (
    CANDIDATE_SCHEMA, canonical_bytes, parse_json, read_json, seal_task, sha256,
    validate_candidate, verify_task,
)


@pytest.fixture
def task_bundle(tmp_path, dag):
    context = b"public compiler context\n"
    (tmp_path / "reference.hpp").write_bytes(context)
    contract = {"input_count": 1, "output_count": 1, "multiplicative_depth": 1,
                "logic_comparisons": 0, "openfhe_depth": 1, "resources": {"operations": 4}}
    files = {"reference.hpp": context, "program.mlir": b"module {}\n",
             "contract.json": canonical_bytes(contract), "dag.json": canonical_bytes(dag),
             "candidate.schema.json": canonical_bytes(CANDIDATE_SCHEMA),
             "api_context.json": canonical_bytes({"backend": "seal-ckks", "entries": []}),
             "PROMPT.md": b"immutable serial schedule\n", "seal_runtime.hpp": b"public runtime\n"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    task = seal_task({
        "schema_version": 2, "mode": "schedule-v1", "backend": "seal-ckks",
        "contract": contract,
        "validation_policy": {"security_profile": "tc128", "insecure_demo": False,
            "atol": 0.001, "rtol": 0.001, "input_bound": 0.5, "comparison_scale": 1024.0,
            "seed": 17, "max_attempts": 10, "warmups": 1, "repetitions": 3},
        "tensor_mapping": {"inputs": [{"name": "x", "shape": [1], "offset": 0, "count": 1}],
                           "outputs": [{"name": "y", "shape": [1], "offset": 0, "count": 1}]},
        "approximation": {"relu_input_bound": 1.0},
        "immutable": ["ABI", "approximation"], "library_revision": "pinned-test-revision",
        "context_hashes": {name: sha256(data) for name, data in files.items()},
    })
    (tmp_path / "task.json").write_bytes(canonical_bytes(task))
    return tmp_path, task


@pytest.fixture
def dag():
    return {
        "schema_version": 1,
        "backend": "seal-ckks",
        "inputs": [{"id": "arg0", "type": "cipher"}],
        "nodes": [
            {"id": "n0", "operands": ["arg0"], "dependencies": []},
            {"id": "n1", "operands": ["arg0"], "dependencies": []},
            {"id": "n2", "operands": ["n0"], "dependencies": ["n0"]},
            {"id": "n3", "operands": ["n1", "n2"], "dependencies": ["n1", "n2"]},
        ],
        "outputs": ["n3"],
        "reference_schedule": ["n0", "n1", "n2", "n3"],
    }


def proposal(task, schedule=None):
    return {
        "schema_version": 1, "status": "proposal", "mode": "schedule-v1",
        "task_hash": task["task_hash"],
        "schedule": schedule if schedule is not None else ["n0", "n2", "n1", "n3"],
        "rationale": ["Finish the first branch before starting the second."],
    }


def test_task_hash_is_canonical_and_does_not_mutate_input():
    task = {"z": ["密文", {"a": 1}], "a": 2, "task_hash": "obsolete"}
    assert canonical_bytes({"z": "密文", "a": 1}) == '{"a":1,"z":"密文"}'.encode()
    sealed = seal_task(task)
    assert sealed == seal_task({"a": 2, "z": ["密文", {"a": 1}]})
    sealed["z"][1]["a"] = 9
    assert task["z"][1]["a"] == 1
    assert task["task_hash"] == "obsolete"


@pytest.mark.parametrize("text", [
    '{"a":1,"a":2}', '{"nested":{"a":1,"a":2}}',
    '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}',
])
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, text):
    path = tmp_path / "response.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate JSON key|non-finite JSON number"):
        read_json(path)


@pytest.mark.parametrize("text", ['```json\n{}\n```', 'Here is the plan: {}', '{} {}'])
def test_model_response_wrappers_and_trailing_values_are_rejected(text):
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_json(text)


def test_task_and_context_integrity_are_both_checked(task_bundle):
    directory, task = task_bundle
    assert verify_task(directory) == task
    (directory / "reference.hpp").write_text("modified evaluator")
    with pytest.raises(ValueError, match="context hash mismatch"):
        verify_task(directory)
    task["validation_policy"]["atol"] = 100
    (directory / "task.json").write_bytes(canonical_bytes(task))
    with pytest.raises(ValueError, match="task hash mismatch"):
        verify_task(directory)


@pytest.mark.parametrize("name", ["/tmp/reference.hpp", "../reference.hpp", "sub/reference.hpp", "secret.key"])
def test_task_cannot_authorize_reading_nonpublic_paths(task_bundle, name):
    directory, task = task_bundle
    task["context_hashes"] = {name: "0" * 64}
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match="not allowlisted"):
        verify_task(directory)


def test_legal_independent_branch_reordering_is_accepted(task_bundle, dag):
    _, task = task_bundle
    candidate = proposal(task)
    assert validate_candidate(candidate, task, dag) == candidate
    no_candidate = {"schema_version": 1, "status": "no_candidate",
                    "task_hash": task["task_hash"], "reason": "No beneficial new order."}
    assert validate_candidate(no_candidate, task, dag) == no_candidate


@pytest.mark.parametrize(("change", "diagnostic"), [
    ({"schema_version": True}, "schema_version"),
    ({"task_hash": "0" * 64}, "task hash mismatch"),
    ({"mode": "cpp-adapter-v1"}, "candidate mode"),
    ({"code": "decrypt(input);"}, "fields mismatch"),
    ({"rationale": "not an array"}, "rationale"),
    ({"schedule": ["n0", "n0", "n1", "n3"]}, "duplicate schedule node"),
    ({"schedule": ["n0", "n1", "n2"]}, "node set mismatch"),
    ({"schedule": ["n0", "n1", "n2", "extra"]}, "node set mismatch"),
    ({"schedule": ["n2", "n0", "n1", "n3"]}, "dependency used before definition"),
])
def test_candidate_edits_outside_legal_schedule_are_rejected(task_bundle, dag, change, diagnostic):
    _, task = task_bundle
    candidate = proposal(task)
    candidate.update(change)
    with pytest.raises(ValueError, match=diagnostic):
        validate_candidate(candidate, task, dag)


def test_effect_dependency_is_enforced_without_a_data_edge(task_bundle, dag):
    _, task = task_bundle
    dag["nodes"][1]["dependencies"] = ["n2"]
    with pytest.raises(ValueError, match="dependency used before definition"):
        validate_candidate(proposal(task, ["n0", "n1", "n2", "n3"]), task, dag)
    assert validate_candidate(proposal(task), task, dag)["status"] == "proposal"


def test_rationale_is_required_and_no_candidate_cannot_carry_a_schedule(task_bundle, dag):
    _, task = task_bundle
    candidate = proposal(task)
    del candidate["rationale"]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_candidate(candidate, task, dag)
    candidate = {"schema_version": 1, "status": "no_candidate", "task_hash": task["task_hash"],
                 "reason": "No improvement", "schedule": ["n0", "n1", "n2", "n3"]}
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_candidate(candidate, task, dag)


def test_missing_data_dependency_cannot_hide_an_invalid_dag(task_bundle, dag):
    _, task = task_bundle
    dag["nodes"][2]["dependencies"] = []
    with pytest.raises(ValueError, match="DAG omits data dependency"):
        validate_candidate(proposal(task), task, dag)


def test_messages_preserve_frozen_contract_previous_candidate_and_diagnostics(task_bundle, dag):
    _, task = task_bundle
    candidate = proposal(task)
    feedback = {"stage": "dependency", "node": "n2", "error": "n0 was not defined"}
    original = copy.deepcopy((task, dag, candidate, feedback))
    messages = build_messages(task, dag, {"public_api": "signature"}, candidate, feedback)
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert "cryptographic parameters" in messages[0]["content"]
    assert "effect dependencies" in messages[0]["content"]
    assert "external driver" in messages[0]["content"]
    public = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert public["immutable_contract"] == task["contract"]
    assert public["validation_policy"] == task["validation_policy"]
    for key in ("tensor_mapping", "approximation", "immutable", "library_revision"):
        assert public[key] == task[key]
    assert public["dag"] == dag
    assert json.loads(messages[2]["content"]) == candidate
    assert json.loads(messages[3]["content"].split("\n", 1)[1]) == feedback
    assert (task, dag, candidate, feedback) == original


def test_feedback_without_previous_candidate_is_rejected(task_bundle, dag):
    _, task = task_bundle
    with pytest.raises(ValueError, match="requires the previous candidate"):
        build_messages(task, dag, {}, feedback={"stage": "build"})


@pytest.mark.parametrize(("field", "value"), [
    ("atol", True), ("rtol", -1), ("comparison_scale", 0), ("input_bound", "0.5"),
    ("atol", 10**400), ("max_attempts", True), ("max_attempts", 11),
    ("repetitions", 0), ("warmups", -1), ("seed", 2**32),
    ("insecure_demo", 0), ("insecure_demo", True), ("security_profile", "INSECURE_TOY"),
])
def test_sealed_invalid_validation_policy_is_rejected(task_bundle, field, value):
    directory, task = task_bundle
    task["validation_policy"][field] = value
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match="validation_policy"):
        verify_task(directory)


@pytest.mark.parametrize(("value", "accepted"), [(1, True), (9, True), (10, True),
                                                   (True, False), (0, False), (11, False)])
def test_task_and_policy_attempt_limits_cannot_disagree(task_bundle, value, accepted):
    """The sealed policy is the only source of the driver's attempt limit."""
    directory, task = task_bundle
    assert "max_attempts" not in task and "max_repair_rounds" not in task
    task["validation_policy"]["max_attempts"] = value
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    if accepted:
        assert verify_task(directory)["validation_policy"]["max_attempts"] == value
    else:
        with pytest.raises(ValueError, match="max_attempts"):
            verify_task(directory)


def test_empty_policy_or_omitted_context_cannot_be_sealed_into_a_valid_task(task_bundle):
    directory, task = task_bundle
    changed = copy.deepcopy(task)
    changed["validation_policy"] = {}
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(changed)))
    with pytest.raises(ValueError, match="validation_policy"):
        verify_task(directory)
    del task["context_hashes"]["dag.json"]
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match="missing required context"):
        verify_task(directory)


@pytest.mark.parametrize(("direction", "field", "value"), [
    ("inputs", "shape", [True]), ("outputs", "count", True),
    ("inputs", "shape", [0]), ("outputs", "count", 2),
    ("inputs", "offset", 1), ("outputs", "offset", False),
])
def test_tensor_mapping_shape_count_and_offsets_are_checked(task_bundle, direction, field, value):
    directory, task = task_bundle
    task["tensor_mapping"][direction][0][field] = value
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match="tensor_mapping"):
        verify_task(directory)


@pytest.mark.parametrize("direction", ["inputs", "outputs"])
def test_empty_tensor_mapping_cannot_make_validation_vacuous(task_bundle, direction):
    directory, task = task_bundle
    task["tensor_mapping"][direction] = []
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match="nonempty array"):
        verify_task(directory)


def test_scalar_tensor_shape_is_valid_and_contract_mismatch_is_rejected(task_bundle):
    directory, task = task_bundle
    task["tensor_mapping"]["inputs"][0]["shape"] = []
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    assert verify_task(directory)["tensor_mapping"]["inputs"][0]["count"] == 1
    task["contract"]["output_count"] = 2
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match="contract differs"):
        verify_task(directory)


def test_packed_tensor_elements_are_distinct_from_ciphertext_counts(task_bundle):
    directory, task = task_bundle
    task["contract"].update(layout="periodic", packed_period=4, input_elements=3, output_elements=2)
    task["tensor_mapping"]["inputs"][0].update(shape=[1, 3], count=3)
    task["tensor_mapping"]["outputs"][0].update(shape=[1, 2], count=2)
    content = canonical_bytes(task["contract"])
    (directory / "contract.json").write_bytes(content)
    task["context_hashes"]["contract.json"] = sha256(content)
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    verified = verify_task(directory)
    assert verified["contract"]["input_count"] == 1
    assert verified["tensor_mapping"]["inputs"][0]["count"] == 3


@pytest.mark.parametrize(("field", "value"), [("multiplicative_depth", True),
                                               ("logic_comparisons", -1), ("resources", {"operations": False})])
def test_resource_contract_does_not_accept_boolean_or_negative_counts(task_bundle, field, value):
    directory, task = task_bundle
    task["contract"][field] = value
    content = canonical_bytes(task["contract"])
    (directory / "contract.json").write_bytes(content)
    task["context_hashes"]["contract.json"] = sha256(content)
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match="contract"):
        verify_task(directory)


@pytest.mark.parametrize(("filename", "new_content", "diagnostic"), [
    ("candidate.schema.json", {"type": "object"}, "candidate schema differs"),
    ("dag.json", {"schema_version": True, "backend": "seal-ckks"}, "DAG backend/schema"),
    ("api_context.json", {"backend": "phantom-ckks"}, "API context backend"),
])
def test_rehashed_context_still_must_match_its_protocol_role(task_bundle, filename, new_content, diagnostic):
    directory, task = task_bundle
    content = canonical_bytes(new_content)
    (directory / filename).write_bytes(content)
    task["context_hashes"][filename] = sha256(content)
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    with pytest.raises(ValueError, match=diagnostic):
        verify_task(directory)


@pytest.mark.parametrize(("backend", "profile", "insecure"), [
    ("seal-ckks", "tc128", False), ("phantom-ckks", "tc128", False),
    ("openfhe-cross", "HEStd_128_classic+STD128", False),
    ("openfhe-cross", "INSECURE_TOY", True),
])
def test_supported_backend_policy_and_public_context_remain_valid(task_bundle, backend, profile, insecure):
    directory, task = task_bundle
    task["backend"] = backend
    task["validation_policy"].update(security_profile=profile, insecure_demo=insecure)
    dag = read_json(directory / "dag.json")
    dag["backend"] = backend
    files = {"dag.json": canonical_bytes(dag),
             "api_context.json": canonical_bytes({"backend": backend, "entries": []})}
    runtime = backend.split("-", 1)[0] + "_runtime.hpp"
    del task["context_hashes"]["seal_runtime.hpp"]
    files[runtime] = b"pinned public runtime\n"
    for name, content in files.items():
        (directory / name).write_bytes(content)
        task["context_hashes"][name] = sha256(content)
    (directory / "task.json").write_bytes(canonical_bytes(seal_task(task)))
    assert verify_task(directory)["validation_policy"]["security_profile"] == profile
