"""Native DAG identity, schedule constraints and generated ciphertext lifetimes."""
import json
import os
from pathlib import Path

import pytest

from crossfhe.pipeline import command


BRANCHES = """module {
  func.func @model(%x: !heir.rlwecipher<1 x f64>, %y: !heir.rlwecipher<1 x f64>)
      -> !heir.rlwecipher<1 x f64>
      attributes {ppml.multiplicative_depth = 1 : i64,
                  ppml.logic_comparisons = 0 : i64, ppml.openfhe_depth = 1 : i64} {
    %a = heir.rlweadd(%x, %y) {noise = 0.0 : f64} : (!heir.rlwecipher<1 x f64>, !heir.rlwecipher<1 x f64>) -> !heir.rlwecipher<1 x f64>
    %b = heir.rlweadd(%y, %x) {noise = 0.0 : f64} : (!heir.rlwecipher<1 x f64>, !heir.rlwecipher<1 x f64>) -> !heir.rlwecipher<1 x f64>
    %c = heir.rlwemul(%a, %a) {noise = 0.0 : f64} : (!heir.rlwecipher<1 x f64>, !heir.rlwecipher<1 x f64>) -> !heir.rlwecipher<1 x f64>
    %d = heir.rlwemul(%b, %b) {noise = 0.0 : f64} : (!heir.rlwecipher<1 x f64>, !heir.rlwecipher<1 x f64>) -> !heir.rlwecipher<1 x f64>
    %e = heir.rlweadd(%c, %d) {noise = 0.0 : f64} : (!heir.rlwecipher<1 x f64>, !heir.rlwecipher<1 x f64>) -> !heir.rlwecipher<1 x f64>
    return %e : !heir.rlwecipher<1 x f64>
  }
}
"""


def translate(tmp_path, config, *, text=BRANCHES, backend="seal-ckks", schedule=None, dag=False):
    suffix = ".exe" if os.name == "nt" else ""
    source = tmp_path / "source.mlir"
    source.write_text(text, encoding="utf-8")
    target = tmp_path / ("dag.json" if dag else "generated.hpp")
    args = [Path(config["native_build"]) / "bin" / ("crossfhe-translate" + suffix),
            source, f"--backend={backend}", "-o", target]
    if dag:
        args.append("--emit-dag")
    if schedule is not None:
        candidate = tmp_path / "candidate.json"
        candidate.write_text(json.dumps({"schedule": schedule}), encoding="utf-8")
        args.append(f"--schedule={candidate}")
    command(args, config)
    return target.read_text(encoding="utf-8")


@pytest.mark.native
def test_dag_preserves_typed_dependencies_and_output_roots(tmp_path, config):
    dag = json.loads(translate(tmp_path, config, dag=True))
    assert dag["schema_version"] == 1 and dag["backend"] == "seal-ckks"
    assert dag["inputs"] == [{"id": "arg0", "type": "!heir.rlwecipher<1 x f64>"},
                             {"id": "arg1", "type": "!heir.rlwecipher<1 x f64>"}]
    assert dag["reference_schedule"] == ["n0", "n1", "n2", "n3", "n4"]
    assert dag["outputs"] == ["n4"]
    assert [node["dependencies"] for node in dag["nodes"]] == [[], [], ["n0"], ["n1"], ["n2", "n3"]]
    assert dag["nodes"][2]["operands"] == ["n0", "n0"]
    assert all(node["effect"] == "pure" for node in dag["nodes"])


@pytest.mark.native
def test_schedule_reorders_independent_branches_and_releases_last_uses(tmp_path, config):
    reference = translate(tmp_path, config)
    candidate = translate(tmp_path, config, schedule=["n0", "n2", "n1", "n3", "n4"])
    # ABI declarations remain byte-identical; only the function body is planned.
    assert reference.split("inline std::vector")[0] == candidate.split("inline std::vector")[0]
    assert "after last use" not in reference
    assert candidate.index("runtime.mul(") < candidate.rindex("runtime.add(inputs.at(1)")
    assert candidate.index("release n0 after last use") < candidate.index("runtime.add(inputs.at(1)")
    assert "release n4" not in candidate and "release arg" not in candidate
    assert candidate.count("runtime.mul(") == reference.count("runtime.mul(") == 2
    assert candidate.count("runtime.add(") == reference.count("runtime.add(") == 3


@pytest.mark.parametrize("schedule,diagnostic", [
    (["n0", "n1", "n2", "n3"], "every node exactly once"),
    (["n0", "n1", "n2", "n3", "n3"], "duplicate node"),
    (["n0", "n1", "n2", "n3", "arg0"], "unknown node"),
    (["n2", "n0", "n1", "n3", "n4"], "data/effect dependency"),
    (["n0", "n1", "n2", "n3", 4], "node ID strings"),
])
@pytest.mark.native
def test_invalid_schedule_is_rejected_before_output(tmp_path, config, schedule, diagnostic):
    with pytest.raises(RuntimeError, match=diagnostic):
        translate(tmp_path, config, schedule=schedule)
    assert not (tmp_path / "generated.hpp").exists()


@pytest.mark.native
def test_public_constants_are_schedule_nodes(tmp_path, config):
    text = (Path(__file__).parent / "inputs/affine_fhe.mlir").read_text()
    dag = json.loads(translate(tmp_path, config, text=text, dag=True))
    assert [node["op"] for node in dag["nodes"]] == ["arith.constant", "heir.rlweadd"]
    assert dag["nodes"][1]["operands"] == ["arg0", "n0"]
    assert dag["nodes"][1]["dependencies"] == ["n0"]
    with pytest.raises(RuntimeError, match="data/effect dependency"):
        translate(tmp_path, config, text=text, schedule=["n1", "n0"])


@pytest.mark.native
def test_openfhe_keeps_conservative_effect_order(tmp_path, config):
    dag = json.loads(translate(tmp_path, config, backend="openfhe-cross", dag=True))
    assert all(node["effect"] == "barrier" for node in dag["nodes"])
    with pytest.raises(RuntimeError, match="data/effect dependency"):
        translate(tmp_path, config, backend="openfhe-cross", schedule=["n0", "n2", "n1", "n3", "n4"])


@pytest.mark.native
def test_random_zero_materialization_is_an_effect_barrier(tmp_path, config):
    # sub(x,x) invokes public-key zero encryption in both CKKS adapters.
    text = BRANCHES.replace("heir.rlweadd(%y, %x)", "heir.rlwesub(%x, %x)")
    dag = json.loads(translate(tmp_path, config, text=text, dag=True))
    assert dag["nodes"][1]["effect"] == "barrier"
    assert dag["nodes"][1]["dependencies"] == ["n0"]
    assert "n1" in dag["nodes"][2]["dependencies"]
    with pytest.raises(RuntimeError, match="data/effect dependency"):
        translate(tmp_path, config, text=text, schedule=["n0", "n2", "n1", "n3", "n4"])
