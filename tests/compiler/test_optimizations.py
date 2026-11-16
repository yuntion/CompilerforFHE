"""Check transformations, rejected contracts and opt-in semantics through native tools."""
import json
from pathlib import Path

import pytest
from crossfhe.pipeline import command, read_config


@pytest.fixture(scope="module")
def config():
    return read_config()


def lower(tmp_path, config, body, *, args="%x: f64, %y: f64", outputs="f64", flags=(), attrs=""):
    source=tmp_path/"source.mlir"
    source.write_text(f"module {{ func.func @model({args}) -> {outputs} {attrs} {{\n{body}\n}} }}")
    target=tmp_path/"fhe.mlir"
    tool=Path(config["native_build"])/"bin/crossfhe-opt"
    command([tool,source,"--ppml-to-scfhe",*([f for f in flags if 'scfhe-' in f]),
             "--ppml-to-heir","--ppml-fhe-refresh-contract",
             *[f for f in flags if 'scfhe-' not in f],"--verify-each","-o",target],config)
    return target


def contract(path, config):
    result=path.with_suffix(".json")
    command([Path(config["native_build"])/"bin/crossfhe-translate",path,
             "--emit-contract","-o",result],config)
    return json.loads(result.read_text())


def test_duplicate_cipher_multiplications_are_shared_across_source_tags(tmp_path,config):
    path=lower(tmp_path,config,'''%a = arith.mulf %x, %y {ppml.source = "a"} : f64
%b = arith.mulf %x, %y {ppml.source = "b"} : f64
%c = arith.addf %a, %b : f64
return %c : f64''',flags=["--ppml-fhe-cse"])
    assert contract(path,config)["resources"]["ciphertext_multiplications"]==1


def test_cse_preserves_operand_order(tmp_path,config):
    path=lower(tmp_path,config,'''%a = arith.mulf %x, %y : f64
%b = arith.mulf %y, %x : f64
return %a, %b : f64, f64''',outputs="(f64, f64)",flags=["--ppml-fhe-cse"])
    assert contract(path,config)["resources"]["ciphertext_multiplications"]==2


def test_repeated_comparisons_share_the_whole_switching_chain(tmp_path,config):
    path=lower(tmp_path,config,'''%z = arith.constant 0.0 : f64
%a = arith.cmpf olt, %x, %z {ppml.source = "first"} : f64
%b = arith.cmpf olt, %x, %z {ppml.source = "second"} : f64
%u = arith.select %a, %z, %x : f64
%v = arith.select %b, %z, %x : f64
return %u, %v : f64, f64''',outputs="(f64, f64)",flags=["--ppml-fhe-optimize"])
    result=contract(path,config)
    assert result["logic_comparisons"]==1
    assert [result["resources"][key] for key in ("ckks_to_logic","logic_bootstraps","logic_to_ckks")]==[1,1,1]
    assert result["resources"]["selects"]==1


def test_dead_switching_chain_no_longer_reserves_levels(tmp_path,config):
    path=lower(tmp_path,config,'''%z = arith.constant 0.0 : f64
%c = arith.cmpf olt, %x, %z : f64
%dead = arith.select %c, %z, %x : f64
return %x : f64''',flags=["--ppml-fhe-dce"])
    result=contract(path,config)
    assert result["logic_comparisons"]==result["openfhe_depth"]==0
    assert result["resources"]["operations"]==0


@pytest.mark.parametrize("operation,constant",[("mulf",1.0),("addf",0.0),("subf",0.0),("mulf",0.0)])
def test_arithmetic_identities_refresh_depth(tmp_path,config,operation,constant):
    path=lower(tmp_path,config,f'''%c = arith.constant {constant} : f64
%a = arith.{operation} %x, %c : f64
return %a : f64''',flags=["--ppml-fhe-optimize"])
    result=contract(path,config)
    assert result["multiplicative_depth"]==0
    assert result["resources"]["plaintext_multiplications"]==0
    assert result["resources"]["materializations"]==(1 if operation=="mulf" and constant==0 else 0)


def test_random_constant_materializations_are_not_cse(tmp_path,config):
    path=lower(tmp_path,config,'''%a = arith.constant 0.25 : f64
%b = arith.constant 0.25 : f64
return %a, %b : f64, f64''',outputs="(f64, f64)",flags=["--ppml-fhe-optimize"])
    assert contract(path,config)["resources"]["materializations"]==2


def test_explicitly_disabled_function_is_untouched(tmp_path,config):
    path=lower(tmp_path,config,'''%one = arith.constant 1.0 : f64
%r = arith.mulf %x, %one : f64
return %r : f64''',flags=["--ppml-fhe-optimize"],attrs="attributes {ppml.fhe_enabled = false}")
    assert "heir.rlwemul" in path.read_text()
    assert "ppml.resources" not in path.read_text()


def test_unmarked_plain_function_is_untouched(tmp_path,config):
    path=tmp_path/"plain.mlir"
    path.write_text("module { func.func @plain(%x: f64) -> f64 { return %x : f64 } }")
    result,_=command([Path(config["native_build"])/"bin/crossfhe-opt",path,"--ppml-fhe-optimize"],config)
    assert "ppml.resources" not in result
    assert "@plain" in result


def test_balancing_is_explicit_and_lowers_chain_depth(tmp_path,config):
    args=", ".join(f"%x{i}: f64" for i in range(8))
    body="%v1 = arith.mulf %x0, %x1 : f64\n"+"\n".join(f"%v{i} = arith.mulf %v{i-1}, %x{i} : f64" for i in range(2,8))+"\nreturn %v7 : f64"
    plain=lower(tmp_path,config,body,args=args,flags=["--ppml-fhe-optimize"])
    assert contract(plain,config)["multiplicative_depth"]==7
    balanced=lower(tmp_path,config,body,args=args,flags=["--ppml-fhe-balance-mul"])
    result=contract(balanced,config)
    assert result["multiplicative_depth"]==3
    assert result["resources"]["ciphertext_multiplications"]==7
    assert "ppml.reassociated = true" in balanced.read_text()


RELU='''%z = arith.constant 0.0 : f64
%c = arith.cmpf olt, %x, %z : f64
%r = arith.select %c, %z, %x : f64
return %r : f64'''


def test_bounded_relu_removes_switching_and_records_local_error(tmp_path,config):
    path=lower(tmp_path,config,RELU,flags=["--ppml-scfhe-approx-relu=bound=1 max-error=0.0625","--ppml-fhe-optimize"])
    result=contract(path,config)
    assert result["logic_comparisons"]==0
    assert result["multiplicative_depth"]==2
    assert result["relu_local_error_bound"]==0.0625
    assert result["relu_approximation_count"]==1
    assert "heir.lut_lt" not in path.read_text()


@pytest.mark.parametrize("options",["bound=1 max-error=0.001","bound=0 max-error=1","bound=-1 max-error=1"])
def test_invalid_approximation_budget_is_rejected(tmp_path,config,options):
    with pytest.raises(RuntimeError,match="max-error >= bound/16"):
        lower(tmp_path,config,RELU,flags=[f"--ppml-scfhe-approx-relu={options}"])


def test_optimizer_rejects_unknown_operations_even_if_unused(tmp_path,config):
    path=lower(tmp_path,config,"return %x : f64")
    source=path.read_text().replace("    return", "    %bad = builtin.unrealized_conversion_cast %arg0 : !heir.rlwecipher<1 x f64> to !heir.rlwecipher<1 x f64>\n    return")
    path.write_text(source)
    with pytest.raises(RuntimeError,match="outside the scalar FHE optimization contract"):
        command([Path(config["native_build"])/"bin/crossfhe-opt",path,"--ppml-fhe-optimize"],config)
