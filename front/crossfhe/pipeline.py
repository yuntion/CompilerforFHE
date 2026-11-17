from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import warnings

import numpy as np
import onnx
from onnx import external_data_helper
from .frontend import FrontendError, import_onnx
from .packed import import_packed_onnx

ROOT = Path(__file__).resolve().parents[2]


def read_config(path=None):
    path = Path(path or os.environ.get("CROSSFHE_CONFIG",ROOT/".crossfhe/toolchain.json"))
    if not path.is_file():
        raise RuntimeError(f"missing toolchain config {path}; run scripts/bootstrap.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def environment(config):
    env = os.environ.copy()
    directories = {str(Path(config[name]).parent) for name in ("cxx","cmake","ninja")}
    env["PATH"] = os.pathsep.join(sorted(directories)) + os.pathsep + env.get("PATH","")
    return env


def command(args, config, *, cwd=None, log=None, timeout=1800):
    started = time.perf_counter()
    result = subprocess.run([str(a) for a in args],cwd=cwd,env=environment(config),
                            capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
    text = result.stdout + result.stderr
    if log:
        Path(log).write_text(text,encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args[0]}\n{text[-8000:]}")
    return result.stdout, time.perf_counter()-started


def _native(config, name):
    suffix = ".exe" if os.name=="nt" else ""
    tool = Path(config["native_build"])/"bin"/(name+suffix)
    if not tool.is_file(): raise RuntimeError(f"native tool missing: {tool}")
    return tool


def compile_onnx(model_path, output, *, config=None, backend="openfhe-cross", max_operations=50000,
                 optimization="scalar", relu_bound=None, relu_max_error=None, packing="none"):
    config = config or read_config()
    if backend not in ("openfhe-cross","seal-ckks","phantom-ckks"):
        raise ValueError("implemented backends: openfhe-cross, seal-ckks, phantom-ckks")
    if backend=="phantom-ckks":
        for name in ("phantom_source", "phantom_library", "cuda_compiler"):
            if not config.get(name):
                raise ValueError("phantom-ckks is not configured; run scripts/bootstrap.py --with-phantom")
    if optimization not in ("none", "scalar", "depth"):
        raise ValueError("optimization must be none, scalar or depth")
    if packing not in ("none", "diagonal", "bsgs", "auto"):
        raise ValueError("packing must be none, diagonal, bsgs or auto")
    if packing!="none" and (backend not in ("seal-ckks","phantom-ckks") or relu_bound is not None or relu_max_error is not None):
        raise ValueError("packed MatVec currently requires seal-ckks or phantom-ckks and no ReLU approximation")
    if (relu_bound is None) != (relu_max_error is None):
        raise ValueError("ReLU approximation requires both relu_bound and relu_max_error")
    if relu_bound is not None and (not np.isfinite([relu_bound, relu_max_error]).all()
            or relu_bound <= 0 or relu_max_error < relu_bound/16):
        raise ValueError("ReLU approximation requires finite bound > 0 and max error >= bound/16")
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    logs = output/"logs"; logs.mkdir(exist_ok=True)
    status=output/"compilation_status.json"
    status.write_text('{"state":"compiling"}\n',encoding="utf-8")
    model = onnx.load(str(model_path),load_external_data=True)
    external_data_helper.convert_model_from_external_data(model)
    onnx.save_model(model,str(output/"model.onnx"))
    opt, translate = _native(config,"crossfhe-opt"), _native(config,"crossfhe-translate")
    if packing=="none":
        mlir, manifest = import_onnx(output/"model.onnx",max_operations=max_operations)
        (output/"01_arith.mlir").write_text(mlir,encoding="utf-8")
        steps = [([opt,output/"01_arith.mlir","--ppml-to-scfhe","--verify-each","-o",output/"02_scfhe.mlir"],"scfhe")]
        lowering = [opt, output/"02_scfhe.mlir"]
        if relu_bound is not None:
            lowering += [f"--ppml-scfhe-approx-relu=bound={relu_bound:.17g} max-error={relu_max_error:.17g}"]
        lowering += ["--ppml-to-heir", "--ppml-fhe-refresh-contract", "--verify-each", "-o", output/"03_fhe.mlir"]
        steps.append((lowering,"fhe"))
    else:
        mlir, manifest = import_packed_onnx(output/"model.onnx",max_elements=max_operations)
        (output/"01_matvec.mlir").write_text(mlir,encoding="utf-8")
        steps = [([opt,output/"01_matvec.mlir",f"--ppml-fhe-matvec=strategy={packing}",
                   "--verify-each","-o",output/"03_fhe.mlir"],"matvec")]
    optimizations = ["--ppml-fhe-refresh-contract"] if optimization=="none" else ["--ppml-fhe-optimize"]
    if optimization=="depth": optimizations += ["--ppml-fhe-balance-mul"]
    steps.append(([opt,output/"03_fhe.mlir",*optimizations,"--verify-each","-o",output/"04_optimized.mlir"],"optimization"))
    times = {}
    for args,name in steps:
        _,times[name] = command(args,config,log=logs/(name+".log"))
    contracts = {}
    for stage, name in (("03_fhe.mlir", "before"), ("04_optimized.mlir", "after")):
        path=output/f"resources_{name}.json"
        _,times[f"contract_{name}"] = command([translate,output/stage,f"--backend={backend if packing!='none' else 'openfhe-cross'}",
            "--emit-contract","-o",path],config,log=logs/f"contract_{name}.log")
        contracts[name]=json.loads(path.read_text())
    if contracts["after"]["logic_comparisons"] and backend!="openfhe-cross":
        raise FrontendError("model needs logic FHE; selected backend supports CKKS only; choose openfhe-cross")
    _,times["codegen"] = command([translate,output/"04_optimized.mlir",f"--backend={backend}",
        "-o",output/"model.hpp"],config,log=logs/"codegen.log")
    build = output/"build"
    args = [config["cmake"],"-S",ROOT/"backend","-B",build,"-G","Ninja",
            "-DCMAKE_BUILD_TYPE=Release",f"-DCMAKE_MAKE_PROGRAM={config['ninja']}",
            f"-DCMAKE_CXX_COMPILER={config['cxx']}", f"-DCROSSFHE_BACKEND={backend}",
            f"-DCROSSFHE_MODEL_HEADER={(output/'model.hpp').as_posix()}"]
    if backend=="phantom-ckks":
        args += [f"-DCROSSFHE_PHANTOM_SOURCE={config['phantom_source']}",
                 f"-DCROSSFHE_PHANTOM_LIBRARY={config['phantom_library']}",
                 f"-DCMAKE_CUDA_COMPILER={config['cuda_compiler']}",
                 f"-DCMAKE_CUDA_HOST_COMPILER={config['cxx']}",
                 f"-DCMAKE_CUDA_ARCHITECTURES={config.get('cuda_architectures','native')}"]
    else:
        args.append(f"-DSEAL_DIR={config['seal_dir']}" if backend=="seal-ckks" else f"-DOpenFHE_DIR={config['openfhe_dir']}")
    _,times["configure"] = command(args,config,log=logs/"configure.log")
    _,times["cxx_build"] = command([config["cmake"],"--build",build,"--target","model-runner","--parallel","2"],config,log=logs/"build.log")
    manifest.update({"schema_version":1,"backend":backend,"compile_seconds":times,
                     "model_sha256":hashlib.sha256((output/"model.onnx").read_bytes()).hexdigest(),
                     "optimization":optimization, "packing":packing, "resource_analysis":contracts,
                     "source_logic_comparisons":manifest["logic_comparisons"],
                     "logic_comparisons":contracts["after"]["logic_comparisons"],
                     "multiplicative_depth":contracts["after"]["multiplicative_depth"],
                     "approximation":{k:v for k,v in contracts["after"].items() if k.startswith("relu_")},
                     "upstream_revisions":config.get("revisions",{})})
    suffix=".exe" if os.name=="nt" else ""
    manifest["artifact_hashes"]={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ("model.onnx","model.hpp","03_fhe.mlir","04_optimized.mlir",
                     "resources_before.json","resources_after.json",f"build/bin/model-runner{suffix}")}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    status.write_text('{"state":"ready"}\n',encoding="utf-8")
    return output


def export_torch(model, inputs, path, *, input_names=None):
    import torch
    if isinstance(inputs,torch.Tensor): inputs = (inputs,)
    inputs = tuple(inputs)
    if not all(isinstance(x,torch.Tensor) and x.dtype in (torch.float32,torch.float64) for x in inputs):
        raise FrontendError("export inputs must be floating-point torch tensors")
    model = model.cpu().eval()
    inputs = tuple(x.detach().cpu() for x in inputs)
    with torch.no_grad(): result = model(*inputs)
    outputs = (result,) if isinstance(result,torch.Tensor) else tuple(result)
    if not all(isinstance(x,torch.Tensor) for x in outputs):
        raise FrontendError("model outputs must be a tensor or tuple of tensors")
    names = input_names or [f"input_{i}" for i in range(len(inputs))]
    if len(names)!=len(inputs) or len(set(names))!=len(names): raise ValueError("invalid input names")
    output_names = [f"output_{i}" for i in range(len(outputs))]
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always",torch.jit.TracerWarning)
        torch.onnx.export(model,inputs,str(path),input_names=names,output_names=output_names,
                          opset_version=17,dynamo=False,export_params=True,do_constant_folding=True)
    for warning in captured:
        if issubclass(warning.category,torch.jit.TracerWarning) and "Converting a tensor to a Python" in str(warning.message):
            raise FrontendError("tensor-dependent Python control/data conversion would be frozen by ONNX tracing; use tensor operations or Where")
        warnings.warn_explicit(warning.message,warning.category,warning.filename,warning.lineno)
    arrays = {name:tensor.numpy() for name,tensor in zip(names,inputs)}
    references = {name:tensor.detach().cpu().numpy() for name,tensor in zip(output_names,outputs)}
    return arrays,references


def compile_torch(model, inputs, output, **options):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    arrays,reference = export_torch(model,inputs,output/"export.onnx")
    artifact = compile_onnx(output/"export.onnx",output,**options)
    return artifact,arrays,reference


def _write_values(path, values):
    values = np.asarray(values,dtype="<f8").reshape(-1)
    with Path(path).open("wb") as stream:
        stream.write(struct.pack("<Q",values.size)); stream.write(values.tobytes())


def _read_values(path):
    raw = Path(path).read_bytes()
    if len(raw)<8: raise RuntimeError("truncated decrypted values")
    count = struct.unpack("<Q",raw[:8])[0]
    if len(raw)!=8+count*8: raise RuntimeError("invalid decrypted values length")
    return np.frombuffer(raw[8:],dtype="<f8").copy()


def run_encrypted(artifact, inputs, *, reference=None, config=None, insecure_demo=False,
                  comparison_scale=1024.0, atol=0.005, rtol=0.005):
    config = config or read_config()
    artifact = Path(artifact).resolve()
    status=artifact/"compilation_status.json"
    if not status.is_file() or json.loads(status.read_text())["state"]!="ready":
        raise RuntimeError("artifact is not ready; compilation was incomplete or failed")
    manifest = json.loads((artifact/"manifest.json").read_text(encoding="utf-8"))
    for name,digest in manifest["artifact_hashes"].items():
        if hashlib.sha256((artifact/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"compiled artifact changed: {name}; compile again")
    if insecure_demo and manifest["backend"]!="openfhe-cross":
        raise ValueError("--insecure-demo applies only to OpenFHE; CKKS-only backends always use tc128")
    if not np.isfinite(comparison_scale) or comparison_scale<=0:
        raise ValueError("comparison scale must be finite and positive")
    if min(atol,rtol)<0 or not np.isfinite([atol,rtol]).all(): raise ValueError("invalid tolerances")
    expected_names = {i["name"] for i in manifest["inputs"]}
    if set(inputs)!=expected_names: raise ValueError(f"input names must be {sorted(expected_names)}")
    flat = []
    for info in manifest["inputs"]:
        value = np.asarray(inputs[info["name"]],dtype=np.float64)
        if list(value.shape)!=info["shape"]: raise ValueError(f"shape mismatch for {info['name']}")
        if not np.isfinite(value).all(): raise ValueError("inputs must be finite")
        flat.extend(value.flat)
    runs = artifact/"runs"; runs.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-",dir=runs))
    client,server,logs = run/"client",run/"server",run/"logs"
    logs.mkdir()
    suffix = ".exe" if os.name=="nt" else ""
    runner = artifact/"build/bin"/("model-runner"+suffix)
    phases = {}
    args = [runner,"keygen",client,server]
    if manifest["backend"]=="openfhe-cross":
        args += ["--comparison-scale",str(comparison_scale)]
        if insecure_demo: args.append("--insecure-demo")
    text,wall = command(args,config,log=logs/"keygen.log")
    phases["keygen"] = {"wall_seconds":wall,"runtime":json.loads(text.strip().splitlines()[-1])}
    _write_values(client/"input.bin",flat)
    _,wall = command([runner,"encrypt",client,server,client/"input.bin",server/"input.ct"],config,log=logs/"encrypt.log")
    phases["encrypt"] = {"wall_seconds":wall}
    text,wall = command([runner,"evaluate",server,server/"input.ct",server/"output.ct"],config,cwd=server,log=logs/"evaluate.log")
    phases["evaluate"] = {"wall_seconds":wall,"runtime":json.loads(text.strip().splitlines()[-1])}
    _,wall = command([runner,"decrypt",client,server/"output.ct",client/"output.bin"],config,log=logs/"decrypt.log")
    phases["decrypt"] = {"wall_seconds":wall}
    flat_output = _read_values(client/"output.bin")
    if not np.isfinite(flat_output).all(): raise RuntimeError("nonfinite decrypted outputs")
    if flat_output.size != sum(i["count"] for i in manifest["outputs"]):
        raise RuntimeError("decrypted output count mismatch")
    outputs = {i["name"]:flat_output[i["offset"]:i["offset"]+i["count"]].reshape(i["shape"]) for i in manifest["outputs"]}
    np.savez(client/"outputs.npz",**outputs)
    secret_free = not any("secret" in file.name.lower() or file.suffix==".npy" for file in server.rglob("*"))
    if not secret_free: raise RuntimeError("server artifacts contain private material")
    report = {"backend":manifest["backend"],"model_sha256":manifest["model_sha256"],
              "optimization":manifest["optimization"],"phases":phases,
              "server_has_no_secret_key":secret_free,"run_directory":str(run),
              "ciphertext_input_bytes":(server/"input.ct").stat().st_size,
              "ciphertext_output_bytes":(server/"output.ct").stat().st_size,
              "evaluation_bundle_bytes":sum(f.stat().st_size for f in server.iterdir() if f.is_file() and f.suffix in (".bin",".seal",".phantom"))}
    if reference is not None:
        if set(reference)!=set(outputs): raise ValueError("reference output names differ")
        errors = {}
        for name,value in outputs.items():
            expected = np.asarray(reference[name],dtype=np.float64)
            if expected.shape!=value.shape or not np.isfinite(expected).all(): raise ValueError("invalid reference output")
            errors[name] = {"max_abs_error":float(np.max(np.abs(value-expected))),
                            "matches":bool(np.allclose(value,expected,atol=atol,rtol=rtol))}
        report["validation"] = {"atol":atol,"rtol":rtol,"outputs":errors,
                                "passed":all(e["matches"] for e in errors.values())}
    (run/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    shutil.copyfile(run/"report.json",artifact/"latest_report.json")
    if reference is not None and not report["validation"]["passed"]:
        raise RuntimeError(f"encrypted result mismatch; see {run/'report.json'}")
    return outputs,report
