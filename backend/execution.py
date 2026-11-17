"""Backend runner construction and encrypted execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

from .toolchain import ROOT, read_config, command, _native


BACKEND_DIRECTORIES = {"seal-ckks": "seal", "openfhe-cross": "openfhe", "phantom-ckks": "phantom"}


def backend_context(backend, config):
    """Fingerprint the trusted generator and backend sources used for a build."""
    directory = BACKEND_DIRECTORIES[backend]
    names = ["backend/CMakeLists.txt", f"backend/{directory}/CMakeLists.txt",
             f"backend/{directory}/{directory}_runtime.hpp",
             f"backend/{directory}/runner.{'cu' if directory == 'phantom' else 'cpp'}"]
    if directory == "phantom":
        names.append("backend/phantom/phantom_decode.hpp")
    if directory == "seal":
        libraries = sorted((Path(config["seal_dir"]).parent/"lib").glob("libseal*.a"))
    elif directory == "openfhe":
        libraries = sorted(Path(config["openfhe_dir"]).parent.glob("libOPENFHE*_static.a"))
    else:
        libraries = [Path(config["phantom_library"])]
    if not libraries:
        raise RuntimeError("cannot fingerprint backend libraries; check the configured library installation")
    result = {
        "backend": backend, "build_type": "Release",
        "library_revision": config.get("revisions", {}).get(directory),
        "native_translator_sha256": hashlib.sha256(_native(config, "crossfhe-translate").read_bytes()).hexdigest(),
        "cxx_sha256": hashlib.sha256(Path(config["cxx"]).read_bytes()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        "library_hashes": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries},
    }
    if directory == "phantom":
        result["cuda_compiler_sha256"] = hashlib.sha256(Path(config["cuda_compiler"]).read_bytes()).hexdigest()
        result["cuda_architectures"] = config.get("cuda_architectures", "native")
    return result


def build_runner(header, build, backend, config, logs, *, contract_header=None):
    """Build a runner from trusted generated code, optionally sharing a frozen contract ID."""
    header, build, logs = Path(header).resolve(), Path(build).resolve(), Path(logs).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    args = [config["cmake"], "-S", ROOT/"backend", "-B", build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={config['ninja']}",
            f"-DCMAKE_CXX_COMPILER={config['cxx']}", f"-DCROSSFHE_BACKEND={backend}",
            f"-DCROSSFHE_MODEL_HEADER={header.as_posix()}"]
    if backend == "phantom-ckks":
        args += [f"-DCROSSFHE_PHANTOM_SOURCE={config['phantom_source']}",
                 f"-DCROSSFHE_PHANTOM_LIBRARY={config['phantom_library']}",
                 f"-DCMAKE_CUDA_COMPILER={config['cuda_compiler']}",
                 f"-DCMAKE_CUDA_HOST_COMPILER={config['cxx']}",
                 f"-DCMAKE_CUDA_ARCHITECTURES={config.get('cuda_architectures', 'native')}"]
        if contract_header is not None:
            args.append(f"-DCROSSFHE_CONTRACT_HEADER={Path(contract_header).resolve().as_posix()}")
    else:
        args.append(f"-DSEAL_DIR={config['seal_dir']}" if backend == "seal-ckks" else f"-DOpenFHE_DIR={config['openfhe_dir']}")
    _, configure_seconds = command(args, config, log=logs/"configure.log")
    _, build_seconds = command([config["cmake"], "--build", build, "--target", "model-runner", "--parallel", "2"],
                               config, log=logs/"build.log")
    runner = build/"bin"/("model-runner.exe" if os.name == "nt" else "model-runner")
    if not runner.is_file():
        raise RuntimeError("C++ build produced no model-runner")
    return runner, {"configure": configure_seconds, "cxx_build": build_seconds}


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
