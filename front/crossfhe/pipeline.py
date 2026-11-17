"""End-to-end compilation driver joining model imports and FHE backends."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import external_data_helper

from .frontend import FrontendError, import_onnx
from .exporter import export_torch
from backend.packed import import_packed_onnx
from backend.execution import (
    BACKEND_DIRECTORIES, backend_context, build_runner, run_encrypted,
    _read_values, _write_values,
)
from backend.toolchain import ROOT, _native, command, environment, read_config


def compile_onnx(model_path, output, *, config=None, backend="openfhe-cross", max_operations=50000,
                 optimization="scalar", relu_bound=None, relu_max_error=None, packing="none"):
    config = config or read_config()
    if backend not in ("openfhe-cross","seal-ckks","phantom-ckks"):
        raise ValueError("implemented backends: openfhe-cross, seal-ckks, phantom-ckks")
    if backend=="phantom-ckks":
        for name in ("phantom_source", "phantom_library", "cuda_compiler"):
            if not config.get(name):
                raise ValueError("phantom-ckks is not configured; run scripts/bootstrap.py --enable-cuda")
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
        manifest.update(numeric_semantics="float32 model coefficients promoted to CKKS f64",
                        layout="one-scalar-per-ciphertext")
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
    _,times["dag"] = command([translate,output/"04_optimized.mlir",f"--backend={backend}",
        "--emit-dag","-o",output/"dag.json"],config,log=logs/"dag.log")
    _, build_times = build_runner(output/"model.hpp", output/"build", backend, config, logs)
    times.update(build_times)
    manifest.update({"schema_version":1,"backend":backend,"compile_seconds":times,
                     "model_sha256":hashlib.sha256((output/"model.onnx").read_bytes()).hexdigest(),
                     "optimization":optimization, "packing":packing, "resource_analysis":contracts,
                     "source_logic_comparisons":manifest["logic_comparisons"],
                     "logic_comparisons":contracts["after"]["logic_comparisons"],
                     "multiplicative_depth":contracts["after"]["multiplicative_depth"],
                     "approximation":{k:v for k,v in contracts["after"].items() if k.startswith("relu_")},
                     "upstream_revisions":config.get("revisions",{}),
                     "backend_context":backend_context(backend, config)})
    suffix=".exe" if os.name=="nt" else ""
    manifest["artifact_hashes"]={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ("model.onnx","model.hpp","03_fhe.mlir","04_optimized.mlir",
                     "resources_before.json","resources_after.json","dag.json",f"build/bin/model-runner{suffix}")}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    status.write_text('{"state":"ready"}\n',encoding="utf-8")
    return output


def compile_torch(model, inputs, output, **options):
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    arrays,reference = export_torch(model,inputs,output/"export.onnx")
    artifact = compile_onnx(output/"export.onnx",output,**options)
    return artifact,arrays,reference
