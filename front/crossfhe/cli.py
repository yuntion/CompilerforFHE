import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from .demos import DEMOS,make_demo
from .pipeline import compile_onnx,compile_torch,export_torch,read_config,run_encrypted


def main(argv=None):
    parser=argparse.ArgumentParser(description="ONNX/MLIR to executable CKKS + logic FHE")
    parser.add_argument("--config",type=Path,help="toolchain.json from bootstrap.py")
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("compile","demo"):
        command=commands.add_parser(name)
        command.add_argument("model",choices=DEMOS if name=="demo" else None)
        command.add_argument("--output",type=Path,required=True)
        command.add_argument("--backend",choices=("openfhe-cross","seal-ckks"),default="openfhe-cross")
        command.add_argument("--max-operations",type=int,default=50000)
        command.add_argument("--optimization",choices=("none","scalar","depth"),default="scalar")
        command.add_argument("--packing",choices=("none","diagonal","bsgs","auto"),default="none",
                             help="Explicit CPU CKKS SIMD path for a single public affine MatVec (SEAL)")
        command.add_argument("--relu-bound",type=float,help="Declared absolute bound on every approximated ReLU input")
        command.add_argument("--relu-max-error",type=float,help="Accepted local polynomial approximation error")
        if name=="demo": _run_options(command)
    runner=commands.add_parser("run")
    runner.add_argument("artifact",type=Path)
    runner.add_argument("--inputs",type=Path,required=True,help="NPZ keyed by ONNX input name")
    runner.add_argument("--reference",type=Path,help="optional NPZ keyed by ONNX output name")
    _run_options(runner)
    exporter=commands.add_parser("export")
    exporter.add_argument("--factory",required=True,help="module:function returning a PyTorch nn.Module")
    exporter.add_argument("--inputs",type=Path,required=True,help="NPZ tensors, passed in NPZ name order")
    exporter.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=="export":
            module,function=args.factory.split(":",1)
            model=getattr(importlib.import_module(module),function)()
            import torch
            with np.load(args.inputs,allow_pickle=False) as data:
                names=list(data.files); tensors=tuple(torch.from_numpy(data[n]) for n in names)
            arrays,reference=export_torch(model,tensors,args.output,input_names=names)
            np.savez(args.output.with_suffix(".inputs.npz"),**arrays)
            np.savez(args.output.with_suffix(".reference.npz"),**reference)
            print(args.output.resolve()); return 0
        config=read_config(args.config)
        if args.command=="compile":
            artifact=compile_onnx(args.model,args.output,config=config,backend=args.backend,max_operations=args.max_operations,
                optimization=args.optimization,relu_bound=args.relu_bound,relu_max_error=args.relu_max_error,packing=args.packing)
            print(artifact); return 0
        if args.command=="demo":
            model,inputs=make_demo(args.model)
            artifact,arrays,reference=compile_torch(model,inputs,args.output,config=config,backend=args.backend,max_operations=args.max_operations,
                optimization=args.optimization,relu_bound=args.relu_bound,relu_max_error=args.relu_max_error,packing=args.packing)
        else:
            artifact=args.artifact
            with np.load(args.inputs,allow_pickle=False) as data: arrays={n:data[n] for n in data.files}
            reference=None
            if args.reference:
                with np.load(args.reference,allow_pickle=False) as data: reference={n:data[n] for n in data.files}
        outputs,report=run_encrypted(artifact,arrays,reference=reference,config=config,
            insecure_demo=args.insecure_demo,comparison_scale=args.comparison_scale,atol=args.atol,rtol=args.rtol)
        print(json.dumps({"outputs":{n:x.tolist() for n,x in outputs.items()},"report":report},indent=2))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        print(f"CrossFHE error: {exc}",file=sys.stderr); return 1


def _run_options(parser):
    parser.add_argument("--insecure-demo",action="store_true",help="explicit TOY parameters, for functional tests only")
    parser.add_argument("--comparison-scale",type=float,default=1024.0)
    parser.add_argument("--atol",type=float,default=0.005)
    parser.add_argument("--rtol",type=float,default=0.005)


if __name__=="__main__": raise SystemExit(main())
