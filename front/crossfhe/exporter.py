"""PyTorch to ONNX export, model inputs and independent reference values."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

from .frontend import FrontendError


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
