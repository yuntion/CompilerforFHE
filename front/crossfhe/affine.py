"""Parse one public ONNX affine transform without choosing an execution layout."""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .frontend import FrontendError, load_onnx


def import_affine_onnx(path: str | Path, *, max_elements=50000):
    """Return the public matrix, bias and model semantics for a single MatVec."""
    model = load_onnx(path)
    public = {a.name: numpy_helper.to_array(a) for a in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in public]
    if len(inputs) != 1 or len(model.graph.output) != 1:
        raise FrontendError("packed MatVec needs exactly one secret input and one output")
    info = inputs[0]
    tensor = info.type.tensor_type
    shape = tuple(d.dim_value for d in tensor.shape.dim)
    if tensor.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE) or not shape or min(shape) <= 0:
        raise FrontendError("packed input needs a positive static floating-point shape")
    secret = {info.name: ("input", shape)}
    matrix = bias = result_shape = None

    def coefficient(value):
        value = np.asarray(value)
        if value.size > max_elements or value.dtype.kind != "f":
            raise FrontendError("packed coefficients must be floats within the element limit")
        return value.astype(np.float64)

    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or len(node.output) != 1:
            raise FrontendError("packed path supports standard single-output ONNX operations")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        kind, args, out = node.op_type, list(node.input), node.output[0]
        if kind == "Constant" and "value" in attrs:
            public[out] = numpy_helper.to_array(attrs["value"])
        elif kind == "Identity" and len(args) == 1:
            if args[0] in public: public[out] = public[args[0]]
            elif args[0] in secret: secret[out] = secret[args[0]]
            else: raise FrontendError("unknown packed Identity input")
        elif kind == "Transpose" and len(args) == 1 and args[0] in public:
            public[out] = np.transpose(public[args[0]], attrs.get("perm"))
        elif kind in ("MatMul", "Gemm"):
            if matrix is not None or len(args) < 2 or len(args) > (3 if kind == "Gemm" else 2):
                raise FrontendError("packed path supports one affine MatVec only")
            left_secret, right_secret = args[0] in secret, args[1] in secret
            if left_secret == right_secret:
                raise FrontendError("packed MatVec requires one secret vector and one public matrix")
            name = args[0] if left_secret else args[1]
            stage, dims = secret[name]
            if stage != "input": raise FrontendError("packed MatVec input must be the secret vector")
            weight_name = args[1] if left_secret else args[0]
            if weight_name not in public: raise FrontendError("MatVec matrix must be public")
            weight = coefficient(public[weight_name])
            if weight.ndim != 2: raise FrontendError("packed MatVec matrix must have rank two")
            if kind == "Gemm":
                if not left_secret or len(dims) != 2 or dims[0] != 1 or attrs.get("transA", 0):
                    raise FrontendError("packed Gemm requires one row vector and transA=0")
                if attrs.get("transB", 0): weight = weight.T
                alpha, beta = float(attrs.get("alpha", 1)), float(attrs.get("beta", 1))
            else:
                alpha = beta = 1.0
            if left_secret:
                if dims not in ((weight.shape[0],), (1, weight.shape[0])):
                    raise FrontendError("packed MatMul needs a vector or a single row")
                matrix = weight.T * alpha
                result_shape = (weight.shape[1],) if len(dims) == 1 else (1, weight.shape[1])
            else:
                if dims not in ((weight.shape[1],), (weight.shape[1], 1)):
                    raise FrontendError("packed MatMul needs a vector or a single column")
                matrix = weight * alpha
                result_shape = (weight.shape[0],) if len(dims) == 1 else (weight.shape[0], 1)
            bias = np.zeros(matrix.shape[0], dtype=np.float64)
            if kind == "Gemm" and len(args) == 3 and args[2]:
                if args[2] not in public: raise FrontendError("Gemm bias must be public")
                bias = np.broadcast_to(coefficient(public[args[2]]), result_shape).reshape(-1).copy() * beta
            secret[out] = ("result", result_shape)
        elif kind == "Add" and len(args) == 2:
            secret_name = next((a for a in args if a in secret), None)
            other = args[1] if secret_name == args[0] else args[0]
            if secret_name is None or other not in public or secret[secret_name][0] != "result":
                raise FrontendError("packed Add supports a public bias after MatVec")
            bias = bias + np.broadcast_to(coefficient(public[other]), result_shape).reshape(-1)
            # Restrict to a single result chain: an older alias cannot be reused.
            secret = {k: v for k, v in secret.items() if v[0] == "input"}
            secret[out] = ("result", result_shape)
        else:
            raise FrontendError(f"unsupported packed operation {kind}; use the scalar path for general models")

    output = model.graph.output[0]
    if matrix is None or output.name not in secret or secret[output.name][0] != "result":
        raise FrontendError("packed model must return its affine MatVec result")
    # Check the composed coefficients once, including scaling/addition overflow.
    if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
        raise FrontendError("packed affine coefficients must be finite")
    rows, cols = matrix.shape
    return matrix, bias, {
        "inputs": [{"name": info.name, "shape": list(shape), "offset": 0, "count": cols, "onnx_dtype": tensor.elem_type}],
        "outputs": [{"name": output.name, "shape": list(result_shape), "offset": 0, "count": rows}],
        "logic_comparisons": 0, "scalar_operations": 0, "model_parameters": "public",
    }
