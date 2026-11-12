"""Structured ONNX import to verified scalar func/arith MLIR.

Scalarization is deliberately a baseline. No packing, fusion or scheduling
optimization runs here. Only public expressions are evaluated at compile time.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import helper, numpy_helper


class FrontendError(ValueError):
    pass


@dataclass(frozen=True)
class Scalar:
    name: str | None = None
    value: float | int | bool | None = None
    kind: str = "f64"

    @property
    def public(self):
        return self.name is None


def _quote(text):
    return '"' + "".join(chr(b) if 32 <= b < 127 and b not in (34, 92)
                         else f"\\{b:02X}" for b in text.encode("utf-8")) + '"'


def _array(values):
    values = np.asarray(values)
    return np.array([Scalar(value=v.item(), kind="i1" if isinstance(v, np.bool_) else "f64")
                     for v in values.flat], dtype=object).reshape(values.shape)


class Importer:
    def __init__(self, model, max_operations=50000, max_elements=50000):
        self.model = model
        self.max_operations, self.max_elements = max_operations, max_elements
        self.lines, self.args, self.input_info, self.output_info = [], [], [], []
        self.values, self.constants = {}, {}
        self.source = ""
        self.counter = 0
        self.logic_count = 0

    def check_size(self, shape):
        if math.prod(shape) > self.max_elements:
            raise FrontendError(f"tensor {shape} exceeds scalar baseline limit {self.max_elements}")

    def fresh(self):
        if self.counter >= self.max_operations:
            raise FrontendError(f"scalar operation limit {self.max_operations} exceeded")
        name = f"%v{self.counter}"
        self.counter += 1
        return name

    def ref(self, value):
        if not value.public:
            return value.name
        if value.kind == "i1":
            key, literal = ("i1", bool(value.value)), str(bool(value.value)).lower()
        else:
            number = float(value.value)
            if not math.isfinite(number):
                raise FrontendError("NaN/Inf public values are not supported")
            key = ("f64", struct.pack(">d", number))
            literal = "0x" + key[1].hex().upper()
        if key not in self.constants:
            name = self.fresh()
            self.lines.append(f"    {name} = arith.constant {literal} : {value.kind}")
            self.constants[key] = name
        return self.constants[key]

    def binary(self, kind, a, b):
        if a.kind != "f64" or b.kind != "f64":
            raise FrontendError("arithmetic on boolean tensors is unsupported")
        if a.public and b.public:
            op = {"addf": lambda x, y: x+y, "subf": lambda x, y: x-y,
                  "mulf": lambda x, y: x*y}[kind]
            return Scalar(value=op(a.value, b.value))
        x, y, name = self.ref(a), self.ref(b), self.fresh()
        self.lines.append(f"    {name} = arith.{kind} {x}, {y} "
                          f"{{ppml.source = {_quote(self.source)}}} : f64")
        return Scalar(name=name)

    def cmp(self, a, b, predicate="olt"):
        if a.kind != "f64" or b.kind != "f64":
            raise FrontendError("only floating-point ordering is supported")
        if a.public and b.public:
            result = a.value < b.value if predicate == "olt" else a.value > b.value
            return Scalar(value=result, kind="i1")
        x, y, name = self.ref(a), self.ref(b), self.fresh()
        self.lines.append(f"    {name} = arith.cmpf {predicate}, {x}, {y} "
                          f"{{ppml.source = {_quote(self.source)}}} : f64")
        self.logic_count += 1
        return Scalar(name=name, kind="i1")

    def select(self, condition, a, b):
        if condition.kind != "i1" or a.kind != "f64" or b.kind != "f64":
            raise FrontendError("select requires a boolean condition and numeric branches")
        if condition.public:
            return a if condition.value else b
        c, x, y, name = self.ref(condition), self.ref(a), self.ref(b), self.fresh()
        self.lines.append(f"    {name} = arith.select {c}, {x}, {y} "
                          f"{{ppml.source = {_quote(self.source)}}} : f64")
        return Scalar(name=name)

    def elementwise(self, function, *arrays):
        arrays = np.broadcast_arrays(*arrays)
        self.check_size(arrays[0].shape)
        out = np.empty(arrays[0].shape, dtype=object)
        for index in np.ndindex(out.shape):
            out[index] = function(*(array[index] for array in arrays))
        return out

    def public(self, tensor, role="shape/attribute"):
        if any(not v.public for v in tensor.flat):
            raise FrontendError(f"{role} must be public and statically known")
        return np.array([v.value for v in tensor.flat]).reshape(tensor.shape)

    def integers(self, tensor, role):
        raw = self.public(tensor, role)
        if np.any(raw != np.floor(raw)):
            raise FrontendError(f"{role} must contain integers")
        return raw.astype(np.int64)

    def sum(self, values):
        result = Scalar(value=0.0)
        for value in values:
            result = self.binary("addf", result, value)
        return result

    def matmul(self, a, b):
        if a.ndim < 1 or b.ndim < 1:
            raise FrontendError("MatMul requires rank >= 1")
        av, bv = a.ndim == 1, b.ndim == 1
        if av: a = a[None, :]
        if bv: b = b[:, None]
        if a.shape[-1] != b.shape[-2]:
            raise FrontendError("MatMul contraction dimensions differ")
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a = np.broadcast_to(a, batch + a.shape[-2:])
        b = np.broadcast_to(b, batch + b.shape[-2:])
        shape = batch + (a.shape[-2], b.shape[-1])
        self.check_size(shape)
        out = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            prefix, row, col = index[:-2], index[-2], index[-1]
            out[index] = self.sum(self.binary("mulf", a[prefix+(row,k)], b[prefix+(k,col)])
                                  for k in range(a.shape[-1]))
        if av: out = np.squeeze(out, axis=-2)
        if bv: out = np.squeeze(out, axis=-1)
        return out

    def reduce(self, x, axes, keepdims, mean=False):
        if any(not -x.ndim <= int(a) < x.ndim for a in axes):
            raise FrontendError("reduction axis out of range")
        axes = tuple(sorted({int(a) % x.ndim for a in axes}))
        remaining = tuple(i for i in range(x.ndim) if i not in axes)
        shape = tuple(x.shape[i] for i in remaining)
        out = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            full = [slice(None)] * x.ndim
            for dim, offset in zip(remaining, index): full[dim] = offset
            values = np.asarray(x[tuple(full)],dtype=object).flat
            result = self.sum(values)
            if mean:
                count = math.prod(x.shape[i] for i in axes)
                if not count: raise FrontendError("empty mean")
                result = self.binary("mulf", result, Scalar(value=1.0/count))
            out[index] = result
        if keepdims:
            out = out.reshape(tuple(1 if i in axes else x.shape[i] for i in range(x.ndim)))
        return out

    def conv(self, x, w, bias, attrs):
        if x.ndim != 4 or w.ndim != 4:
            raise FrontendError("Conv baseline supports NCHW 2D convolution only")
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", b""):
            raise FrontendError("Conv auto_pad must be resolved to explicit pads")
        strides, dilations = attrs.get("strides", [1,1]), attrs.get("dilations", [1,1])
        pads, groups = attrs.get("pads", [0,0,0,0]), int(attrs.get("group", 1))
        n, channels, height, width = x.shape
        outputs, per_group, kh, kw = w.shape
        if groups <= 0 or channels != per_group * groups or outputs % groups:
            raise FrontendError("invalid grouped Conv shape")
        oh = (height+pads[0]+pads[2]-dilations[0]*(kh-1)-1)//strides[0]+1
        ow = (width+pads[1]+pads[3]-dilations[1]*(kw-1)-1)//strides[1]+1
        if min(oh,ow) <= 0: raise FrontendError("empty Conv output")
        shape = (n, outputs, oh, ow)
        self.check_size(shape)
        out = np.empty(shape, dtype=object)
        for batch, channel, row, col in np.ndindex(shape):
            products = []
            group = channel // (outputs//groups)
            for c, r, s in product(range(per_group), range(kh), range(kw)):
                yy, xx = row*strides[0]-pads[0]+r*dilations[0], col*strides[1]-pads[1]+s*dilations[1]
                if 0 <= yy < height and 0 <= xx < width:
                    products.append(self.binary("mulf", x[batch,group*per_group+c,yy,xx], w[channel,c,r,s]))
            value = self.sum(products)
            out[batch,channel,row,col] = self.binary("addf", value, bias[channel]) if bias is not None else value
        return out

    def pool(self, x, attrs, maximum=False):
        if x.ndim != 4 or attrs.get("ceil_mode", 0) or attrs.get("auto_pad", b"NOTSET") != b"NOTSET":
            raise FrontendError("pooling requires NCHW, floor mode and explicit pads")
        if attrs.get("dilations", [1,1]) != [1,1]:
            raise FrontendError("dilated MaxPool is unsupported")
        kernel, strides, pads = attrs["kernel_shape"], attrs.get("strides", [1,1]), attrs.get("pads", [0,0,0,0])
        n,c,h,w = x.shape
        oh = (h+pads[0]+pads[2]-kernel[0])//strides[0]+1
        ow = (w+pads[1]+pads[3]-kernel[1])//strides[1]+1
        if min(oh,ow) <= 0: raise FrontendError("empty pooling output")
        shape = (n,c,oh,ow)
        self.check_size(shape)
        out = np.empty(shape, dtype=object)
        for batch,channel,row,col in np.ndindex(shape):
            values = []
            for r,s in product(range(kernel[0]), range(kernel[1])):
                yy,xx = row*strides[0]-pads[0]+r, col*strides[1]-pads[1]+s
                if 0 <= yy < h and 0 <= xx < w: values.append(x[batch,channel,yy,xx])
                elif not maximum and attrs.get("count_include_pad", 0): values.append(Scalar(value=0.0))
            if not values: raise FrontendError("empty pooling window")
            if maximum:
                value = values[0]
                for other in values[1:]: value = self.select(self.cmp(value,other),other,value)
            else:
                value = self.binary("mulf", self.sum(values), Scalar(value=1.0/len(values)))
            out[batch,channel,row,col] = value
        return out

    def node(self, node):
        if node.domain not in ("", "ai.onnx"):
            raise FrontendError(f"custom ONNX domain {node.domain!r} has no registered lowering")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        args = [self.values[n] if n else None for n in node.input]
        op = node.op_type
        x = args[0] if args else None
        if op == "Constant":
            if "value" not in attrs: raise FrontendError("Constant requires tensor value")
            return [_array(numpy_helper.to_array(attrs["value"]))]
        if op == "Identity": return [x]
        if op in ("Add", "Sub", "Mul"):
            kind = {"Add":"addf", "Sub":"subf", "Mul":"mulf"}[op]
            return [self.elementwise(lambda a,b: self.binary(kind,a,b), *args)]
        if op == "Neg": return [self.elementwise(lambda a: self.binary("subf",Scalar(value=0.0),a),x)]
        if op == "Div":
            denominator = self.public(args[1], "Div denominator")
            if np.any(denominator == 0): raise FrontendError("division by zero")
            return [self.elementwise(lambda a,b: self.binary("mulf",a,b),x,_array(1.0/denominator))]
        if op == "Pow":
            exponents = self.public(args[1], "Pow exponent")
            if np.any(exponents < 0) or np.any(exponents > 8) or np.any(exponents != np.floor(exponents)):
                raise FrontendError("Pow supports public integer exponents 0..8")
            def power(a,b):
                result = Scalar(value=1.0)
                for _ in range(int(b.value)): result = self.binary("mulf",result,a)
                return result
            return [self.elementwise(power,*args)]
        if op == "MatMul": return [self.matmul(*args)]
        if op == "Gemm":
            if any(a.ndim != 2 for a in args[:2]): raise FrontendError("Gemm requires matrices")
            a,b = args[:2]
            result = self.matmul(a.T if attrs.get("transA",0) else a, b.T if attrs.get("transB",0) else b)
            result = self.elementwise(lambda v: self.binary("mulf",v,Scalar(value=attrs.get("alpha",1.0))),result)
            if len(args)>2 and args[2] is not None:
                bias = self.elementwise(lambda v: self.binary("mulf",v,Scalar(value=attrs.get("beta",1.0))),args[2])
                result = self.elementwise(lambda a,b: self.binary("addf",a,b),result,bias)
            return [result]
        if op == "Conv": return [self.conv(x,args[1],args[2] if len(args)>2 else None,attrs)]
        if op in ("AveragePool", "MaxPool"):
            if len(node.output)>1 and node.output[1]: raise FrontendError("MaxPool indices are unsupported")
            return [self.pool(x,attrs,op=="MaxPool")]
        if op == "GlobalAveragePool": return [self.reduce(x,range(2,x.ndim),True,True)]
        if op in ("ReduceMean", "ReduceSum"):
            axes = self.integers(args[1],"reduction axes").tolist() if len(args)>1 and args[1] is not None else attrs.get("axes", list(range(x.ndim)))
            if not axes:
                if attrs.get("noop_with_empty_axes",0): return [x]
                axes = list(range(x.ndim))
            return [self.reduce(x,axes,attrs.get("keepdims",1),op=="ReduceMean")]
        if op == "Reshape":
            shape = self.integers(args[1],"Reshape shape").tolist()
            if not attrs.get("allowzero",0): shape = [x.shape[i] if v==0 else v for i,v in enumerate(shape)]
            out = x.reshape(shape)
            self.check_size(out.shape)
            return [out]
        if op == "Flatten":
            axis = attrs.get("axis",1)
            if axis<0: axis += x.ndim
            if not 0<=axis<=x.ndim: raise FrontendError("invalid Flatten axis")
            return [x.reshape(math.prod(x.shape[:axis]), math.prod(x.shape[axis:]))]
        if op == "Transpose": return [x.transpose(attrs.get("perm",list(reversed(range(x.ndim)))))]
        if op == "Shape":
            return [_array(np.array(x.shape[attrs.get("start",0):attrs.get("end",x.ndim)],dtype=np.int64))]
        if op in ("Squeeze", "Unsqueeze"):
            axes = self.integers(args[1],"axes").tolist() if len(args)>1 else attrs.get("axes")
            if op == "Squeeze": return [np.squeeze(x, axis=None if axes is None else tuple(axes))]
            return [np.expand_dims(x, tuple(axes))]
        if op == "Concat": return [np.concatenate(args,axis=attrs["axis"])]
        if op == "Gather": return [np.take(x,self.integers(args[1],"Gather indices"),axis=attrs.get("axis",0))]
        if op == "BatchNormalization":
            if attrs.get("training_mode",0) or len(node.output)!=1:
                raise FrontendError("only inference BatchNormalization is supported")
            scale,bias,mean,var = [self.public(a,"BatchNorm parameter") for a in args[1:]]
            denominator = var+attrs.get("epsilon",1e-5)
            if np.any(denominator<=0): raise FrontendError("invalid BatchNorm variance")
            shape = (1,-1)+(1,)*(x.ndim-2)
            factor = scale / np.sqrt(denominator)
            shift = bias - mean*factor
            result = self.elementwise(lambda a,b: self.binary("mulf",a,b),x,_array(factor.reshape(shape)))
            return [self.elementwise(lambda a,b: self.binary("addf",a,b),result,_array(shift.reshape(shape)))]
        if op in ("Less", "Greater"):
            return [self.elementwise(lambda a,b: self.cmp(a,b,"olt" if op=="Less" else "ogt"),*args)]
        if op == "Where": return [self.elementwise(self.select,*args)]
        if op == "Relu":
            zero = Scalar(value=0.0)
            return [self.elementwise(lambda a: self.select(self.cmp(a,zero),zero,a),x)]
        if op in ("Max", "Min"):
            result = x
            for other in args[1:]:
                result = self.elementwise(lambda a,b: self.select(self.cmp(a,b),b,a) if op=="Max"
                                          else self.select(self.cmp(a,b),a,b),result,other)
            return [result]
        if op == "Cast":
            if attrs["to"] not in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE):
                raise FrontendError("secret integer casts require quantization semantics")
            if any(v.kind != "f64" for v in x.flat): raise FrontendError("boolean casts unsupported")
            return [x]
        raise FrontendError(f"{op} has no exact implemented FHE lowering; no plaintext fallback or implicit approximation")

    def run(self):
        initializers = {a.name for a in self.model.graph.initializer}
        for initializer in self.model.graph.initializer:
            value = numpy_helper.to_array(initializer)
            self.check_size(value.shape)
            self.values[initializer.name] = _array(value)
        for info in self.model.graph.input:
            if info.name in initializers: continue
            tensor = info.type.tensor_type
            if tensor.elem_type not in (onnx.TensorProto.FLOAT,onnx.TensorProto.DOUBLE):
                raise FrontendError(f"input {info.name} must be float32 or float64")
            if any(not dim.HasField("dim_value") or dim.dim_value<=0 for dim in tensor.shape.dim):
                raise FrontendError(f"input {info.name} needs a positive static shape")
            shape = tuple(dim.dim_value for dim in tensor.shape.dim)
            self.check_size(shape)
            offset = len(self.args)
            values = []
            for _ in range(math.prod(shape)):
                name = f"%arg{len(self.args)}"
                self.args.append(name)
                values.append(Scalar(name=name))
            self.values[info.name] = np.array(values,dtype=object).reshape(shape)
            self.input_info.append({"name":info.name,"shape":list(shape),"offset":offset,"count":len(values),"onnx_dtype":tensor.elem_type})
        if not self.args: raise FrontendError("model needs at least one secret input")
        for i,node in enumerate(self.model.graph.node):
            self.source = f"{i}:{node.name or node.op_type}:{node.op_type}"
            try:
                outputs = self.node(node)
                if len(outputs)!=len([n for n in node.output if n]):
                    raise FrontendError("unsupported optional/multiple outputs")
                for name,value in zip((n for n in node.output if n),outputs):
                    self.check_size(value.shape)
                    self.values[name] = value
            except (ValueError,IndexError,KeyError,TypeError) as exc:
                raise FrontendError(f"ONNX node {self.source}: {exc}") from exc
        returns = []
        for info in self.model.graph.output:
            values = self.values[info.name]
            if any(v.kind != "f64" for v in values.flat):
                raise FrontendError("boolean graph outputs require an explicit numeric Where")
            self.output_info.append({"name":info.name,"shape":list(values.shape),"offset":len(returns),"count":values.size})
            returns.extend(self.ref(v) for v in values.flat)
        if not returns: raise FrontendError("model has no numeric outputs")
        signature = ", ".join(f"{arg}: f64" for arg in self.args)
        types = ", ".join(["f64"]*len(returns))
        mlir = "module {\n  func.func @model("+signature+") -> ("+types+") {\n"
        mlir += "\n".join(self.lines)+"\n    return "+", ".join(returns)+" : "+types+"\n  }\n}\n"
        return mlir, {"inputs":self.input_info,"outputs":self.output_info,
                      "scalar_operations":self.counter,"logic_comparisons":self.logic_count,
                      "numeric_semantics":"float32 model coefficients promoted to CKKS f64",
                      "model_parameters":"public", "layout":"one-scalar-per-ciphertext"}


def import_onnx(path: str | Path, *, max_operations=50000, max_elements=50000):
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    versions = {o.domain:o.version for o in model.opset_import}
    if not 13 <= versions.get("",versions.get("ai.onnx",0)) <= 21:
        raise FrontendError("supported ONNX opsets are 13..21")
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return Importer(model,max_operations,max_elements).run()
