# ONNX support

The frontend validates ONNX opsets 13..21 and positive static floating-point input shapes. PyTorch export uses opset 17. Numeric tensors are promoted to f64; model initializers and shape tensors are public. Unsupported operations fail with the ONNX node's index, name and operator.

| Category | Supported operations | Constraints |
| --- | --- | --- |
| Arithmetic | Add, Sub, Mul, Neg, Div, Pow | Div denominator public/nonzero; Pow public integer 0..8 |
| Dense | MatMul, Gemm | Static shapes, ONNX batch broadcasting, Gemm transA/transB/alpha/beta |
| Convolution | Conv | 2D NCHW, explicit pads, groups, strides, dilation |
| Pooling | AveragePool, GlobalAveragePool, MaxPool | NCHW, floor mode, explicit pads; no MaxPool indices/dilation |
| Reductions | ReduceSum, ReduceMean | Public axes; keepdims and scalar outputs supported |
| Shapes | Shape, Reshape, Flatten, Transpose, Squeeze, Unsqueeze, Concat, Gather, Identity | Shape/index operands public; no dynamic tensor dimensions |
| Normalization | BatchNormalization | Inference mode, public affine and running-stat parameters |
| Comparison | Less, Greater | Quantized sign contract, OpenFHE cross-scheme backend |
| Selection | Where, Relu, Min, Max | Numeric branches; arithmetic mask selection |
| Constants | Constant, graph initializers | Finite values when materialized; tensor-valued Constant |
| Cast | FLOAT/DOUBLE casts | Numeric promotion only, no secret integer/Boolean quantization |

Rejected examples include Softmax, Sigmoid, Tanh, Exp, Sqrt, GELU, LayerNormalization, encrypted Div, secret Gather indices, tensor-dependent loops, control flow and custom ONNX domains.

The default limit is 50000 scalar operations and 50000 elements per tensor. This guard is necessary because the baseline expands tensor operations without SIMD packing. Raising it does not make a full BERT graph practical.
