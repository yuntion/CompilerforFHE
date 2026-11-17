# Frontend and CLI

`front/crossfhe/` 提供 Python 包 `crossfhe`，负责 PyTorch/ONNX 模型接入、统一 CLI 和编译流程编排。原生方言、lowering 与代码生成位于 `compiler/`；密文布局、工具链、runner 构建和加密执行位于 `backend/`。整个项目的架构与构建见 [根 README](../README.md)。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `crossfhe/frontend.py` | 读取 ONNX 模型，校验形状/类型，展开静态 tensor 运算，输出 `func/arith` MLIR 和 tensor mapping |
| `crossfhe/affine.py` | 解析单个公开仿射 MatVec 的矩阵、偏置、输入输出形状与名称 |
| `crossfhe/exporter.py` | PyTorch CPU/eval 推理、静态 ONNX 导出及明文参考结果 |
| `crossfhe/demos.py` | 仿射、平方 MLP、CNN、多项式 attention 及逻辑激活示例 |
| `crossfhe/cli.py`、`__main__.py` | `python -m crossfhe` / `crossfhe` 命令入口 |
| `crossfhe/pipeline.py` | 编排模型导入、原生编译器、后端 runner 构建，保存 IR、资源契约、hash、日志与 manifest |

模型导入与仿射解析保留模型语义和公开矩阵结构。CKKS 周期布局与 HEIR MatVec 由 `backend/packed.py` 生成，标量编译产物的密文布局注解由编译流程补充。

## 安装与统一命令

在仓库根目录、`CompilerforFHE` 环境中执行 `python -m pip install --no-deps -e .`。直接使用源码时，设置：

```bash
export PYTHONPATH="$PWD:$PWD/front${PYTHONPATH:+:$PYTHONPATH}"
```

随后可切换工作目录。统一命令支持导出、编译、示例和加密执行：

```bash
python -m crossfhe export --factory examples.user_model:create_model --inputs inputs.npz --output model.onnx
python -m crossfhe compile model.onnx --backend openfhe-cross --output artifacts/model
python -m crossfhe run artifacts/model --inputs model.inputs.npz --reference model.reference.npz --insecure-demo
```

NPZ key 必须对应 ONNX tensor 名，输入形状与编译时一致。`export` 会生成 `.inputs.npz` 和 `.reference.npz`，其 factory 模块必须是可信 Python 代码。示例使用显式 TOY 参数做功能调试；默认安全参数运行时去掉 `--insecure-demo`，资源要求见 [根 README](../README.md)。

编译配置在仓库根 `.crossfhe/toolchain.json`，可用 `CROSSFHE_CONFIG` 环境变量或 CLI 全局 `--config` 覆盖。配置读取与子进程调用由 `backend.toolchain` 实现，runner 构建与加密执行由 `backend.execution` 实现。

## 导出与导入 API

```python
import torch
from pathlib import Path
from crossfhe.exporter import export_torch
from crossfhe.frontend import import_onnx
from examples.user_model import create_model

x = torch.tensor([[-0.5, 0.25, 0.75]], dtype=torch.float32)
inputs, reference = export_torch(create_model(), (x,), "artifacts/model.onnx")
mlir, mapping = import_onnx("artifacts/model.onnx")
Path("artifacts/01_arith.mlir").write_text(mlir, encoding="utf-8")
```

`export_torch` 使用 CPU/eval 推理模式、opset 17 和静态输入形状。输出是一个 tensor 或 tensor tuple。若模型将秘密 tensor 转成 Python 值，或使用依赖 tensor 数据的 Python `if`，导出器会拒绝相应 tracing 警告，避免固化运行时逻辑。使用 `torch.where` 等 tensor 操作表达选择。

`import_onnx` 接受 opset 13..21。运行时浮点输入标为秘密，initializer、形状和索引常量公开；数值运算提升到 f64。错误包含 ONNX node 的序号、名称与算子。完整支持表见 [OPERATORS.md](../docs/OPERATORS.md)。

`crossfhe.affine.import_affine_onnx` 返回公开矩阵、偏置和模型输入输出 mapping，支持单个秘密向量、公开 MatMul/Gemm 与偏置。模型不满足该结构时明确拒绝。通用算子使用 `import_onnx` 的标量展开路径。

## 编译 API 与选项

`crossfhe.pipeline` 提供 `compile_torch`、`compile_onnx`、`export_torch` 和 `run_encrypted`，并保留既有工具函数导入。后端工具代码直接导入 `backend.toolchain`、`backend.execution` 或 `backend.packed`。

编译 API 支持 `optimization="none"/"scalar"/"depth"` 和显式 `relu_bound`/`relu_max_error` 参数；实际前后资源与近似契约写入 manifest，详见 [Pass 文档](../docs/development/PASSES.md)。

公开仿射 MatVec 可传入 `backend="seal-ckks", packing="auto"`，也可选 `diagonal` / `bsgs`。前端保留公开矩阵结构，后端选择 CKKS 周期布局，原生 pass 再将 MatVec 降为旋转、明文乘法与累加：

```bash
python -m crossfhe demo affine --backend seal-ckks --packing auto --output artifacts/affine-packed
```

可选 `backend="phantom-ckks"` 在 GPU 上运行同一 MatVec 语义，需先配置 CUDA 依赖，见 [Phantom 后端](../docs/PHANTOM_BACKEND.md)。详细布局和验证见 [MatVec 报告](../docs/development/MATVEC.md)。

AI 命令使用 `python -m crossfhe ai prepare|baseline|run`，任务协议、模型配置与验证边界见 [AI 后端说明](../docs/development/AI_BACKEND.md)。测试按 `tests/compiler`、`tests/e2e` 和 `tests/backend/ai` 分层，运行方式见 [测试说明](../tests/README.md)。

## 前端测试

在仓库根目录执行，不需要构建密码库或原生编译器：

```bash
python scripts/run_tests.py --suite front
```

覆盖不支持的非线性、秘密除法、动态/空形状、dtype、公开形状计算、scalar limit 和非有限常量等。也可直接执行 `python -m pytest tests/front -q`，仓库 pytest 配置会加载源码，无需手动设置 PYTHONPATH 或安装 PyTorch。使用 PyTorch 导出和参考计算的测试另行声明 `torch` 依赖。原生诊断与完整推理分别在 `tests/compiler` 和 `tests/e2e`。详见 [测试说明](../tests/README.md)。
