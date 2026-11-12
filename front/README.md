# Frontend

`front/` 是 CrossFHE 的模型接入层。它把 PyTorch/ONNX 程序转换为编译器可接收的 `func/arith` MLIR，不定义 FHE 方言、不实现原生 lowering 或密码运算。

整个项目的架构、构建、测试和端到端用法见 [根 README](../README.md)。本文件只说明前端接口。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `crossfhe/frontend.py` | 用 ONNX 结构化 API 读取模型，校验形状/类型，展开静态 tensor 运算，输出 MLIR 和 tensor mapping |
| `crossfhe/pipeline.py` | PyTorch 导出及驱动编译器/后端工具，保留编译产物、hash、日志和验证报告 |
| `crossfhe/cli.py` | `export`、`compile`、`demo`、`run` 的 Python 命令入口 |
| `crossfhe/demos.py` | 仿射、平方 MLP、CNN、多项式 attention 及逻辑激活示例 |

Python 驱动只编排工具调用。SCFHE/HEIR 降级、方言 verifier 和 C++ 生成均在 `compiler/`，库运算及密钥序列化在 `backend/`。

## 导出与导入 API

在仓库根目录安装 `python -m pip install --no-deps -e .` 后：

```python
import torch
from pathlib import Path
from crossfhe.pipeline import export_torch
from crossfhe.frontend import import_onnx
from examples.user_model import create_model

x = torch.tensor([[-0.5, 0.25, 0.75]], dtype=torch.float32)
inputs, reference = export_torch(create_model(), (x,), "artifacts/model.onnx")
mlir, mapping = import_onnx("artifacts/model.onnx")
Path("artifacts/01_arith.mlir").write_text(mlir, encoding="utf-8")
```

`export_torch` 使用 CPU/eval 推理模式、opset 17 和静态输入形状。输出是一个 tensor 或 tensor tuple。若模型将秘密 tensor 转成 Python 值，或使用依赖 tensor 数据的 Python `if`，导出器会拒绝相应 tracing 警告，避免固化运行时逻辑。使用 `torch.where` 等 tensor 操作表达选择。

`import_onnx` 接受 opset 13..21。运行时浮点输入标为秘密，initializer、形状和索引常量公开；数值运算提升到 f64。错误包含 ONNX node 的序号、名称与算子。完整支持表见 [OPERATORS.md](../docs/OPERATORS.md)。

## 驱动整个编译器

```powershell
python -m crossfhe export --factory examples.user_model:create_model --inputs inputs.npz --output model.onnx
python -m crossfhe compile model.onnx --backend openfhe-cross --output artifacts\model
python -m crossfhe run artifacts\model --inputs inputs.npz --reference reference.npz
```

NPZ key 必须对应 ONNX tensor 名，输入形状与编译时一致。`export` 会生成 `.inputs.npz` 和 `.reference.npz`，其 factory 模块必须是可信 Python 代码。

编译配置在仓库根 `.crossfhe/toolchain.json`，可用 `CROSSFHE_CONFIG` 环境变量或 CLI 全局 `--config` 覆盖。`compile_torch`、`compile_onnx`、`run_encrypted` 保持原有 Python API。

## 前端测试

在仓库根目录执行，不需要构建密码库或原生编译器：

```powershell
python -m pytest tests\front -q --basetemp .crossfhe\test-front
```

覆盖不支持的非线性、秘密除法、动态/空形状、dtype、公开形状计算、scalar limit 和非有限常量等。原生诊断与完整推理分别在 `tests/compiler` 和 `tests/e2e`。
