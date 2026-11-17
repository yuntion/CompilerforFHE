# FHE Backends

`backend/` 包含 CKKS 布局生成、工具链与执行工具、编译器生成代码调用的密码库适配器、模型专用 runner，以及 AI 候选生成和验证驱动。MLIR 方言、方案降级和 C++ 代码生成在 `compiler/`；模型接入、统一 CLI 与全链路编排在 `front/crossfhe/`。

## Python 后端模块

| 模块 | 职责 |
| --- | --- |
| [toolchain.py](toolchain.py) | 工具链配置读取、运行环境、原生工具定位、子进程超时与日志 |
| [execution.py](execution.py) | 后端源码与库指纹、模型 runner 构建、四阶段加密执行和验证报告 |
| [packed.py](packed.py) | 读取前端解析的公开仿射结构，选择 CKKS 周期布局并生成 HEIR MatVec |
| [ai/](ai/README.md) | 任务、提示词、候选生成、调度与验证闭环 |
| [api/](api/README.md) | 本地函数映射和版本约束资产 |

AI 实现直接复用 `backend.toolchain` 和 `backend.execution`。[前端编译流程](../front/README.md) 调用这些模块完成 runner 构建和加密执行。

## 库适配

| 目录 | 后端名称 | 库与能力 |
| --- | --- | --- |
| `seal/` | `seal-ckks` | Microsoft SEAL 4.1.2，CKKS 算术及必要的 rescale、level alignment、relinearization |
| `openfhe/` | `openfhe-cross` | OpenFHE 1.4.2，CKKS 算术、CKKS/FHEW 转换、sign bootstrapping、掩码选择 |
| `phantom/` | `phantom-ckks` | 可选 PhantomFHE 2.1 CUDA CKKS 算术与周期布局 MatVec，实验性后端 |

每个目录包含 runtime header、runner 源文件（CPU 为 `.cpp`，Phantom 为 `.cu`）和 `CMakeLists.txt`。密码库完整源码属于外部依赖，由根 `scripts/bootstrap.py` 按 revision 获取到 `third_party/`，或者通过参数引用已有源码。

各库的密文类型不同。跨方案链路完整地在 OpenFHE 中执行；本版不将 SEAL 或 Phantom 密文直接传给 FHEW/TFHEpp。新增 TFHEpp 后端须实现自己的转换键和数值编码。

## 函数映射入口

密码库调用链在本项目中可直接查看：[CodeGen.cpp](../compiler/CodeGen/FHE/CodeGen.cpp) 将 FHE IR 转为 runtime 调用，各后端 runtime 再调用相应密码库。[api/README.md](api/README.md) 说明映射格式；[seal.json](api/seal.json)、[openfhe.json](api/openfhe.json)、[phantom.json](api/phantom.json) 分别列出 IR、runtime 签名、实际 `library_calls`、约束和固定库 revision。修改接口时，应同时检查代码生成器、runtime 与这份映射。

AI 任务默认读取本地映射，并在 `api_context.json.function_mapping` 中封存目标后端的完整内容。无需复制完整上游文档，也无需安装或引用 `/shizy/fhe-coder`。`ai prepare --documentation-root /path/to/fhe-coder` 可选追加 SEAL/OpenFHE 辅助摘要；Phantom 使用自己的 API 映射，HEonGPU 文档不能替代它。独立 TFHE 后端尚未实现。使用命令见 [AI 后端说明](../docs/development/AI_BACKEND.md)。

## AI 候选后端

[ai/](ai/README.md) 是 Python 包 `backend.ai`。它负责公开任务封存、函数映射上下文、提示词、模型 provider、严格候选协议、调度编排和独立加密差分。候选只提出固定 DAG 的排列，实际 C++ 仍由原生编译器生成。

| 模块 | 职责 |
| --- | --- |
| [task.py](ai/task.py)、[context.py](ai/context.py) | 准备任务并读取 `backend/api/` 的本地函数映射 |
| [protocol.py](ai/protocol.py)、[prompts.py](ai/prompts.py) | 校验任务/候选与构造冻结契约下的模型消息 |
| [provider.py](ai/provider.py)、[driver.py](ai/driver.py) | 离线回放或显式模型调用、有限轮候选生成和真实构建编排 |
| [validation.py](ai/validation.py) | 可信客户端与候选求值进程分离、同密文差分和重复计时 |

`front/crossfhe` 为 AI 功能提供 CLI 接入，命令仍为 `python -m crossfhe ai prepare|baseline|run`。`backend/api/` 保存映射资产，`backend/ai/` 保存生成与验证实现；SEAL/OpenFHE/Phantom 目录继续负责实际密码库运算。

## 执行接口

生成代码调用 `constant`、`add`、`sub`、`mul`；跨方案代码另外调用 `to_logic`、`sign`、`to_ckks_mask`、`select`。当前返回 CKKS 的是比较掩码，不是任意实数 LUT 的无损回转。

runner 提供四个独立进程步骤：

```text
model-runner keygen  CLIENT_DIR SERVER_DIR [参数]
model-runner encrypt CLIENT_DIR SERVER_DIR PLAINTEXT_INPUT CIPHERTEXT_INPUT
model-runner evaluate SERVER_DIR CIPHERTEXT_INPUT CIPHERTEXT_OUTPUT
model-runner decrypt CLIENT_DIR CIPHERTEXT_OUTPUT PLAINTEXT_OUTPUT
```

服务端 evaluator 类不持有 secret key 或 decrypt API。该设计是本地客户端/服务端流程分离，并非网络服务或操作系统隔离。模型权重、形状和计算图公开。

## 构建方式

作为根项目子目录时，构建 SEAL 依赖。作为独立 CMake 源目录时，传入生成的 `model.hpp` 和已安装库配置，构建模型 runner：

```text
cmake -S backend -B model-build -G Ninja -DCROSSFHE_BACKEND=seal-ckks -DCROSSFHE_MODEL_HEADER=/absolute/path/model.hpp -DSEAL_DIR=/absolute/path/seal/cmake
cmake --build model-build --target model-runner --parallel 2
```

OpenFHE 使用 `-DCROSSFHE_BACKEND=openfhe-cross` 和 `-DOpenFHE_DIR=...`。通常由 Python 驱动自动指定兼容的工具链和库目录。

## 测试

先在根目录运行 bootstrap，然后：

```bash
python scripts/run_tests.py --suite cpu
python scripts/run_tests.py --suite gpu
```

CPU 套件更新 SEAL 和 OpenFHE（含安装），验证秘密矩阵乘、分组/空洞卷积、公开输出与零乘法，以及 OpenFHE 算术、ReLU、ReLU-MLP、MaxPool 和转换调用计数。GPU 套件更新 Phantom，要求已完成 `--enable-cuda` 配置。底层 pytest 可通过 `--run-fhe -m seal`、`--run-fhe -m openfhe` 或 `--run-fhe --run-gpu -m gpu` 分别选择后端；底层命令需要先自行构建。默认安全配置须另外运行 demo 测量；回归测试的 OpenFHE 使用显式 `INSECURE_TOY`。详见 [测试说明](../tests/README.md)。

数值范围、精度、安全参数、评估键和转换延迟的完整契约见 [BACKENDS.md](../docs/BACKENDS.md)。Phantom 的依赖、四阶段执行、数值约定与 GPU 测试见 [Phantom 后端](../docs/PHANTOM_BACKEND.md)。
