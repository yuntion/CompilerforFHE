# 2026-11-16 开发与验证记录（Ubuntu / Conda）

今日完成环境与 README 基线验证、文献驱动的 FHE 优化 pass、AI 后端离线任务准备器，以及 SEAL CPU CKKS SIMD / 对角线 MatVec 的完整编译执行路径。最新一次完整回归为 **87 项 Python 测试、7 项 CTest 全部通过**；之后新增槽位回绕测试，连同原生 pass 用例进行定向验证，得到 **18 项全部通过、9 次真实加密执行成功**。

第 1—6 节保留当日最初的 README 基线实测，**不代表开发完成后的最新能力和测试数量**；新增开发内容、实际指令、性能/精度对比及后续边界统一整理在[第 7 节：今日开发总结](#7-今日开发总结)。后续增加的 3 组参数化测试未重新纳入一次全量运行，不能将 87 项完整回归和 18 项定向验证相加作为独立用例数。

初轮 README 基线的原生 CTest **3/3 通过**；完整 Python 测试 **34/34 通过，0 失败、0 跳过**。完整测试中的 **11 次真实加密执行全部通过精度验证**。随后按 README 分模块再次执行，15 项 front、8 项 compiler、3 项 backend、8 项 e2e 全部通过，产生另外 11 次成功的加密执行。该基线阶段未修改编译器、后端或测试用例来使测试通过。

- 日期与时区：2026-11-16，Asia/Shanghai（UTC+08:00）。
- 测试命令开始：2026-11-16T15:46:49.066879+08:00；最后一组结束：2026-11-16T15:54:04.348124+08:00。
- 7 组测试命令累计进程耗时：406.246 秒，不含依赖安装、指标收集、源码打包及命令间隔。
- 完整测试的 pytest 原始摘要：`34 passed, 29 warnings in 199.91s (0:03:19)`。pytest 耗时与外层进程耗时的差别来自 Python 启动和退出等开销。
- 日志、XML、环境快照和数据均保存于 [`docs/validation/2026-11-16-conda/`](validation/2026-11-16-conda/)。仓库原有的历史 XML、`docs/VALIDATION.md` 和 `docs/measurements/results.*` 均保留。

## 1. 运行环境与前置状态

| 项目 | 本次实际配置 |
| --- | --- |
| 工作目录 | `/shizy/CompilerforFHE` |
| Conda 环境 | `CompilerforFHE`，`/root/miniconda3/envs/CompilerforFHE` |
| 操作系统 | Ubuntu 24.04.5 LTS，Linux x86-64 |
| CPU | AMD EPYC 9754 128-Core Processor |
| CPU 可用配额 | cgroup `3200000 100000`，相当于 32 核 CPU 时间；宿主逻辑 CPU 数不等于可用配额 |
| Python | 3.12.16 |
| PyTorch | 2.8.0+cpu，CPU wheel，无 CUDA 运行时 |
| ONNX / NumPy / pytest | 1.19.0 / 2.2.6 / 8.4.2 |
| CMake / Ninja Python 包 | 3.31.6 / 1.11.1.4 |
| C++ 编译器 | g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| 原生构建配置 | Release，LLVM assertions 开启；沿用上一轮已构建的工具链 |
| 仓库 HEAD | `8808512e5c8635efd71c27cc7b079c888cf101d4`；工作区包含上一轮 README 调整和已有文件删除，并非干净 checkout |
| 密码库 | SEAL `119dc32e135cb89c1062076a69310d4413ebc824`；OpenFHE `aa391988d354d4360f390f223a90e0d1b98839d7` |
| cereal | `984e3f194862b17916536b5fade40cba6e47a6fe` |

原生工具位于 `build/bin/`，SEAL 配置位于 `build/seal/cmake/`，OpenFHE 配置位于 `.crossfhe/openfhe-install/lib/OpenFHE/`；驱动读取 `.crossfhe/toolchain.json`。本轮未重复执行 bootstrap。

继续遵守上一轮“不安装 front Python 包”的要求：没有执行 `pip install -e .`，环境中没有 `crossfhe-front` distribution；通过 `PYTHONPATH=front` 从仓库加载前端和测试驱动源码。完整测试确实运行了前端代码，并需要安装 PyTorch/ONNX 等测试依赖。

环境详情见 [environment.json](validation/2026-11-16-conda/environment.json)，完整 Python 依赖见 [pip-freeze.txt](validation/2026-11-16-conda/pip-freeze.txt)。

## 2. 执行过程及实际指令

### 2.1 激活环境和补齐测试依赖

所有后续命令均在仓库根目录、`CompilerforFHE` 环境执行。CPU 版 PyTorch 先安装，之后 `requirements.txt` 中的 `torch==2.8.0` 由已安装的 `2.8.0+cpu` 满足。

```bash
cd /shizy/CompilerforFHE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CompilerforFHE

python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/front${PYTHONPATH:+:$PYTHONPATH}"
python -m pip freeze > docs/validation/2026-11-16-conda/pip-freeze.txt
```

安装均成功，输出见 [01-install-torch.log](validation/2026-11-16-conda/01-install-torch.log) 和 [02-install-requirements.log](validation/2026-11-16-conda/02-install-requirements.log)。

### 2.2 运行 README 的测试命令

先运行 CTest、未启用 FHE 的 Python 测试、完整 FHE 测试；完整测试通过后再按模块执行。以下为实际测试参数：在 README 基础上增加 `-ra` 显示跳过原因，并为每组保存 JUnit XML；临时目录改成独立的日期目录，避免覆盖已有 artifact。没有改变测试集合、断言、精度容差或方案参数。

```bash
# 原生 CTest
ctest --test-dir build --output-on-failure --output-junit /shizy/CompilerforFHE/docs/validation/2026-11-16-conda/native-junit.xml

# Python，未启用 FHE
python -m pytest -q -ra --basetemp /shizy/CompilerforFHE/.crossfhe/validation-2026-11-16/test-unit --junitxml /shizy/CompilerforFHE/docs/validation/2026-11-16-conda/unit-junit.xml

# Python，完整 FHE
python -m pytest --run-fhe -q -ra --basetemp /shizy/CompilerforFHE/.crossfhe/validation-2026-11-16/test-all --junitxml /shizy/CompilerforFHE/docs/validation/2026-11-16-conda/full-fhe-junit.xml

# front 模块
python -m pytest tests/front -q -ra --basetemp /shizy/CompilerforFHE/.crossfhe/validation-2026-11-16/test-front --junitxml /shizy/CompilerforFHE/docs/validation/2026-11-16-conda/front-junit.xml

# compiler 模块
python -m pytest tests/compiler -q -ra --basetemp /shizy/CompilerforFHE/.crossfhe/validation-2026-11-16/test-compiler --junitxml /shizy/CompilerforFHE/docs/validation/2026-11-16-conda/compiler-junit.xml

# backend 模块，启用 FHE
python -m pytest tests/backend --run-fhe -q -ra --basetemp /shizy/CompilerforFHE/.crossfhe/validation-2026-11-16/test-backend --junitxml /shizy/CompilerforFHE/docs/validation/2026-11-16-conda/backend-junit.xml

# e2e 模块，启用 FHE
python -m pytest tests/e2e --run-fhe -q -ra --basetemp /shizy/CompilerforFHE/.crossfhe/validation-2026-11-16/test-e2e --junitxml /shizy/CompilerforFHE/docs/validation/2026-11-16-conda/e2e-junit.xml
```

每条命令的工作目录、环境名、开始/结束时间、退出码、进程耗时与日志路径见 [commands.json](validation/2026-11-16-conda/commands.json)。七条测试命令的退出码全部为 `0`。

### 2.3 加密测试内部做了什么

每个正向集成用例依次执行：PyTorch 计算明文参考值 → 导出 ONNX → front 生成 `01_arith.mlir` → 原生工具降级/代码生成 → CMake 编译该模型的 runner → 密钥生成、加密、服务端计算、客户端解密 → 与参考值比较。

其内部命令结构如下，`<artifact>`、`<runner>`、`<client>`、`<server>` 为各用例的独立绝对路径；`cmake` 和 C++ 编译器由 toolchain.json 指定：

```bash
build/bin/crossfhe-opt <artifact>/01_arith.mlir --ppml-to-scfhe --verify-each -o <artifact>/02_scfhe.mlir
build/bin/crossfhe-opt <artifact>/02_scfhe.mlir --ppml-to-heir --verify-each -o <artifact>/03_fhe.mlir
build/bin/crossfhe-opt <artifact>/03_fhe.mlir --ppml-fhe-optimize --verify-each -o <artifact>/04_optimized.mlir
build/bin/crossfhe-translate <artifact>/04_optimized.mlir --backend=<backend> -o <artifact>/model.hpp
cmake -S backend -B <artifact>/build -G Ninja -DCMAKE_BUILD_TYPE=Release ...
cmake --build <artifact>/build --target model-runner --parallel 2
<runner> keygen <client> <server> [--comparison-scale 1024.0 --insecure-demo]
<runner> encrypt <client> <server> <client>/input.bin <server>/input.ct
<runner> evaluate <server> <server>/input.ct <server>/output.ct
<runner> decrypt <client> <server>/output.ct <client>/output.bin
```

方括号内选项仅在本轮 OpenFHE 测试中使用。实际 configure/build 输出和四个密码执行阶段日志保留在各模型 artifact 的 `logs/`、`runs/run-*/logs/` 中；模型 artifact 路径收录在 [cases.json](validation/2026-11-16-conda/measurements/cases.json)。

### 2.4 收集指标并执行源码打包

README 的示例从 `artifacts/relu-toy` 和 `artifacts/affine` 收集；本轮直接从测试实际产物收集，覆盖完整测试的 11 个用例。分模块复跑的 11 个用例单独保存，避免混在同一张性能表中。

```bash
python scripts/collect_reports.py .crossfhe/validation-2026-11-16/test-all --output docs/validation/2026-11-16-conda/measurements
python scripts/collect_reports.py .crossfhe/validation-2026-11-16/test-backend .crossfhe/validation-2026-11-16/test-e2e --output docs/validation/2026-11-16-conda/measurements-modules
python scripts/package_source.py --output CrossFHE-source-2026-11-16.zip
```

两轮各收集 11 条成功记录。源码包已生成；检查未包含 `.crossfhe/`、`artifacts/`、构建目录、密钥或密文文件。归档文件数、大小、检查结果及 SHA-256 见 [package.json](validation/2026-11-16-conda/package.json)。

## 3. 分组测试结果

下表耗时为外层进程墙钟时间；每组的 pytest 自报时间在对应 `.log` 中。

| 执行组 | 通过 | 失败/错误 | 跳过 | 进程耗时（秒） | 原始结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原生 CTest | 3 | 0 | 0 | 0.124 | [native-junit.xml](validation/2026-11-16-conda/native-junit.xml) |
| Python，未启用 FHE | 23 | 0 | 11 | 2.105 | [unit-junit.xml](validation/2026-11-16-conda/unit-junit.xml) |
| Python，完整 FHE | 34 | 0 | 0 | 200.551 | [full-fhe-junit.xml](validation/2026-11-16-conda/full-fhe-junit.xml) |
| front 模块 | 15 | 0 | 0 | 0.406 | [front-junit.xml](validation/2026-11-16-conda/front-junit.xml) |
| compiler 模块 | 8 | 0 | 0 | 1.883 | [compiler-junit.xml](validation/2026-11-16-conda/compiler-junit.xml) |
| backend 模块，启用 FHE | 3 | 0 | 0 | 12.804 | [backend-junit.xml](validation/2026-11-16-conda/backend-junit.xml) |
| e2e 模块，启用 FHE | 8 | 0 | 0 | 188.374 | [e2e-junit.xml](validation/2026-11-16-conda/e2e-junit.xml) |

未启用 `--run-fhe` 时，`tests/conftest.py` 主动跳过 3 个 backend 和 8 个 e2e 用例。这里的 11 次跳过符合设计；完整 FHE 和所有模块测试没有跳过。分模块执行与完整测试存在重复，独立 Python 测试用例仍为 **34 个**，不能把复跑次数当成新增覆盖。

| 模块 | 独立用例数 | 验证内容 |
| --- | ---: | --- |
| 原生 CTest | 3 | 方言注册、MLIR 降级、原生 C++ 代码生成 |
| front | 15 | 不支持的非线性诊断、秘密除法拒绝、静态形状/dtype、比较与选择保留、操作数量限制、公开 shape、NaN/Inf、custom domain、标量 reduction |
| compiler | 8 | 原生方言、SEAL 拒绝逻辑模型、非法 SSA、SEAL/GPU codegen 拒绝且不生成头文件、不完整 artifact、依赖 tensor 的 Python 分支/item tracing 拒绝 |
| backend | 3 | 分组/空洞卷积与 BatchNorm、秘密批量 MatMul 和广播、常量输出及零乘法的真实 SEAL 计算 |
| e2e | 8 | SEAL affine/square-mlp/tiny-cnn/poly-attention；OpenFHE affine/ReLU/ReLU-MLP/MaxPool；服务端无私钥、优化前后 IR 字节相同、跨方案调用次数 |

34 个完整测试用例的名称、状态和 JUnit 耗时见 [test-cases.csv](validation/2026-11-16-conda/test-cases.csv)。完整测试输出 **29 条 warning**：28 条 PyTorch 旧 ONNX 导出 API 的弃用提示，以及 1 条“故意输出常量 tensor”的 TracerWarning；没有导致测试失败。本轮没有遇到需要修复源代码或重跑失败用例的问题。

## 4. 真实加密执行指标

以下表格只使用首次完整 FHE 测试的 11 条记录。它们均通过 `atol=0.001, rtol=0.001` 的比较；NumPy 判断条件为逐元素 `abs(actual-reference) <= atol + rtol*abs(reference)`。最大绝对误差为 **1.764835452e-06**，来自 OpenFHE MaxPool。

| 模型/用例 | 后端参数 | 环维度 | depth | 最大绝对误差 | evaluator 内部秒 | evaluate 进程秒 | 服务端 bundle MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 秘密批量 MatMul | SEAL tc128 | 8192 | 1 | 8.545e-08 | 0.031934 | 0.043500 | 1.13 |
| 常量输出/零乘法 | SEAL tc128 | 8192 | 1 | 1.038e-09 | 0.013852 | 0.023506 | 1.13 |
| 分组空洞卷积/BN | SEAL tc128 | 8192 | 2 | 2.495e-07 | 0.044056 | 0.070769 | 2.00 |
| affine | SEAL tc128 | 8192 | 2 | 1.759e-07 | 0.009343 | 0.022516 | 2.00 |
| square-mlp | SEAL tc128 | 16384 | 5 | 6.718e-07 | 0.116680 | 0.173602 | 12.25 |
| tiny-cnn | SEAL tc128 | 16384 | 5 | 3.967e-07 | 0.358574 | 0.433863 | 12.25 |
| poly-attention | SEAL tc128 | 16384 | 6 | 3.644e-07 | 0.431721 | 0.509003 | 16.00 |
| affine | OpenFHE TOY | 8192 | 2 | 1.304e-08 | 0.004863 | 0.024257 | 2.00 |
| ReLU | OpenFHE TOY | 8192 | 19 | 7.354e-07 | 3.934428 | 4.463025 | 422.83 |
| ReLU-MLP | OpenFHE TOY | 8192 | 22 | 9.671e-08 | 4.803706 | 5.366280 | 440.46 |
| MaxPool | OpenFHE TOY | 8192 | 20 | 1.765e-06 | 6.349068 | 6.932983 | 428.71 |

- `evaluator 内部秒` 来自 runner 输出的 `runtime.seconds`，不含整个进程的启动、密钥加载等开销；`evaluate 进程秒` 是 Python 计时得到的完整评估进程耗时。
- `depth` 来自密钥参数记录；OpenFHE 含跨方案所需的预留深度，不能全部解释为模型的算术乘法深度。
- `bundle` 使用现有收集器的口径：服务端目录下 `.bin`/`.seal` 文件的字节总和，包括公开/评估键相关文件；不是单个 bootstrap key 的大小。MiB = 1,048,576 字节。
- 7 次 SEAL 使用 `tc128`；4 次 OpenFHE 显式使用 `INSECURE_TOY`。所有 11 条报告的 `validation.passed` 和 `server_has_no_secret_key` 都为 `true`。
- 8 项 e2e 还逐项断言优化前后的 IR 字节相同；逻辑模型确认 `heir.depack` 存在，且三类跨方案调用次数与 manifest 一致。

OpenFHE 逻辑路径细分如下，时间为 evaluator 内部累计值：

| 模型 | CKKS→LWE / sign / LWE→CKKS 次数 | CKKS→LWE 秒 | sign 秒 | LWE→CKKS 秒 | keygen 进程秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| ReLU | 2 / 2 / 2 | 0.048195 | 0.346245 | 3.459851 | 2.393738 |
| ReLU-MLP | 2 / 2 / 2 | 0.050048 | 0.348024 | 4.251860 | 2.594299 |
| MaxPool | 3 / 3 / 3 | 0.058270 | 0.522717 | 5.678552 | 2.728597 |

完整测试中，11 个模型的 lowering/codegen/CMake configure/C++ build 阶段累计 **170.756 秒**，evaluate 进程累计 **18.063 秒**。因此这次全套用时主要消耗在为每个模型编译 runner，不能将 pytest 总耗时解释为密文推理延迟。更完整的编译及 keygen/encrypt/evaluate/decrypt 分段时间见 [cases.json](validation/2026-11-16-conda/measurements/cases.json)。

分模块复跑也得到 11/11 成功记录，最大绝对误差为 `3.529643502e-06`。由于重新生成密钥、重新加密，两轮误差和延迟会变化；复跑原始指标见 [measurements-modules/results.csv](validation/2026-11-16-conda/measurements-modules/results.csv)。

## 5. 本轮结论与覆盖边界

本机 Ubuntu + Conda 环境已实际验证 README 所列的原生编译器、前端诊断、编译产物检查和完整加密回归测试。CPU 版 PyTorch、GCC 13.3.0、SEAL/OpenFHE 固定版本组合可以完成模型导出、MLIR 降级、runner 编译和真实密文计算。

本轮未运行默认 `HEStd_128_classic+STD128` 参数的 OpenFHE ReLU。该项在 README 中属于单独测量，其中的 OpenFHE 回归用例使用 TOY 参数；历史 Windows 记录中的约 11.9 GB 评估 bundle 与默认安全 ReLU 耗时不能视为本机结果。也未测试 legacy CPU/GPU 路径、真实 GPU 执行、未支持的网络算子或完整 Transformer。

本轮为功能回归验证，完整测试和分模块复跑均按命令顺序执行，没有进行多次受控采样、CPU 绑定或负载隔离。指标用于解释本次运行，不构成统计性能基准，也不用于与历史 Windows 机器比较性能。

## 6. 结果文件索引

- [测试命令、环境、起止时间、退出码和耗时](validation/2026-11-16-conda/commands.json)
- [环境与依赖版本](validation/2026-11-16-conda/environment.json)、[完整 Python 依赖](validation/2026-11-16-conda/pip-freeze.txt)
- [原生 CTest 日志](validation/2026-11-16-conda/03-ctest.log)、[完整 FHE 测试日志](validation/2026-11-16-conda/05-full-fhe.log)
- [完整 FHE JUnit XML](validation/2026-11-16-conda/full-fhe-junit.xml)、[34 个用例明细 CSV](validation/2026-11-16-conda/test-cases.csv)
- [首次完整测试指标 JSON](validation/2026-11-16-conda/measurements/results.json)、[CSV](validation/2026-11-16-conda/measurements/results.csv)、[分阶段详细指标](validation/2026-11-16-conda/measurements/cases.json)
- [按模块复跑指标 JSON](validation/2026-11-16-conda/measurements-modules/results.json)、[CSV](validation/2026-11-16-conda/measurements-modules/results.csv)
- [指标收集日志](validation/2026-11-16-conda/10-collect-full.log)、[复跑指标收集日志](validation/2026-11-16-conda/11-collect-modules.log)、[源码打包日志](validation/2026-11-16-conda/12-package-source.log)、[归档核验](validation/2026-11-16-conda/package.json)

真实模型、阶段 IR、runner、运行日志及本轮生成的密码材料保留在 `.crossfhe/validation-2026-11-16/`；报告目录只收集测试输出和指标，不复制密钥、密文或明文输入。

## 7. 今日开发总结

### 7.1 工作推进与交付

全部开发、构建和 Python 测试均使用 `CompilerforFHE` Conda 环境，通过 `PYTHONPATH=front` 加载源码，未安装 front Python 包。环境与固定密码库版本延续第 1 节；后续快照再次确认 `front_package_installed=false`，见 [MatVec 环境记录](development/validation/matvec/environment.json)。

| 阶段 | 完成内容 | 交付与证据 |
| --- | --- | --- |
| 环境与基线 | 按 README 完成原生/完整 Python/分模块测试，保存环境、命令、指标与源码包核验 | 本文第 1—6 节 |
| 文献与方案 | 阅读开题报告研究方案及答辩意见；对 253 篇 PDF 建立分类、文本检索与 hash 索引，重点阅读相关章节；审阅 `/shizy/fhe-coder` | [文献记录](development/ARTICLES.md)、[开发计划](development/PLAN.md)；未宣称逐页精读全部论文 |
| 标量 FHE 优化 | 手写简化、CSE、DCE、资源分析、乘法树平衡和有界 ReLU 近似，接入默认流水线与 Python/CLI | [Pass 文档](development/PASSES.md)、[优化实测](development/RESULTS.md) |
| AI 后端准备 | 实现从真实编译 artifact 导出经过 hash 校验的公开任务包，构思检索、编译、独立验证和有限轮修复流程 | [AI 后端方案](development/AI_BACKEND.md)、`scripts/prepare_ai_backend.py` |
| CPU SIMD / MatVec | 保留矩阵结构，新增 diagonal/BSGS lowering、周期槽位布局、SEAL 旋转与密钥、打包加解密和资源统计 | [MatVec 开发计划](development/MATVEC_PLAN.md)、[实现与结果](development/MATVEC.md) |
| 独立边界测试 | 新增 5×5 非二次幂矩阵的首槽、末槽及正负混合输入验证，三种 packing 共 9 次加密执行 | [槽位回绕测试报告](development/validation/matvec-wraparound/REPORT.md) |

### 7.2 新增 FHE pass 与流水线变化

| Pass | 已实现的行为 |
| --- | --- |
| `--ppml-fhe-simplify` | 消除加/减公开零、乘一及冗余 select；乘零转换为重新加密的公开零 |
| `--ppml-fhe-cse` | 共享同类型、同操作数顺序、同语义属性的重复计算及完整方案转换链；不共享随机 materialize / packed_zero |
| `--ppml-fhe-dce` | 反向删除没有输出使用的已知密文原语与常量 |
| `--ppml-fhe-refresh-contract` | 重新计算乘法深度、方案转换次数、原语数量，以及打包旋转与明文加法数量 |
| `--ppml-fhe-balance-mul` | 显式启用时，对单次使用的密文乘法子树重新结合，只有深度降低才改写 |
| `--ppml-scfhe-approx-relu` | 根据显式输入范围 B 与误差预算 E，将 ReLU 替换为二次多项式；要求 `B>0, E>=B/16` |
| `--ppml-fhe-matvec` | 将公开仿射 MatVec 降为 CKKS SIMD 对角线或 BSGS 原语，消除零对角线并复用 baby rotations |

默认 `--ppml-fhe-optimize` 组合简化、CSE、DCE 和资源刷新。乘法树平衡、ReLU 近似与 MatVec packing 均需显式选择；变换作用于 `ppml.fhe_enabled=true` 的目标函数。

驱动增加 `optimization=none/scalar/depth`、ReLU 范围/预算及 `packing=none/diagonal/bsgs/auto`；原生工具可以导出优化前后资源 JSON。manifest 保存实际深度、逻辑比较数、布局、资源和产物 hash。旧 e2e 的“优化前后 IR 字节必须相同”检查已替换为资源和真实加密结果验证，因此第 3—4 节描述的字节相同断言仅适用于最初基线。

已用真实密文确认的标量优化效果：重复密文乘法 **4→2**，重复 ReLU 的三类跨方案调用均 **4→2**，八项连乘深度 **7→3**。二次 ReLU 在声明的 `[-B,B]` 范围内有局部实数误差界 `B/16`；它不包含 CKKS 误差和下游网络误差，也不等于模型准确率保证。具体数值、容差和参数见 [优化结果](development/RESULTS.md)。

### 7.3 CPU CKKS SIMD / 对角线 MatVec

参考 GALA/GAZELLE 的打包矩阵运算、Coyote 的旋转成本论述，以及 LinearTransform-SmallKey 第 5 节的经典 diagonal/BSGS 公式，实现了从 ONNX 到真实 SEAL 密文执行的路径。这里的 SIMD 是密文内槽位并行，由 SEAL CPU 库执行；未新增 AVX 指令后端。

关键代码为 `front/crossfhe/packed.py`、`compiler/Pipelines/FHE/MatVec.cpp`、`compiler/Dialect/HEIR/PackedOps.cpp`、HEIR ODS、FHE codegen 和 `backend/seal/`。高层 `heir.matvec` 被真正改写为旋转、明文乘法、密文累加、偏置加法或公开零原语，随后生成并编译模型专用 C++。

对 m×n 矩阵取 `L=next_power_of_two(max(m,n))`，将矩阵与向量补零到逻辑周期 L。由于 SEAL 的旋转周期实际为物理槽数 `S=N/2`，加密时将向量周期重复 `S/L` 次，公开对角线和偏置同样重复，保证回绕正确。BSGS 在二次幂 baby-step 候选中根据非零对角线的实际旋转数量选择方案；按最终 IR 的旋转 steps 集合生成 Galois keys。当前 `bsgs` 与 `auto` 使用同一旋转数量模型，未做自动计时调优。

16×16 随机稠密矩阵的 A/B 实测如下，均使用相同输入/系数、SEAL tc128、N=8192、4096 物理槽、40-bit scale 和一层乘法预算：

| 路径 | 输入/输出密文数 | 明文乘法 | 旋转次数 | evaluator 时间 ms | 最大绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认标量 | 16 / 16 | 256 | 0 | 224.671 | 1.01e-7 |
| diagonal | 1 / 1 | 16 | 15 | 44.023 | 9.85e-7 |
| BSGS | 1 / 1 | 16 | 6 | 26.679 | 5.27e-7 |
| auto，选择 BSGS | 1 / 1 | 16 | 6 | 27.418 | 8.42e-7 |

输入密文文件 **4,196,120→262,265 B**，输出 **2,098,968→131,193 B**。旋转需要额外评估键：server bundle 从标量的 1,180,180 B 增至 diagonal 的 13,045,642 B，BSGS/auto 为 5,965,720 B。计时为单次 evaluator 内部测量，未包含编译、keygen 和传输，不作为统计性能结论。

另覆盖矩形、非二次幂、1×1、稀疏/全零矩阵、向量左右位置、行/列表示、PyTorch Linear、Gemm alpha/beta 和标量偏置。该阶段新增 16 次加密执行，最大绝对误差不超过 **1.55e-6**；运行时资源与原生计划逐项核对通过，见 [16 个用例明细](development/validation/matvec/cases.json)。

之后新增的独立回绕测试使用 5×5 矩阵、L=8、4096 物理槽；每条 packing 路径用首槽基向量、末槽基向量和正负混合输入重新生成密钥并加密。结果如下，9 次均使用 1 个输入密文、1 个输出密文和 5 次打包明文乘法：

| 路径 | 每次旋转 | 三组最大绝对误差 | 验证阈值 |
| --- | ---: | ---: | ---: |
| diagonal | 5 | 1.132e-6 | 绝对误差 ≤1e-5 |
| BSGS | 3 | 1.772e-6 | 绝对误差 ≤1e-5 |
| auto | 3 | 1.497e-6 | 绝对误差 ≤1e-5 |

解密值、NumPy 参考及各阶段指标保存在 [回绕结果 JSON](development/validation/matvec-wraparound/results.json)。这 9 次执行是后续单独验证，不混入前述 16 次的最大误差统计。

### 7.4 AI 后端的实际进度

参考 `/shizy/fhe-coder` 的 agent/RAG/反馈结构，确定 AI 的范围是已确定 FHE 算法之后的库适配和调度候选。已实现 `scripts/prepare_ai_backend.py`：检查真实 artifact 的 ready 状态、公开文件 SHA-256 与资源契约一致性，再按白名单导出 typed IR、参考 evaluator、runtime API 和任务约束。

真实 artifact 已成功导出任务，状态为 **`prepared_unvalidated`**，`validation_performed=false`；4 项任务准备测试覆盖公开导出、hash 篡改拒绝、artifact 覆盖拒绝和非空目录保护。没有将准备成功描述为 AI 生成成功。

在线生成路线已写明：固定版本 API 检索 → 结构化候选 → 隔离真实编译 → 独立客户端解密差分 → 最多 10 轮反馈修复 → 固定参数测量 → 通过检查后才可选择候选。在线模型调用、候选执行、自动修复和晋升尚未实现，详见 [AI 后端方案](development/AI_BACKEND.md) 与 [任务准备日志](development/validation/ai-prepare.log)。

### 7.5 后续开发阶段的主要执行命令

以下为当日构建、完整回归、后续定向验证及真实 AI 任务准备的命令。每个阶段的临时目录与日志独立保存；最初 README 的指令仍见第 2 节。

```bash
cd /shizy/CompilerforFHE
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CompilerforFHE
export PYTHONPATH="$PWD/front"

# 新增 pass 后构建原生工具
cmake --build build --target crossfhe-opt crossfhe-translate --parallel 8

# 标量优化阶段的完整回归：59 项
python -m pytest --run-fhe -q -ra \
  --basetemp .crossfhe/test-optimized-all \
  --junitxml docs/development/validation/full-junit.xml

# MatVec 阶段原生验证：7 项
ctest --test-dir build --output-on-failure \
  --output-junit "$PWD/docs/development/validation/matvec/native-junit.xml"

# MatVec 阶段完整回归：87 项
python -m pytest --run-fhe -q -ra \
  --basetemp .crossfhe/test-matvec-all \
  --junitxml docs/development/validation/matvec/full-junit.xml

# 后续新增回绕测试 + 原生 MatVec 验证：18 项，9 次真实加密
python -m pytest --run-fhe -q -s \
  tests/compiler/test_matvec_pass.py \
  tests/e2e/test_packed_matvec_execution.py::test_periodic_slot_wraparound_with_repeated_encryption \
  --basetemp .crossfhe/test-matvec-wraparound \
  --junitxml docs/development/validation/matvec-wraparound/junit.xml

# 验证 CLI 的 packed affine 执行链路
python -m crossfhe demo affine --backend seal-ckks --packing auto \
  --output .crossfhe/matvec-cli

# 从经过真实编译/验证的 artifact 准备公开 AI 后端任务
python scripts/prepare_ai_backend.py \
  .crossfhe/test-optimized-all/test_approximate_relu_runs_on_0/relu \
  --output .crossfhe/ai-backend-verified-input
```

打包路径的阶段文件为 `01_matvec.mlir → 03_fhe.mlir → 04_optimized.mlir → model.hpp`，随后仍由 CMake 构建 runner 并执行 keygen/encrypt/evaluate/decrypt。自己的单仿射 ONNX 可使用 `python -m crossfhe compile model.onnx --backend seal-ckks --packing bsgs --output artifacts/my-packed-model`。

### 7.6 分阶段验证汇总与开发中修复

| 验证阶段 | Python 结果 | CTest 结果 | 加密执行覆盖 | 日志 |
| --- | --- | --- | --- | --- |
| 初轮 README 完整基线 | 34 passed，199.91s | 3/3 | 11 次；另有按模块复跑 11 次 | [基线日志](validation/2026-11-16-conda/05-full-fhe.log) |
| 标量优化完整回归 | 59 passed，35 warnings，308.10s | 最终 6/6，0.02s | 18 次，包含原有 11 次 | [优化日志](development/validation/full.log)、[CTest](development/validation/native.log) |
| MatVec 完整回归 | 87 passed，37 warnings，372.26s | 7/7，0.03s | 在原有执行基础上新增 16 次 MatVec A/B 和边界运行 | [完整日志](development/validation/matvec/full.log)、[JUnit](development/validation/matvec/full-junit.xml)、[CTest](development/validation/matvec/native.log) |
| 后续回绕定向验证 | 18 passed，12.51s | 未在该轮重复执行 | 3 项参数化测试共 9 次 SEAL 加密；另 15 项为已有原生用例 | [定向日志](development/validation/matvec-wraparound/pytest.log)、[JUnit](development/validation/matvec-wraparound/junit.xml) |

以上完整/定向验证均无失败、无跳过；34、59、87 表示各阶段逐步扩展的测试集合，不能相加。18 项定向验证含 15 项重复原生用例，新增部分为 3 组 packing 参数；其加入后尚未重跑一次包含全部用例的完整回归。本次文档更新核对了已有日志和指标，没有重新执行上述测试。

开发中实际修复了三个验证问题：PyTorch 导出会提前合并重复表达式，A/B 测试改为显式重复节点的 ONNX；新增 compiler/e2e 测试同名导致收集冲突，调整了文件名；MatVec codegen 初版布局检查误拒绝旧标量布局，补齐兼容并由 87 项完整回归验证。未以放宽既有精度容差来消除这些失败。

### 7.7 当前能力边界与下一步

SEAL CPU packing 当前支持**单个秘密向量、公开二维矩阵与偏置的单输出仿射变换**。支持的 MatMul/Gemm 形状与属性、周期/容量和 verifier 拒绝规则见 [MatVec 报告](development/MATVEC.md)。多个 MatVec 的网络布局、多个秘密向量、秘密矩阵、跨密文分块及 OpenFHE packed 路径尚未实现；默认仍走既有标量编译路径。

SEAL 新增测试全部使用 tc128；原有 OpenFHE 回归仍为显式 TOY 功能配置。噪声分析、全网络误差证明、数据集准确率、通用 GPU 执行、hoisting、parallel BSGS 和在线 AI 生成不计入当日已实现能力。下一步可优先扩展连续 MatVec/激活的布局传播与能力检查，并在固定参数下做多次性能采样；AI 候选验证执行器按独立路线推进。
