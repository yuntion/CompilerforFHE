# CrossFHE Compiler

CrossFHE 是基于 MLIR 的同态加密编译器。项目主干是 `compiler/`：它定义 FHE 方言，将公开算术程序降级为加密计算，并生成调用 FHE 库的 C++ evaluator。`front/` 负责 PyTorch/ONNX 模型接入、统一 CLI 和编译流程编排，`backend/` 负责密文布局、密码库适配、加密执行及 AI 候选生成与验证。

当前研究方向是 **CKKS 算术 + Logic FHE 非线性计算**。本版采用 Libra 的 SCFHE 方言及 HEIR/CHLOE 的 RLWE/LWE 方言，接入 Microsoft SEAL 和 OpenFHE。已加入手写的 FHE 简化、公共子表达式消除、无用运算裁剪和资源分析，并支持显式开启乘法平衡与有界 ReLU 近似；文献依据、开发计划和使用方式见 [开发计划](docs/development/PLAN.md) 与 [Pass 文档](docs/development/PASSES.md)。

已增加 **SEAL CPU CKKS SIMD / 对角线 MatVec**：`--ppml-fhe-matvec` 将公开仿射矩阵向量乘法降为打包旋转、明文乘法与累加，可选择 diagonal、BSGS 或自动旋转数规划。研究依据、周期布局、使用命令和实测对比见 [MatVec 报告](docs/development/MATVEC.md)。

已接入可选的实验性 **PhantomFHE CUDA CKKS 后端**（`phantom-ckks`），复用现有标量算术和 diagonal / BSGS / auto MatVec，提供独立客户端加密、服务端求值与客户端解密。构建、能力边界和验证方式见 [Phantom 后端](docs/PHANTOM_BACKEND.md)，早期实验见 [接入评估](docs/PHANTOM_FHE_ASSESSMENT.md)。

## Quickstart

以下适用于 **Ubuntu 24.04 + Bash** 的首次 CPU 安装。先安装 Conda 并克隆本仓库，然后在仓库根目录执行；依赖子模块由 bootstrap 自动初始化，无需 CUDA。

```bash
# 1. 准备系统和 Python 环境
sudo apt-get update
sudo apt-get install -y build-essential clang git pkg-config ca-certificates
conda create -n CompilerforFHE python=3.12 pip -y
conda activate CompilerforFHE

# 2. 安装 Python 依赖与构建工具
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# 3. 构建编译器、SEAL 和 OpenFHE
python scripts/bootstrap.py --jobs 4

# 4. 编译仿射模型，执行加密、密文求值和解密，并核对参考输出
python -m crossfhe demo affine --backend seal-ckks --output artifacts/quickstart

# 5. 运行日常快速检查
python scripts/run_tests.py
```

demo 成功时终端报告中的 `validation.passed` 为 `true`，编译产物与执行报告位于 `artifacts/quickstart/`，最近一次报告为 `latest_report.json`。快速检查通过时终端显示 `PASS`。

首次构建需要访问 GitHub 和 Python 包源；内存不足时将 `--jobs 4` 改为 `--jobs 2`。已有环境可从激活 Conda 后直接运行 demo 和测试。自定义工具链或仅构建编译器见 [构建整个项目](#构建整个项目)，GPU 路线见 [Phantom 后端](docs/PHANTOM_BACKEND.md)，完整 CPU/GPU 回归见 [测试与结果](#测试与结果)。

## 编译与执行链路

```text
PyTorch nn.Module / ONNX
        |
        v  front/: 静态形状检查、算子展开、公开参数与秘密输入标注
func / arith MLIR
        |
        v  compiler/Conversion/FHE/LowerToSCFHE.cpp
Libra SCFHE: 算术、比较、选择
        |
        v  compiler/Conversion/FHE/LowerToHEIR.cpp
HEIR/CHLOE: RLWE(CKKS) + LWE(Logic FHE) + 显式方案转换
        |
        v  compiler/Pipelines/FHE/: FHE 简化、CSE、DCE、资源契约刷新
FHE MLIR
        |
        v  compiler/CodeGen/FHE/: 合法性检查、模型专用 C++ 生成
model.hpp
        |
        v  backend/: SEAL CKKS / Phantom CUDA CKKS / OpenFHE CKKS <-> FHEW
客户端加密 -> 服务端密文计算 -> 客户端解密与参考结果比较
```

编译器可以直接接收 MLIR，不依赖 PyTorch 才能运行。PyTorch/ONNX 只是当前 PPML 应用的前端。

## 目录与模块职责

```text
Compile_via_MLIR/
├── compiler/
│   ├── Dialect/HEIR/       HEIR/CHLOE 的 RLWE、LWE、转换操作和 ODS
│   ├── Dialect/SCFHE/      Libra 的 SCFHE 操作、类型和注册实现
│   ├── Conversion/FHE/     arith -> SCFHE -> HEIR 的降级 pass
│   ├── Pipelines/FHE/      pass 声明、注册与 FHE 专项优化
│   └── CodeGen/FHE/        FHE MLIR -> C++ 的可复用代码生成库
├── front/crossfhe/         PyTorch/ONNX 模型接入、统一 CLI 与编译流程编排
├── backend/
│   ├── toolchain.py        工具链配置、原生命令及子进程日志
│   ├── execution.py        后端指纹、runner 构建与四阶段加密执行
│   ├── packed.py           CKKS 周期布局与 HEIR MatVec 生成
│   ├── ai/                 backend.ai：任务、提示词、provider、调度及验证驱动
│   ├── api/                IR → runtime → 密码库函数映射、约束与 revision
│   ├── seal/               SEAL CKKS 运行时和客户端/服务端 runner
│   ├── openfhe/            CKKS/FHEW 转换、逻辑自举和 runner
│   └── phantom/            可选 CUDA CKKS 运行时和 runner
├── tools/                 crossfhe-opt、crossfhe-translate 命令入口
├── tests/                 front、compiler、backend、e2e，含 backend/ai 测试
├── examples/              可接入的 PyTorch 示例模型
├── scripts/               固定依赖构建、结果收集、源码打包
├── build-tools/cmake/      LLVM/MLIR 配置及保留的旧构建入口
├── third_party/            LLVM、SEAL、OpenFHE、PhantomFHE 四个 Git 子模块
├── docs/                   方言来源、数值契约、扩展设计与验证记录
├── CMakeLists.txt          默认构建 FHE 编译器的根入口
├── pyproject.toml          Python 包、后端资源与测试配置
└── requirements.txt        已验证的 Python 工具版本
```

| 模块 | 主要接口/产物 | 如何验证 |
| --- | --- | --- |
| `front` | `export_torch`、`import_onnx` 输出 MLIR 和 tensor mapping；`python -m crossfhe`、`compile_onnx`、`compile_torch` 编排编译与执行 | 前端测试、编译器接口和端到端测试 |
| `compiler/Dialect` | `!scfhe.scfhecipher`、`!heir.rlwecipher`、`!heir.lweciphervec` | CTest 方言注册测试；`crossfhe-opt --show-dialects` |
| `compiler/Conversion` | `--ppml-to-scfhe`、`--ppml-to-heir` | `python scripts/run_tests.py --suite compiler` |
| `compiler/Pipelines` | `--ppml-fhe-optimize` 和独立专项 pass | 优化前后资源分析、原生正反例与真实加密 A/B 测试 |
| `compiler/CodeGen` | `generateCpp`，输出 `model.hpp` | CTest 原生代码生成测试及端到端 C++ 编译 |
| `backend` | CKKS 布局、runner 构建/执行、AI 候选验证及 `constant/add/sub/mul/to_logic/sign/to_ckks_mask/select` 密码库接口 | `python scripts/run_tests.py --suite cpu` 或 `--suite gpu`；AI 纯 Python 测试 |
| `tools` | `crossfhe-opt`、`crossfhe-translate` | `ctest --test-dir build --output-on-failure` |
| 全链路 | PyTorch 参考输出与客户端解密输出比较 | `python scripts/run_tests.py --suite all` |

旧仓库的 TOSA/CPU/GPU 文件仍保留在原目录，只有显式设置 `CROSSFHE_ENABLE_LEGACY_COMPILER=ON` 才进入旧构建。默认 FHE 路径不构建这些目标；本版未验证旧路径。Phantom 后端生成调用 CUDA 密码库的 C++，复用 FHE IR。

### 编译器命令入口

| 工具 | 输入 -> 输出 | 当前职责 |
| --- | --- | --- |
| `crossfhe-opt` | MLIR -> MLIR | 注册 FHE 方言，运行 lowering、分析和优化 pass |
| `crossfhe-translate` | 最终 FHE MLIR -> C++ | 检查后端能力，调用 `compiler/CodeGen/FHE` 生成 `model.hpp` |

两个工具分别完成 IR 处理和 C++ 生成。旧 `nisl-compile` 仅用于可选 legacy 路径，行为和实现见 [命令入口说明](tools/README.md)。

## 构建整个项目

以下使用 **Ubuntu 24.04 LTS、Bash、Conda、Python 3.12**。除安装系统依赖外，所有命令都在包含 `compiler/`、`front/`、`backend/` 的仓库根目录执行，并在 `CompilerforFHE` Conda 环境中构建和验证。

### 1. 准备系统环境

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential clang git pkg-config ca-certificates

conda create -n CompilerforFHE python=3.12 pip
conda activate CompilerforFHE
```

`requirements.txt` 记录 pip 包名和版本；先用 Conda 创建 Python 环境，再按下文通过 `python -m pip install -r requirements.txt` 安装依赖。

每次打开新终端都先执行 `conda activate CompilerforFHE`。后续的 `python`、`cmake`、`ctest` 和 `ninja` 都使用该环境中的工具；特别是 Ubuntu 22.04 自带 CMake 可能低于本项目要求的 3.24。Conda 需要配置有效的软件源；若通道只有 `nodefaults`，创建环境时需用 `--override-channels -c conda-forge` 等方式指定可用通道。

### 2. 安装构建工具（前端可选）

构建原生编译器和 SEAL/OpenFHE 后端不需要安装 `front/` 的 Python 包，也不需要 PyTorch/ONNX。跳过前端时，在已激活的 `CompilerforFHE` 环境中只安装构建工具：

```bash
python -m pip install cmake==3.31.6 ninja==1.11.1.4

python --version
cmake --version
ninja --version
```

需要使用后面的 PyTorch/ONNX 接入、Python 驱动或 Python 测试时，再在同一环境安装前端依赖。默认 SEAL/OpenFHE 后端不需要 CUDA；可选 Phantom 后端需要 CUDA toolkit，但前端仍可使用 CPU 版 PyTorch：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`pyproject.toml` 的发行包名为 `crossfhe-front`，包含前端与 CLI 包 `crossfhe`、后端包 `backend`（含 `backend.ai`）及本地函数映射 JSON；命令入口仍为 `crossfhe`。直接使用源码时，在仓库根目录设置 `export PYTHONPATH="$PWD:$PWD/front${PYTHONPATH:+:$PYTHONPATH}"`，使切换目录后仍可导入这些包。跳过前端安装时，使用原生 CTest 和直接编译 MLIR 的方式验证。无需安装另外三个上游编译器的 Python 环境。

### 3. 构建编译器和默认 CPU 后端

```bash
python scripts/bootstrap.py --jobs 4
```

脚本会按固定 revision 获取密码库源码，完成以下工作：

| 路径/产物 | 内容 |
| --- | --- |
| `third_party/llvm-project/` | 官方 `release/17.x` 的 LLVM/MLIR 子模块，使用主仓库记录的提交 |
| `third_party/SEAL/`、`third_party/OpenFHE/` | `v4.1.2` 的 SEAL、固定提交的 OpenFHE，及其 cereal 依赖 |
| `third_party/phantom-fhe/` | 固定提交的 PhantomFHE 子模块，默认仅初始化源码 |
| `build/bin/crossfhe-opt`、`crossfhe-translate` | 从根 CMake 构建的原生 FHE 编译工具 |
| `build/seal/cmake/SEALConfig.cmake` | SEAL 库配置，供模型 runner 构建使用 |
| `.crossfhe/openfhe-build/` | OpenFHE 的构建目录 |
| `.crossfhe/openfhe-install/lib/OpenFHE/` | Linux 下 OpenFHE 的安装及 CMake 配置目录 |
| `.crossfhe/toolchain.json` | 原生工具、密码库和编译工具的实际绝对路径 |

脚本使用 Release 构建，优先选择 `clang/clang++`，没有 Clang 时选择 `gcc/g++`。首次需要访问 GitHub 和 Python 包源；LLVM 编译占用内存较多，可把 `--jobs` 改为 2。此构建不会启用旧 CPU/GPU 编译路径。

四个库均由 [.gitmodules](.gitmodules) 管理，实际版本由主仓库 gitlink 固定：LLVM 来自 `release/17.x`，SEAL 对应 `v4.1.2` 标签，OpenFHE 和 PhantomFHE 使用指定提交。普通 bootstrap 使用这些固定提交；`.gitmodules` 当前还在部分 `branch` 字段中填写了版本号或提交哈希，这些值不能当作远端分支使用，不要添加 `git submodule update --remote`。仅初始化依赖、不构建时执行：

```bash
python scripts/bootstrap.py --init-only
```

PhantomFHE 是可选实验性 CUDA 后端。需要同时构建该库并注册工具链配置时执行（下例适用于本机 L20）：

```bash
python scripts/bootstrap.py --enable-cuda \
  --cuda-compiler /usr/local/cuda-13.4/bin/nvcc \
  --cuda-architectures 89 --jobs 4
```

除 `--init-only` 外，bootstrap 成功后会重写 `.crossfhe/toolchain.json`；不带 `--enable-cuda` 会移除原有的 Phantom 注册。已配置 GPU 的环境重新 bootstrap 时，应继续带上 `--enable-cuda` 及相应 CUDA、源码和构建目录参数。日常增量更新使用 `cmake --build` 或下文测试脚本，无需重新 bootstrap。

源码包不包含第三方源码和 Git 元数据。使用源码包时，先准备好四个依赖目录，通过 `--llvm-source`、`--seal-source`、`--openfhe-source`、`--phantom-source` 指定路径。脚本仅通过 Git 子模块初始化依赖，不额外实现克隆回退，也不自动下载编译工具链。

已有依赖或工具链时，可以引用它们：

```bash
python scripts/bootstrap.py \
  --llvm-source /opt/crossfhe/llvm-project \
  --seal-source /opt/crossfhe/SEAL \
  --openfhe-source /opt/crossfhe/OpenFHE \
  --toolchain /opt/llvm/bin \
  --jobs 4
```

这些是路径示例；`--toolchain` 目录须包含兼容的 `clang` 和 `clang++`。自备依赖应匹配 [UPSTREAM.md](docs/UPSTREAM.md) 中的 revision，调用者需自行核对：脚本会记录实际 revision，但不会逐一校验所有自定义源码的版本；启用 Phantom 时会单独检查其 Git revision。

### 4. 确认构建成功与增量构建

```bash
test -x build/bin/crossfhe-opt
test -x build/bin/crossfhe-translate
test -f .crossfhe/toolchain.json
build/bin/crossfhe-opt --show-dialects
ctest --test-dir build --output-on-failure
```

方言列表应包含 `arith`、`func`、`scfhe` 和 `heir`。CTest 当前包含 7 项原生检查。修改编译器 C++ 或 ODS 后，用增量构建更新工具：

```bash
cmake --build build --target crossfhe-opt crossfhe-translate --parallel 4
```

### 5. 只构建原生编译器

仅开发方言和 pass 时不必构建 SEAL/OpenFHE，也不必安装 PyTorch。先准备 C++17 编译器、CMake >= 3.24、Ninja，并初始化仓库 LLVM 子模块。已有上述 Python 环境时可直接执行：

```bash
cmake -S . -B build-compiler -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCROSSFHE_BUILD_BACKEND=OFF
cmake --build build-compiler --target crossfhe-opt crossfhe-translate --parallel 4
ctest --test-dir build-compiler --output-on-failure
```

工具位于 `build-compiler/bin/`。此路线没有生成 `.crossfhe/toolchain.json`，可直接处理 MLIR，尚不能通过 Python 驱动完成加密执行。

若使用兼容的已安装 MLIR，可额外设置 `-DCROSSFHE_LLVM_SOURCE=` 和 `-DMLIR_DIR=/path/to/mlir/cmake`；默认使用仓库固定的 LLVM 17 子模块。更换编译器、LLVM 或构建类型时使用新的 build 目录，避免混用缓存。

## 使用编译器

### 直接编译 MLIR

下面使用仓库的 ReLU MLIR fixture，不经过 PyTorch/ONNX。每一步使用 verifier，并保存阶段产物：

```bash
mkdir -p artifacts/manual
build/bin/crossfhe-opt tests/compiler/inputs/relu.mlir \
  --ppml-to-scfhe --verify-each -o artifacts/manual/02_scfhe.mlir
build/bin/crossfhe-opt artifacts/manual/02_scfhe.mlir \
  --ppml-to-heir --verify-each -o artifacts/manual/03_fhe.mlir
build/bin/crossfhe-opt artifacts/manual/03_fhe.mlir \
  --ppml-fhe-optimize --verify-each -o artifacts/manual/04_optimized.mlir
build/bin/crossfhe-translate artifacts/manual/04_optimized.mlir \
  --backend=openfhe-cross -o artifacts/manual/model.hpp
build/bin/crossfhe-translate artifacts/manual/04_optimized.mlir \
  --backend=openfhe-cross --emit-contract -o artifacts/manual/resources.json
```

默认优化会改变 IR 并重算资源契约，不再要求优化前后字节相同。纯算术 FHE IR 可选择 `--backend=seal-ckks`；默认 ReLU 使用跨方案后端，显式近似路线见 [Pass 文档](docs/development/PASSES.md)。

生成的 C++ 可以单独编译为 runner。在默认 bootstrap 安装路径下：

```bash
cmake -S backend -B artifacts/manual/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCROSSFHE_BACKEND=openfhe-cross \
  -DCROSSFHE_MODEL_HEADER="$PWD/artifacts/manual/model.hpp" \
  -DOpenFHE_DIR="$PWD/.crossfhe/openfhe-install/lib/OpenFHE"
cmake --build artifacts/manual/build --target model-runner --parallel 2
```

如果 bootstrap 使用了自定义路径，以 `.crossfhe/toolchain.json` 为准。runner 提供 `keygen/encrypt/evaluate/decrypt`；Python 驱动会自动完成调用和 tensor 数据编码。

### 接入 PPML 模型

```python
import torch
from examples.user_model import create_model
from crossfhe.pipeline import compile_torch, run_encrypted

model = create_model().eval()
x = torch.tensor([[-0.5, 0.25, 0.75]], dtype=torch.float32)
artifact, inputs, reference = compile_torch(
    model, (x,), "artifacts/my_model", backend="openfhe-cross"
)
outputs, report = run_encrypted(artifact, inputs, reference=reference)
```

内置模型可直接执行；建议先检查纯 CKKS，再检查跨方案链路：

```bash
python -m crossfhe demo affine --backend seal-ckks --output artifacts/affine
python -m crossfhe demo affine --backend seal-ckks --packing auto --output artifacts/affine-packed
python -m crossfhe demo square-mlp --backend seal-ckks --output artifacts/mlp
python -m crossfhe demo tiny-cnn --backend seal-ckks --output artifacts/cnn
python -m crossfhe demo poly-attention --backend seal-ckks --output artifacts/attention
python -m crossfhe demo relu --backend openfhe-cross --insecure-demo --output artifacts/relu-toy
python -m crossfhe demo relu-mlp --backend openfhe-cross --insecure-demo --output artifacts/relu-mlp-toy
python -m crossfhe demo maxpool --backend openfhe-cross --insecure-demo --output artifacts/maxpool-toy
```

SEAL 默认使用 `tc128` 参数检查。OpenFHE 默认请求 `HEStd_128_classic` 与 FHEW `STD128`，密钥生成和转换资源开销较大。仅调试可显式加 `--insecure-demo`；该配置是 `INSECURE_TOY`，不能作为安全参数实验结果。

默认安全参数的跨方案例子去掉该选项：

```bash
python -m crossfhe demo relu --backend openfhe-cross --output artifacts/relu-secure
```

历史标量基线中，这个例子的评估键文件约 11.9 GB，密钥生成耗时数分钟；运行前需要足够内存和磁盘。`poly-attention` 是显式多项式 attention 核心，不是完整 Transformer/BERT。

### 导出、编译和运行自己的模型

仓库已有 `examples.user_model:create_model`。下面准备与其输入匹配的 NPZ，然后分步执行：

```bash
mkdir -p artifacts/user
python -c 'import numpy as np; np.savez("artifacts/user/inputs.npz", input_0=np.array([[-0.5, 0.25, 0.75]], dtype=np.float32))'
python -m crossfhe export \
  --factory examples.user_model:create_model \
  --inputs artifacts/user/inputs.npz --output artifacts/user/model.onnx
python -m crossfhe compile artifacts/user/model.onnx \
  --backend openfhe-cross --output artifacts/user/compiled
python -m crossfhe run artifacts/user/compiled \
  --inputs artifacts/user/model.inputs.npz \
  --reference artifacts/user/model.reference.npz --insecure-demo
```

替换模型时修改 factory 模块和输入数据即可。输入 NPZ key 对应 ONNX 输入名，shape 必须与导出时一致。`export` 生成 `.inputs.npz` 和 `.reference.npz`；参考结果由 PyTorch 得到。输入 tensor 必须是 float32/float64，模型输出为 tensor 或 tensor tuple。

### 查看编译产物和执行报告

```text
artifacts/<model>/
  model.onnx              原始程序
  01_arith.mlir           标量路径的前端结果
  02_scfhe.mlir           标量路径的方案无关 FHE 表示
  01_matvec.mlir          打包路径的公开仿射 MatVec 表示
  03_fhe.mlir             RLWE/LWE 和转换操作
  04_optimized.mlir       优化结果
  resources_before.json  优化前资源契约
  resources_after.json   优化后资源契约
  model.hpp              模型专用 C++ evaluator
  manifest.json          tensor mapping、方案、深度、版本和文件 hash
  compilation_status.json 编译状态：compiling / ready
  build/bin/model-runner  真实密码库执行程序
  logs/                  编译日志
  runs/run-*/client/      客户端密钥、明文输入与解密输出
  runs/run-*/server/      公开/评估键和输入输出密文
  runs/run-*/logs/        四个执行阶段的日志
  latest_report.json     最近一次执行的精度和分段计时
```

默认标量路径生成 `01_arith.mlir`、`02_scfhe.mlir`；使用 `--packing diagonal|bsgs|auto` 时改为生成 `01_matvec.mlir`，随后直接降为 `03_fhe.mlir`。`compilation_status.json` 为 `ready` 后才能运行 artifact；构建失败时需修复原因并重新编译。

CLI 会打印解密输出和 report。参考比较成功时 `validation.passed` 为 `true`；不匹配或编译失败会以非零状态退出。服务端只处理密文，最终输出由独立客户端进程解密。

## 测试与结果

### 运行测试

首次测试前，按上文安装依赖并完成相应后端的 bootstrap；已完成配置的环境只需激活 `CompilerforFHE`。仅前端可用 `python -m pip install '.[dev]'` 安装依赖，不要求 bootstrap。

在仓库根目录选择测试范围：

```bash
conda activate CompilerforFHE
python scripts/run_tests.py                   # 默认 quick：增量构建 + CTest + 非加密测试
python scripts/run_tests.py --suite front     # 仅前端，不需要原生工具链或 PyTorch
python scripts/run_tests.py --suite compiler  # 编译器增量构建 + CTest + 编译器 Python 测试
python scripts/run_tests.py --suite cpu       # 全部 CPU 测试，含 SEAL/OpenFHE 真实加密
python scripts/run_tests.py --suite gpu       # CTest + Phantom GPU 真实加密测试
python scripts/run_tests.py --suite all       # 全部 CPU/GPU 测试
```

没有 GPU 的机器使用 `--suite cpu`；`gpu` / `all` 要求 CUDA、可用 GPU 和已通过 `--enable-cuda` 注册的后端。CPU 套件会更新 SEAL 和 OpenFHE（含安装），GPU 套件会更新 Phantom。脚本自动加载仓库源码，无需额外 editable install 或设置 `PYTHONPATH`。

每次运行在 `.crossfhe/test-runs/` 下创建独立目录，保存构建和测试日志、JUnit、实际命令及 `summary.json`，终端打印 `Reports:` 路径。显示 `PASS` 且退出码为 `0` 表示所选测试通过；失败、跳过或未选中用例均返回非零状态。

筛选示例为 `python scripts/run_tests.py --suite cpu -- -k affine`；筛选只影响 Python 用例，CTest 仍完整执行。自定义配置、构建并发、报告目录、直接 pytest / CTest 和完整参数规则见 [测试说明](tests/README.md)。直接 pytest / CTest 不会更新二进制，需先自行构建。

### 已完成的验证

2026-11-17 精简后执行完整 `--suite all`：**235 项 Python 测试、7 项 CTest 全部通过，零失败、零跳过**，覆盖 SEAL、OpenFHE 和 Phantom CPU/GPU 路径。此次保留 legacy 与 AI 后端，清理范围、保留理由与报告位置见 [清理执行记录](CLEANUP_CANDIDATES.md#11-本次执行记录2026-11-17)。

清理前的 **2026-11-17 AI 后端首版验证**覆盖 **255 项独立 Python 测试、7 项 CTest 全部通过，零失败、零跳过**。先执行 246 项的完整 `--suite all`，随后对最终新增/修改内容执行定向检查与 quick，按测试身份去重后覆盖最终 255 项；未再次运行最终 255 项的单次 all。命令、结果及本机报告位置见 [AI 首版验证记录](docs/development/AI_BACKEND_VALIDATION.md)。

此前的 [Phantom 与 MatVec 验证](docs/VALIDATION-2026-11-17.md) 记录了原有 100 项 Python 测试、58 次加密执行及 Phantom 客户端解码修复。测试基础设施元测试已移除，AI 后端测试位于 `tests/backend/ai/`；历史通过数不代表当前用例总数。OpenFHE 回归使用显式 TOY 参数，默认安全参数的 ReLU 为单独功能测量。基线、优化和 MatVec 的历史结果分别见 [初始验证](docs/VALIDATION.md)、[本机基线报告](docs/VALIDATION-2026-11-16.md)、[优化报告](docs/development/RESULTS.md) 和 [MatVec 报告](docs/development/MATVEC.md)。

可从完成的运行中收集精度、转换次数、各阶段延迟和评估键大小：

```bash
python scripts/collect_reports.py artifacts/relu-toy artifacts/affine --output docs/measurements
python scripts/package_source.py --output CrossFHE-source.zip
```

收集结果和源码打包都排除密钥、密文、明文输入及工具链。`artifacts/`、`build/`、`.crossfhe/` 不进入 Git 或源码包。

## 常见构建和运行问题

| 现象 | 检查和处理 |
| --- | --- |
| `cmake` 版本低于 3.24 | 执行 `conda activate CompilerforFHE`，用 `command -v cmake` 确认使用该环境安装的工具 |
| LLVM 源码检查失败 | 执行 `python scripts/bootstrap.py --init-only`，确认存在 `third_party/llvm-project/mlir/include/mlir/IR/OpBase.td`，或用 `--llvm-source` 指定相同版本 |
| `dependency has local changes` | 先保留依赖目录中的本地改动，再使用干净、匹配固定 revision 的 checkout |
| `unsupported Phantom revision` | 使用主仓库固定提交的 Phantom 子模块，或用 `--phantom-source` 指向相同 revision |
| 编译过程中被系统终止 | 降低 `--jobs`，先构建编译器小目标，再检查内存/交换空间 |
| 找不到 `toolchain.json` | 完成全量 bootstrap；自定义配置可用 `python -m crossfhe --config /path/toolchain.json ...` |
| 找不到 OpenFHE/SEAL CMake 配置 | 检查 `.crossfhe/toolchain.json` 中的库路径；完整 bootstrap 会生成或安装它们 |
| `GPU tests require Phantom` / `phantom-ckks is not configured` | 按上文带 `--enable-cuda` 重新注册后端；普通 bootstrap 会覆盖原有 GPU 注册 |
| `model needs logic FHE` | 模型含比较或选择，使用 `--backend openfhe-cross` |
| 未支持的 ONNX 算子报错 | 查看 [OPERATORS.md](docs/OPERATORS.md)，确认模型算子在当前支持范围内 |
| 解密输出与参考结果不符 | 检查 report、比较数值范围和 margin；调整比较 scale 须基于数据范围校准 |
| 历史 artifact 的 CMake 路径失效 | 在新的 artifact 目录重新编译；跨机器重新 bootstrap，不复用另一台机器的绝对路径配置 |

## 当前能力与研究入口

这是静态、公开模型参数、秘密输入的推理编译器。默认路径仍逐元素展开、每个元素对应一个密文；显式 `--packing diagonal|bsgs|auto` 支持 SEAL CPU 或 Phantom CUDA 上单个公开仿射 MatVec 的周期槽位打包。通用网络布局搜索、自动方案选择、通用训练和完整 BERT 尚未实现。

OpenFHE 后端的 Logic FHE 当前是 **FHEW sign bootstrapping**，支持比较、ReLU、Where、Min/Max、MaxPool。它不是 TFHEpp，也不是任意实数 LUT。Softmax、GELU、Sigmoid、LayerNorm 等不支持的算子会报错。比较受量化、CKKS 误差、数值范围和 margin 条件约束，详见 [docs/BACKENDS.md](docs/BACKENDS.md)。

默认优化组合在 `compiler/Pipelines/FHE/OptimizationHooks.cpp` 中编排；专项实现位于同目录的 `Optimizations.cpp`、`ApproximateRelu.cpp`。后续根据 [开发计划](docs/development/PLAN.md) 扩展 `compiler/Conversion/FHE` 和方言。新增库适配放入 `backend/`；FHE IR 到后端代码的生成仍归 `compiler/CodeGen/FHE`。Phantom 的 GPU 专项代价模型与性能优化留待后续实现。

密码库函数映射可在本仓库的 [backend/api](backend/api/README.md) 查看：按 SEAL、OpenFHE、Phantom 列出 FHE IR → runtime 签名 → 实际库调用、约束与固定 revision。执行代码位于 `compiler/CodeGen/FHE` 和各后端 runtime，无需复制完整上游文档。

AI 辅助后端实现位于 [backend/ai](backend/ai/README.md)（Python 包 `backend.ai`），负责公开任务、提示词、模型 provider、候选调度和验证驱动；`front/crossfhe` 接入 AI 命令，AI 实现直接复用 `backend.execution` 和 `backend.toolchain`。命令仍为 `python -m crossfhe ai prepare|baseline|run`，默认使用上述本地函数映射，并封存于任务的 `api_context.json.function_mapping`。模型仅提出固定 DAG 的调度 JSON，经原生校验生成 C++、真实构建并进行同密文差分。支持现有 SEAL、OpenFHE、Phantom 后端；独立 TFHE 后端尚未实现。默认流程不需要 `/shizy/fhe-coder`，`--documentation-root` 可选追加 SEAL/OpenFHE 辅助摘要。离线运行、在线模型配置和当前边界见 [AI 后端说明](docs/development/AI_BACKEND.md)。

开发指南：[新建方言](docs/ADDING_DIALECT.md)、[新增 pass](docs/ADDING_PASS.md)。模块细节：[编译器](compiler/README.md)、[命令入口](tools/README.md)、[前端与 CLI](front/README.md)、[后端](backend/README.md)、[测试](tests/README.md)。方言和许可证来源见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。
