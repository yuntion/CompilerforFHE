# CrossFHE Compiler

CrossFHE 是基于 MLIR 的同态加密编译器。项目主干是 `compiler/`：它定义 FHE 方言，将公开算术程序降级为加密计算，并生成调用 FHE 库的 C++ evaluator。`front/` 负责 PyTorch/ONNX 接入，`backend/` 负责密码库适配和加密执行。

当前研究方向是 **CKKS 算术 + Logic FHE 非线性计算**。本版采用 Libra 的 SCFHE 方言及 HEIR/CHLOE 的 RLWE/LWE 方言，接入 Microsoft SEAL 和 OpenFHE。优化 pass 留作后续研究入口。

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
        v  compiler/Pipelines/FHE/: 当前为 identity 优化入口
FHE MLIR
        |
        v  compiler/CodeGen/FHE/: 合法性检查、模型专用 C++ 生成
model.hpp
        |
        v  backend/: SEAL CKKS 或 OpenFHE CKKS <-> FHEW
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
│   ├── Pipelines/FHE/      pass 声明、注册和预留优化入口
│   └── CodeGen/FHE/        FHE MLIR -> C++ 的可复用代码生成库
├── front/crossfhe/         PyTorch 导出、ONNX 导入、Python 驱动
├── backend/
│   ├── seal/               SEAL CKKS 运行时和客户端/服务端 runner
│   └── openfhe/            CKKS/FHEW 转换、逻辑自举和 runner
├── tools/                 crossfhe-opt、crossfhe-translate 命令入口
├── tests/                 front、compiler、backend、e2e 分层测试
├── examples/              可接入的 PyTorch 示例模型
├── scripts/               固定依赖构建、结果收集、源码打包
├── build-tools/cmake/      LLVM/MLIR 配置及保留的旧构建入口
├── third_party/            LLVM 源码与按固定版本获取的密码库
├── docs/                   方言来源、数值契约、扩展设计与验证记录
├── CMakeLists.txt          默认构建 FHE 编译器的根入口
├── pyproject.toml          front 的 Python 包与测试配置
└── requirements.txt        已验证的 Python 工具版本
```

| 模块 | 主要接口/产物 | 如何验证 |
| --- | --- | --- |
| `front` | `export_torch`、`import_onnx`，输出 `func/arith` MLIR 和 tensor mapping | `python -m pytest tests/front -q --basetemp .crossfhe/test-front` |
| `compiler/Dialect` | `!scfhe.scfhecipher`、`!heir.rlwecipher`、`!heir.lweciphervec` | CTest 方言注册测试；`crossfhe-opt --show-dialects` |
| `compiler/Conversion` | `--ppml-to-scfhe`、`--ppml-to-heir` | CTest 降级测试；`python -m pytest tests/compiler -q --basetemp .crossfhe/test-compiler` |
| `compiler/Pipelines` | `--ppml-fhe-optimize`，当前不改变 IR | 端到端测试比较优化前后 IR 的字节内容 |
| `compiler/CodeGen` | `generateCpp`，输出 `model.hpp` | CTest 原生代码生成测试及端到端 C++ 编译 |
| `backend` | `constant/add/sub/mul/to_logic/sign/to_ckks_mask/select`，输出密文 | `python -m pytest tests/backend --run-fhe -q --basetemp .crossfhe/test-backend` |
| `tools` | `crossfhe-opt`、`crossfhe-translate` | `ctest --test-dir build --output-on-failure` |
| 全链路 | PyTorch 参考输出与客户端解密输出比较 | `python -m pytest --run-fhe -q --basetemp .crossfhe/test-all` |

旧仓库的 TOSA/CPU/GPU 文件仍保留在原目录，只有显式设置 `CROSSFHE_ENABLE_LEGACY_COMPILER=ON` 才进入旧构建。默认 FHE 路径不构建这些目标；本版未验证旧路径，也未实现 FHE GPU codegen。

### 命令入口为何分开

| 工具 | 输入 -> 输出 | 当前职责 |
| --- | --- | --- |
| `crossfhe-opt` | MLIR -> MLIR | 注册 FHE 方言，运行 lowering、分析和优化 pass |
| `crossfhe-translate` | 最终 FHE MLIR -> C++ | 检查后端能力，调用 `compiler/CodeGen/FHE` 生成 `model.hpp` |
| `nisl-opt` | 旧入口，尚未实现 IR 处理 | `nislOptRunMain` 当前只打印信息并返回 `1` |
| `nisl-compile` | 旧 TOSA/CPU/GPU 编译路径 | 调用 `nislRunMain` 和原 NISL pipeline，当前不注册 FHE 方言/pass |

`opt` 和 `translate` 是同一个编译器的两个阶段，不是两个重复的编译器。旧 `nisl-main` 的入口形式可以复用，但当前实现不能直接承担 FHE 流水线；本版按默认 FHE 与可选 legacy 两条路径分别构建。工具实现和后续复用位置见 [tools/README.md](tools/README.md)。

## 构建整个项目

以下使用 **Ubuntu 24.04 LTS、Bash、Python 3.12**。除安装系统依赖外，所有命令都在包含 `compiler/`、`front/`、`backend/` 的仓库根目录执行。Ubuntu 22.04 的默认 Python 是 3.10；若使用 3.12，请先准备对应解释器和 venv 支持。

### 1. 准备系统环境

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential clang git pkg-config ca-certificates \
  python3.12 python3.12-venv python3.12-dev

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

每次打开新终端都先执行 `source .venv/bin/activate`。后续的 `python`、`cmake`、`ctest` 和 `ninja` 都使用该环境中的工具；特别是 Ubuntu 22.04 自带 CMake 可能低于本项目要求的 3.24。

### 2. 安装前端与构建工具

当前运行库后端使用 CPU，不需要 CUDA。先安装 CPU 版 PyTorch，再安装固定版本依赖和前端驱动：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

python --version
cmake --version
ninja --version
```

`pyproject.toml` 安装的 Python 包名为 `crossfhe-front`，导入名和命令入口仍是 `crossfhe`。无需安装另外三个上游编译器的 Python 环境。

### 3. 构建编译器和两个库后端

```bash
python scripts/bootstrap.py --jobs 4
```

脚本会按固定 revision 获取密码库源码，完成以下工作：

| 路径/产物 | 内容 |
| --- | --- |
| `third_party/llvm-project/` | 使用仓库已有的 LLVM/MLIR 快照 |
| `third_party/SEAL/`、`third_party/OpenFHE/` | 下载的 SEAL、OpenFHE 源码及 OpenFHE cereal 依赖 |
| `build/bin/crossfhe-opt`、`crossfhe-translate` | 从根 CMake 构建的原生 FHE 编译工具 |
| `build/seal/cmake/SEALConfig.cmake` | SEAL 库配置，供模型 runner 构建使用 |
| `.crossfhe/openfhe-build/` | OpenFHE 的构建目录 |
| `.crossfhe/openfhe-install/lib/OpenFHE/` | Linux 下 OpenFHE 的安装及 CMake 配置目录 |
| `.crossfhe/toolchain.json` | 原生工具、密码库和编译工具的实际绝对路径 |

脚本使用 Release 构建，优先选择 `clang/clang++`，没有 Clang 时选择 `gcc/g++`。首次需要访问 GitHub 和 Python 包源；LLVM 编译占用内存较多，可把 `--jobs` 改为 2。此构建不会启用旧 CPU/GPU 编译路径。

源码包不含完整 LLVM 树。如果使用源码包，需用 `--llvm-source` 指定原仓库的同一快照；不要直接用发行版的任意 LLVM 替代方言所依赖的版本。

已有依赖或工具链时，可以引用它们：

```bash
python scripts/bootstrap.py \
  --llvm-source /opt/crossfhe/llvm-project \
  --seal-source /opt/crossfhe/SEAL \
  --openfhe-source /opt/crossfhe/OpenFHE \
  --toolchain /opt/llvm/bin \
  --jobs 4
```

这些是路径示例；`--toolchain` 目录须包含兼容的 `clang` 和 `clang++`，指定的密码库 Git checkout 必须匹配 [UPSTREAM.md](docs/UPSTREAM.md) 中的 revision。

### 4. 确认构建成功与增量构建

```bash
test -x build/bin/crossfhe-opt
test -x build/bin/crossfhe-translate
test -f .crossfhe/toolchain.json
build/bin/crossfhe-opt --show-dialects
ctest --test-dir build --output-on-failure
```

方言列表应包含 `arith`、`func`、`scfhe` 和 `heir`。CTest 当前包含 3 项原生检查。修改编译器 C++ 或 ODS 后，用增量构建更新工具：

```bash
cmake --build build --target crossfhe-opt crossfhe-translate --parallel 4
```

### 5. 只构建原生编译器

仅开发方言和 pass 时不必构建 SEAL/OpenFHE，也不必安装 PyTorch。先准备 C++17 编译器、CMake >= 3.24、Ninja 和仓库 LLVM 快照。已有上述 Python 环境时可直接执行：

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

若使用兼容的已安装 MLIR，可额外设置 `-DCROSSFHE_LLVM_SOURCE=` 和 `-DMLIR_DIR=/path/to/mlir/cmake`；当前验证基线使用仓库快照。更换编译器、LLVM 或构建类型时使用新的 build 目录，避免混用缓存。

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
cmp artifacts/manual/03_fhe.mlir artifacts/manual/04_optimized.mlir
```

当前优化入口为空，`cmp` 应返回成功。纯算术 FHE IR 可选择 `--backend=seal-ckks`；ReLU 需要跨方案后端。

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
  01_arith.mlir           前端结果
  02_scfhe.mlir           方案无关 FHE 表示
  03_fhe.mlir             RLWE/LWE 和转换操作
  04_optimized.mlir       优化结果，当前与上一阶段相同
  model.hpp              模型专用 C++ evaluator
  manifest.json          tensor mapping、方案、深度、版本和文件 hash
  build/bin/model-runner  真实密码库执行程序
  logs/                  编译日志
  runs/run-*/client/      客户端密钥、明文输入与解密输出
  runs/run-*/server/      公开/评估键和输入输出密文
  runs/run-*/logs/        四个执行阶段的日志
  latest_report.json     最近一次执行的精度和分段计时
```

CLI 会打印解密输出和 report。参考比较成功时 `validation.passed` 为 `true`；不匹配或编译失败会以非零状态退出。服务端只处理密文，最终输出由独立客户端进程解密。

## 测试与结果

原生 CTest 直接检查方言注册、降级和代码生成，不依赖 PyTorch。Python 测试覆盖前端拒绝规则、原生诊断、编译产物完整性和真实加密执行。未加 `--run-fhe` 时会跳过加密集成测试。

```bash
ctest --test-dir build --output-on-failure
python -m pytest -q --basetemp .crossfhe/test-unit
python -m pytest --run-fhe -q \
  --basetemp .crossfhe/test-all --junitxml docs/validation-junit.xml
```

按模块执行：

```bash
python -m pytest tests/front -q --basetemp .crossfhe/test-front
python -m pytest tests/compiler -q --basetemp .crossfhe/test-compiler
python -m pytest tests/backend --run-fhe -q --basetemp .crossfhe/test-backend
python -m pytest tests/e2e --run-fhe -q --basetemp .crossfhe/test-e2e
```

未完成 bootstrap 时，原生 Python 测试可能跳过。完整验证要使用 `--run-fhe`，并确认没有意外跳过。`--basetemp` 指向可重新生成的测试目录，不要用自己的模型或长期密钥目录。

集成测试覆盖仿射、平方 MLP、小 CNN、多项式 attention、ReLU、ReLU-MLP、MaxPool、分组/空洞卷积、BatchNorm 和秘密矩阵乘。OpenFHE 回归测试使用显式 TOY 参数；默认安全参数的 ReLU 单独测量。实际记录见 [docs/VALIDATION.md](docs/VALIDATION.md)，数据在 `docs/measurements/results.json` 和 `results.csv`。

当前基线包含 **3 项原生 CTest 和 34 项 Python 测试**，此前验证全部通过。历史 XML 在 `docs/native-validation-junit.xml` 与 `docs/validation-junit.xml`。**Ubuntu 上的完整构建与加密执行尚未在本工作环境实测**，需在目标 Ubuntu 机器按上述步骤确认。默认安全参数的 ReLU 是历史功能测量，不能用作本机性能结果。

可从完成的运行中收集精度、转换次数、各阶段延迟和评估键大小：

```bash
python scripts/collect_reports.py artifacts/relu-toy artifacts/affine --output docs/measurements
python scripts/package_source.py --output CrossFHE-source.zip
```

收集结果和源码打包都排除密钥、密文、明文输入及工具链。`artifacts/`、`build/`、`.crossfhe/` 不进入 Git 或源码包。

## 常见构建和运行问题

| 现象 | 检查和处理 |
| --- | --- |
| `cmake` 版本低于 3.24 | 激活 `.venv`，用 `command -v cmake` 确认使用 requirements 安装的工具 |
| LLVM 源码检查失败 | 确认存在 `third_party/llvm-project/mlir/include/mlir/IR/OpBase.td`，或指定同一快照的 `--llvm-source` |
| `dependency revision mismatch` | 使用独立、匹配固定 revision 的依赖 checkout；不要在有自己改动的依赖目录上强制覆盖 |
| 编译过程中被系统终止 | 降低 `--jobs`，先构建编译器小目标，再检查内存/交换空间 |
| 找不到 `toolchain.json` | 完成全量 bootstrap；自定义配置可用 `python -m crossfhe --config /path/toolchain.json ...` |
| 找不到 OpenFHE/SEAL CMake 配置 | 检查 `.crossfhe/toolchain.json` 中的库路径；完整 bootstrap 会生成或安装它们 |
| `model needs logic FHE` | 模型含比较或选择，使用 `--backend openfhe-cross` |
| 未支持的 ONNX 算子报错 | 查看 [OPERATORS.md](docs/OPERATORS.md)，确认模型算子在当前支持范围内 |
| 解密输出与参考结果不符 | 检查 report、比较数值范围和 margin；调整比较 scale 须基于数据范围校准 |
| 历史 artifact 的 CMake 路径失效 | 在新的 artifact 目录重新编译；跨机器重新 bootstrap，不复用另一台机器的绝对路径配置 |

## 当前能力与研究入口

这是静态、公开模型参数、秘密输入的推理基线。一个 tensor 元素对应一个密文，循环展开；尚无 tensor packing、自动方案选择、通用训练或完整 BERT。

OpenFHE 后端的 Logic FHE 当前是 **FHEW sign bootstrapping**，支持比较、ReLU、Where、Min/Max、MaxPool。它不是 TFHEpp，也不是任意实数 LUT。Softmax、GELU、Sigmoid、LayerNorm 等不支持的算子会报错。比较受量化、CKKS 误差、数值范围和 margin 条件约束，详见 [docs/BACKENDS.md](docs/BACKENDS.md)。

后续优化应加入 `compiler/Pipelines/FHE/OptimizationHooks.cpp`，必要时扩展 `compiler/Conversion/FHE` 和方言。新增库适配放入 `backend/`；FHE IR 到后端代码的生成仍归 `compiler/CodeGen/FHE`。GPU 路线留待后续实现。

开发指南：[新建方言](docs/ADDING_DIALECT.md)、[新增 pass](docs/ADDING_PASS.md)。模块细节：[编译器](compiler/README.md)、[命令入口](tools/README.md)、[前端](front/README.md)、[后端](backend/README.md)、[测试](tests/README.md)。方言和许可证来源见 [docs/UPSTREAM.md](docs/UPSTREAM.md)，重构前后路径对照见 [docs/MIGRATION.md](docs/MIGRATION.md)。
