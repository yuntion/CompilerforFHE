# FHE Compiler

`compiler/` 是项目核心。它用原生 C++ MLIR API 实现方言、降级、pass 注册和库调用代码生成，可直接编译 MLIR。模型接入由 `front/` 提供，密码执行由 `backend/` 提供。

## 方言

| 目录 | 来源 | 编译语义 |
| --- | --- | --- |
| `Dialect/SCFHE/IR` | Libra 固定版本的 ODS 和声明头 | 同态算术、比较、选择，作为通向方案相关操作的中间层 |
| `Dialect/SCFHE/SCFHEDialect.cpp` | 本项目的 LLVM 17 兼容注册实现 | 注册操作/类型、旧版本接口适配；canonicalization 入口为空 |
| `Dialect/HEIR/include/heir/IR/FHE` | CHLOE 扩展的 HEIR ODS 和声明头 | RLWE 算术、LWE 逻辑、depack/repack 转换 |
| `Dialect/HEIR/HEIRDialect.cpp` | CHLOE 原始注册实现 | 生成类型、操作和 dialect 注册 |

HEIR 与 CHLOE 使用相同的 `heir` 方言名，因此只注册 CHLOE 的扩展版本。这里使用的是用户指定 NDSS HEIR 项目，非 Google 的同名项目。许可证与来源随方言迁入各自目录；固定 revision 和兼容改动见 [UPSTREAM.md](../docs/UPSTREAM.md)。

TableGen 输出位于 `build/compiler/Dialect/...`，源码 ODS 位于本目录，均不在 `front/` 或另一个独立子项目中。

## 编译库与 pass

| CMake 目标 | 文件/模块 | 功能 |
| --- | --- | --- |
| `CrossFHEHEIR` | `Dialect/HEIR` | RLWE/LWE 方言 |
| `CrossFHESCFHE` | `Dialect/SCFHE` | SCFHE 方言 |
| `CrossFHEConversion` | `Conversion/FHE` | `--ppml-to-scfhe`、`--ppml-to-heir` |
| `CrossFHEPipelines` | `Pipelines/FHE` | pass 注册与 `--ppml-fhe-optimize` |
| `CrossFHECodeGen` | `CodeGen/FHE` | `generateCpp(ModuleOp, backend, output)` |

`tools/crossfhe-opt.cpp` 和 `crossfhe-translate.cpp` 是薄命令入口，链接以上编译库。代码生成检查 IR 和后端能力，遇到未实现的操作报错；它生成 `model.hpp`，不会在编译过程中解密秘密输入。

ReLU 的降级是：

```text
arith.cmpf / arith.select
  -> scfhe.cmp / scfhe.select
  -> heir.rlwesub
  -> heir.depack     CKKS -> LWE
  -> heir.lut_lt     逻辑 sign bootstrap
  -> heir.repack     sign -> CKKS mask
  -> heir.select    CKKS mask 选择
```

转换的可执行编码契约属于库适配，当前使用 OpenFHE FHEW。`noise = 0` 是上游 ODS 必需字段，附带 `ppml.noise_status = "unanalysed"`；它不是已证明噪声上界。

## 添加优化

`Pipelines/FHE/OptimizationHooks.cpp` 编排手写的简化、CSE、DCE 与资源契约刷新；独立 pass 另含显式乘法平衡和有界 ReLU 近似。函数通过 `ppml.fhe_enabled` 标记控制触发。使用方法、数值前提及当前能力边界见 [Pass 文档](../docs/development/PASSES.md)，文献依据见 [阅读记录](../docs/development/ARTICLES.md)。

## 编译器测试

在仓库根目录完成首次 bootstrap 后，使用统一入口更新工具并执行 CTest / 编译器 Python 测试：

```text
python scripts/run_tests.py --suite compiler
python scripts/run_tests.py --suite compiler -- -k matvec
```

CTest 独立检查方言注册、MLIR 降级和 C++ 生成。Python 原生测试检查 malformed IR、后端拒绝、未完成编译产物和导出安全条件。端到端测试进一步编译生成代码并执行真实密文运算。直接使用 `python -m pytest tests/compiler` 不会更新二进制，且所选原生用例缺少工具链时统一报错。筛选无原生依赖的测试可用 `-m 'not native and not torch'`。详见 [测试说明](../tests/README.md)。

原仓库的 CPU/GPU、TOSA 和旧 NISL 工具代码保留为可选旧路径，默认 FHE 构建不启用它们。

`--ppml-fhe-matvec='strategy=diagonal|bsgs|auto'` 将 `heir.matvec` 降为显式打包原语，接入 SEAL CPU CKKS 或可选的 Phantom CUDA CKKS 后端。此 pass 使用独立的周期槽位布局和经过 verifier 检查的公开矩阵属性，详见 [研究与实现报告](../docs/development/MATVEC.md) 和 [Phantom 后端](../docs/PHANTOM_BACKEND.md)。
