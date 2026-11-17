# 已实现的 FHE pass 与使用方式

实现位于 `compiler/Pipelines/FHE/`，通过现有 `registerCrossFHEPasses()` 注册，使用 MLIR SSA/OpBuilder 修改 IR，不进行文本替换。目标函数必须带 `ppml.fhe_enabled = true`；正常 `--ppml-to-scfhe --ppml-to-heir` 路径自动携带此标记，显式 false 或未标记的函数跳过。支持单块标量 evaluator，以及显式周期布局的 MatVec 打包路径。

| CLI | 作用 | 启用方式 |
| --- | --- | --- |
| `--ppml-fhe-simplify` | 消除公开零的加/减、乘一；零乘法转公开零 materialize；合并相同分支的 select | 默认优化的一部分 |
| `--ppml-fhe-cse` | 类型、操作数顺序、编码等语义属性相同才共享；包含重复的 CKKS→FHEW→sign→CKKS 链；排除随机 materialize | 默认优化的一部分 |
| `--ppml-fhe-dce` | 从块尾反向删除未使用的已知 FHE/constant 操作，未知操作不作为纯函数消除 | 默认优化的一部分 |
| `--ppml-fhe-refresh-contract` | 重算深度、转换计数、各类原语数量；保留 17 层切换预算规则 | 每个优化 pass 后自动执行，也可单独调用 |
| `--ppml-fhe-balance-mul` | 对单次使用的纯密文乘法子树，按深度优先结合，且仅在降低深度时改写 | 显式 opt-in，允许重关联；会改变 CKKS 误差 |
| `--ppml-scfhe-approx-relu='bound=B max-error=E'` | 在 SCFHE 中识别 ReLU 的比较/select 模式，生成有界二次多项式并记录局部预算 | 显式 opt-in，必须在 `--ppml-to-heir` 前；`B>0, E>=B/16` |
| `--ppml-fhe-optimize` | simplify → CSE → DCE → 资源契约刷新 | 默认流水线；不默认平衡或近似 |
| `--ppml-fhe-matvec='strategy=diagonal\|bsgs\|auto'` | 将单个公开仿射 `heir.matvec` 降为 CKKS SIMD 对角线/BSGS 原语，省略零对角线，复用 baby rotations | 显式 `packing=` / `--packing`，当前仅 SEAL CPU；见 [完整说明](MATVEC.md) |

`ppml.resources` 记录 operations、密密/密明乘法、加减、select、materialize、三类切换调用，以及打包 rotations / plaintext_additions 的数量，属于结构化资源分析，不是噪声上界或目标硬件的延迟预测。每次乘法和 select 仍按既有后端约定消耗一层；OpenFHE 有逻辑运算时额外保留 17 层。未知操作和不支持的 evaluator 结构报错，不静默转换。CSE 排除随机 `materialize` 与 `packed_zero`，可共享同类型、同输入、同系数的确定性打包原语。

## 原生调用

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CompilerforFHE
cmake --build build --target crossfhe-opt crossfhe-translate --parallel 8

build/bin/crossfhe-opt tests/compiler/inputs/repeated_relu.mlir \
  --ppml-to-scfhe --ppml-to-heir --ppml-fhe-optimize --verify-each \
  -o artifacts/shared.mlir
build/bin/crossfhe-translate artifacts/shared.mlir \
  --backend=openfhe-cross --emit-contract -o artifacts/shared-resources.json

build/bin/crossfhe-opt tests/compiler/inputs/balanced_mul.mlir \
  --ppml-to-scfhe --ppml-to-heir --ppml-fhe-balance-mul --verify-each

build/bin/crossfhe-opt tests/compiler/inputs/relu.mlir --ppml-to-scfhe \
  '--ppml-scfhe-approx-relu=bound=1 max-error=0.0625' \
  --ppml-to-heir --ppml-fhe-optimize --verify-each -o artifacts/relu-poly.mlir
build/bin/crossfhe-translate artifacts/relu-poly.mlir --backend=seal-ckks \
  -o artifacts/relu-poly.hpp
```

自行创建 `artifacts/` 后运行。最后的 ReLU 路径改变了函数近似，需明确接受指定区间的局部误差；不能用于要求精确比较的 MaxPool、Where 或任意 sign 替换。

## Python / CLI

```bash
export PYTHONPATH="$PWD:$PWD/front${PYTHONPATH:+:$PYTHONPATH}"
python -m crossfhe demo affine --backend seal-ckks --optimization scalar --output artifacts/affine-opt
python -m crossfhe demo affine --backend seal-ckks --optimization none --output artifacts/affine-base
```

`--optimization` 可选 `none`、`scalar`（默认）或 `depth`；`depth` 在默认优化后显式运行乘法平衡。API 同名参数传给 `compile_onnx`/`compile_torch`。

`--relu-bound` 与 `--relu-max-error` 必须同时提供；预算在编译期校验，区间是调用者对每个 ReLU 内部输入的声明，不是对模型原始输入的自动证明。运行时 CLI 的 `--atol/--rtol` 是最终输出比较容差，两者不是同一个概念；近似不能自动放宽最终验证阈值。

```python
artifact, inputs, reference = compile_torch(
    model, tensors, "artifacts/model", backend="seal-ckks",
    optimization="scalar", relu_bound=1.0, relu_max_error=0.0625,
)
outputs, report = run_encrypted(
    artifact, inputs, reference=reference, atol=0.063, rtol=0,
)
```

这只是单个已声明有界 ReLU 的功能示例；后接任意层时必须另行确定输出误差预算。

## 产物契约

- `03_fhe.mlir` 保存基线 HEIR（若显式开启 ReLU 近似，则已包含该近似）；`04_optimized.mlir` 保存标量优化/深度优化之后的 IR。
- `resources_before.json` / `resources_after.json` 由原生 translator 从 typed IR 导出，不通过 Python 文本正则推断。
- manifest 的 `resource_analysis`、`multiplicative_depth` 和 `logic_comparisons` 使用实际生成图；`source_logic_comparisons` 保留前端原始比较数，`approximation` 保存局部近似契约。
- manifest 的 `optimization` 为 `none` / `scalar` / `depth`。原来要求 03/04 字节相同的 identity 检查已替换为资源约束与实际加密结果验证。
- artifact hash 增加两个阶段的 IR 和资源 JSON；修改这些文件后原有执行流程会拒绝继续运行。
- `none` 关闭图变换，但仍重新分析资源；其用途是建立同一编译器和库版本上的 A/B 基线。

测试、实际资源削减及误差记录见 [RESULTS.md](RESULTS.md)。
