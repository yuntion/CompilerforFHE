# CPU CKKS SIMD / 对角线 MatVec：研究与实现

后续独立验证：新增首槽/末槽回绕与重复加密用例，三种 packing 共 9 次真实 SEAL 加密运行，并连同原生 pass 用例得到 **18 项测试全部通过**；指令、输出和误差见 [独立测试报告](validation/matvec-wraparound/REPORT.md)。下文 87 项为此前完整回归记录。

日期：2026-11-16。开发顺序见 [MATVEC_PLAN.md](MATVEC_PLAN.md)。沿用 `CompilerforFHE` Conda 环境，不安装 front Python 包。

本机 CPU 为 AMD EPYC 9754，工具链、包版本与测试汇总见 [environment.json](validation/matvec/environment.json)。原有 OpenFHE 回归仍使用既有 TOY 测试配置；本次新增 MatVec 加密执行全部使用 SEAL tc128。

## 相关工作与采用的部分

本轮查阅本地文章全文，重点阅读线性变换、矩阵向量乘法、槽位布局和旋转成本的章节。下列工作已经提供可直接使用的算法思路：

| 工作 | 定位 | 本次采用 / 边界 |
|---|---|---|
| Halevi、Shoup，*Algorithms in HElib*，CRYPTO 2014 | 作为 LinearTransform-SmallKey 文献的参考文献 [12]，其 §5、图 9–10 解释 diagonal 与 BSGS | 依据该章节实现经典公式；没有声称直接阅读全文或移植 HElib |
| Zhang、Xin、Wu，*GALA: Greedy ComputAtion for Linear Algebra in Privacy-Preserved Neural Networks*；本地 `FHE-Matrix Multiplication/2021-GALA- Greedy ComputAtion for Linear Algebra .pdf` | §IV-A，尤其 GAZELLE hybrid calculation、对角线编码与重复输入 | 参考打包、旋转成本及重复输入的处理；未实现 GALA 的 HE/GC 分享协议和 row-encoding-share-RaS |
| Min、Lee，*Low-Latency Linear Transformations with Small Key Transmission for Private Neural Network on Homomorphic Encryption*；本地 `FHE-PPML/2024-LinearTransform-SmallKey.pdf` | §5，图 9–11，经典 diagonal / BSGS 与 parallel BSGS | 实现经典 diagonal 与 BSGS；尚未实现该文特有的 parallel BSGS、RO-MPConv 或 small-level keys。标题和作者以 PDF 正文为准，文件名前缀不作为发表年份证明 |
| Malik、Sheth、Kulkarni，*Coyote: A Compiler for Vectorizing Encrypted Arithmetic Circuits*，ASPLOS 2023，DOI `10.1145/3582016.3582057` | 导言与向量化/旋转成本论述 | 将旋转开销作为显式规划目标；未复现其通用子电路搜索和调度器 |
| Dathathri 等，*EVA: An Encrypted Vector Arithmetic Language and Compiler for Efficient Homomorphic Computation*，2020；本地 `FHE-Compiler/2020-Efficient Homomorphic Computation.pdf` | 向量算术语言、编译器和 SEAL 执行 | 参考保留向量结构、生成真实密码库调用的分层方式；未移植完整 EVA 参数分析 |

这里的 CPU SIMD 是 CKKS 的 **Single Instruction Multiple Data 槽位打包**，由 SEAL CPU 库实现。未新增 AVX intrinsic、GPU kernel 或明文矩阵计算替代密文执行。当前本机构建 `SEAL_USE_INTRIN=ON`、`SEAL_USE_INTEL_HEXL=OFF`，见构建日志。

## 数学公式与布局

对公开矩阵 `A ∈ R^(m×n)` 和秘密向量 `x ∈ R^n`，先取

```text
L = next_power_of_two(max(m,n))
A' = 将 A 补零到 L×L
x' = [x0,...,x(n-1),0,...,0]，长度 L
D_k[r] = A'[r,(r+k) mod L]
Rot(v,k)[r] = v[(r+k) mod L]       # 正数表示左旋
y' = sum_k D_k * Rot(x',k)
```

只生成存在非零元素的 `D_k`；系数是否为零采用精确判断，不按阈值丢弃小系数。偏置也补零到 L。解密后只返回前 m 个元素并恢复 ONNX 形状。

CKKS 实际旋转的周期是物理槽数 `S=N/2`，不是 n 或 L。仅把 x 放在前 n 个槽并补零到 S，会使回绕部分出错。本实现要求 `L` 为二次幂、`L≤S`，在加密时把 x' **重复 S/L 次**；所有公开对角线和偏置也重复同样次数。这样对物理 S 槽的旋转严格等价于每个 L 槽周期上的循环旋转。密钥生成同时检查安全参数和槽位容量。

BSGS 将 `k=g+j`，其中 `g` 是 baby step 大小 b 的倍数：

```text
y' = sum_g Rot(sum_j RotPlain(D_(g+j),-g) * Rot(x',j), g)
```

`RotPlain` 在编译时完成。每个需要的 `Rot(x',j)` 只生成一次；不生成空 giant group。稠密 L×L 情况下，普通方法需要 `L-1` 次旋转，BSGS 需要 `(b-1)+(L/b-1)` 次旋转，两者通常均需 L 次打包明文乘法且乘法深度为 1。

`strategy=diagonal` 固定 b=L；`bsgs` / `auto` 在二次幂候选 b 中，根据实际非零对角线所需的 baby / giant 旋转数选择最少者。没有更优方案时保留 diagonal，平局选择确定。当前两者采用相同的旋转数量成本模型；没有自动计时调优，也不保证最小延迟。后续 CSE 还可能共享相同系数的中间结果，因此最终资源以优化后 IR 为准。

## 新增代码与流水线

| 位置 | 实现 |
|---|---|
| `front/crossfhe/packed.py` | 在标量化前识别公开 MatMul/Gemm 及公开偏置，保留公开矩阵和向量形状 |
| `compiler/Dialect/HEIR/include/heir/IR/FHE/HEIR.td`、`PackedOps.cpp` | `heir.matvec`、`packed_rotate`、`packed_mul_plain`、`packed_add_plain`、`packed_add`、`packed_zero`；类型/有限系数/矩阵和周期尺寸 verifier |
| `compiler/Pipelines/FHE/MatVec.cpp` | 真正的 MLIR pass `--ppml-fhe-matvec`，构造 SSA 原语图、非零对角线筛选与 BSGS 规划 |
| `Optimizations.cpp` | 新原语的深度/资源统计，复用 CSE 和 DCE；随机 `packed_zero` 不做 CSE |
| `compiler/CodeGen/FHE/CodeGen.cpp` | 验证后端和布局、拒绝陈旧 packed 深度，输出向量常量、实际旋转集合和密文/明文数量 ABI |
| `tools/crossfhe-translate.cpp` | 输出 packed period、输入/输出元素数、策略、baby step、非零对角线及旋转 steps 的 JSON |
| `backend/seal/` | 密文旋转、周期明文乘/加、公开零重新加密；实际使用的 Galois keys 生成/保存/装载；打包加解密与实际计数 |
| `front/crossfhe/pipeline.py`、`cli.py` | 显式 `packing=` / `--packing`；生成构建产物、hash、manifest 和完整客户端/服务器执行 |

打包路径为：

```text
ONNX -> 01_matvec.mlir (heir.matvec)
     -> ppml-fhe-matvec -> 03_fhe.mlir (packed primitives)
     -> 既有优化/资源刷新 -> 04_optimized.mlir
     -> SEAL C++ -> keygen / encrypt / evaluate / decrypt
```

打包原语中的 `!heir.rlwecipher<L x f64>` 表示一个有 L 槽逻辑周期的密文。生成代码中的 `input_count/output_count` 仍是密文数；新增 `input_value_count/output_value_count` 是明文元素数，`packed_period` 表示周期。两者不能混用。服务端不持有私钥；旋转、乘法和累加全部使用真实密文 API。

Python 驱动在生成 `model.hpp` 后用 `cmake -S backend -B <artifact>/build -G Ninja` 配置 Release、SEAL 和模型头文件，再执行 `cmake --build <artifact>/build --target model-runner --parallel 2`。随后四阶段命令为（CLIENT/SERVER 均在每次新建的 run 目录下）：

```text
model-runner keygen CLIENT SERVER
model-runner encrypt CLIENT SERVER CLIENT/input.bin SERVER/input.ct
model-runner evaluate SERVER SERVER/input.ct SERVER/output.ct
model-runner decrypt CLIENT SERVER/output.ct CLIENT/output.bin
```

各阶段完整参数构造见 `front/crossfhe/pipeline.py`，阶段结果和墙钟时间记录于 `cases.json`，编译和运行日志保留在每个 artifact 的 `logs/` 与 `runs/*/logs/`。

非零矩阵的密明乘法会 rescale 一层；全零矩阵直接用公钥重新加密零，没有透明密文乘零问题且不消耗乘法层。偏置通过 `add_plain` 加到相应层。资源统计增加 `rotations`、`plaintext_additions`；保留噪声未分析状态，不将深度统计当作噪声或安全证明。

## 使用命令

所有命令在仓库根目录和原 Conda 环境执行；未安装 front 包：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CompilerforFHE
export PYTHONPATH="$PWD/front"

cmake --build build --target crossfhe-opt crossfhe-translate --parallel 8
python -m crossfhe demo affine --backend seal-ckks --packing auto \
  --output artifacts/affine-packed

# 自己的 ONNX 模型；随后照常通过 run 子命令传 NPZ 输入
python -m crossfhe compile model.onnx --backend seal-ckks --packing bsgs \
  --output artifacts/my-packed-model

# 无 PyTorch 的原生 pass / codegen 示例
build/bin/crossfhe-opt tests/compiler/inputs/matvec.mlir \
  '--ppml-fhe-matvec=strategy=bsgs' --ppml-fhe-optimize --verify-each \
  -o /tmp/matvec-lowered.mlir
build/bin/crossfhe-translate /tmp/matvec-lowered.mlir \
  --backend=seal-ckks --emit-contract -o /tmp/matvec-contract.json
build/bin/crossfhe-translate /tmp/matvec-lowered.mlir \
  --backend=seal-ckks -o /tmp/matvec.hpp
```

Python 使用 `compile_torch(model, inputs, output, backend="seal-ckks", packing="auto")` 或相同参数的 `compile_onnx`。不传 packing 时仍是既有标量路径。`optimization="none"` 可以关闭后续 CSE 等，但仍执行明确请求的 MatVec lowering。

## 验证方法

先使用真实 SEAL 测试 `3×5` 非二次幂矩阵、正负输入和偏置，再完成原生拒绝用例和完整回归。最终命令：

```bash
ctest --test-dir build --output-on-failure \
  --output-junit "$PWD/docs/development/validation/matvec/native-junit.xml"
python -m pytest --run-fhe -q -ra --basetemp .crossfhe/test-matvec-all \
  --junitxml docs/development/validation/matvec/full-junit.xml
```

新增原生测试覆盖策略规划、未标记函数跳过、全零矩阵、非法周期/尺寸/系数、非法旋转、全零明文乘法、陈旧深度与不支持的后端。真实加密测试覆盖随机稠密 16×16 的四路径 A/B、3×5、7×3、5×7、1×9、9×1、1×1、列向量/行向量、稀疏/全零矩阵、PyTorch Linear/Gemm 以及 Gemm alpha/beta/标量偏置。

每个 packed 用例均与 NumPy / PyTorch 参考比较，要求绝对误差 `≤1e-4 + 1e-5*|reference|`，并逐项核对运行时的旋转、明文乘法、密文乘法、加法、偏置加法、materialize 和 rescale 数量与原生资源计划一致；Galois keys 数量与 IR 实际旋转 steps 集合一致。稠密 A/B 的四路径使用相同矩阵、输入、SEAL tc128 策略和层数。

最终结果：**87 项 Python 测试全部通过，37 条 warning，耗时 372.26 秒；0 失败、0 跳过。7/7 CTest 通过，0.03 秒。** 相对上一轮新增 15 项原生/导入拒绝测试、13 项加密测试函数；后者实际执行 16 次加密流程。warning 来自 PyTorch 旧 ONNX 导出接口和既有常量 tracing 提示。

16×16 随机稠密矩阵 A/B 的实际测量如下。四路径均使用 SEAL tc128、N=8192、4096 个物理槽、相同 40-bit scale 和一层乘法预算；packed 周期为 16。

| 路径 | 输入/输出密文数 | 明文乘法 | 旋转 / Galois keys | evaluator 时间 ms | 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| 默认标量 | 16 / 16 | 256 | 0 / 0 | 224.671 | 1.01e-7 |
| diagonal | 1 / 1 | 16 | 15 / 15 | 44.023 | 9.85e-7 |
| BSGS | 1 / 1 | 16 | 6 / 6 | 26.679 | 5.27e-7 |
| auto（选择 BSGS） | 1 / 1 | 16 | 6 / 6 | 27.418 | 8.42e-7 |

标量到 packed 的输入密文文件从 **4,196,120 B 降至 262,265 B**，输出从 **2,098,968 B 降至 131,193 B**，均为 16 倍缩减。旋转需要额外评估键：包含参数、公钥、重线性化键和旋转键的 server bundle，标量为 1,180,180 B、diagonal 为 13,045,642 B、BSGS/auto 为 5,965,720 B。密文体积减少不代表评估键也低于标量方案。

其余代表性结果：

| 用例 | 打包明文乘法 | 旋转 | 层数 | 最大绝对误差 |
|---|---:|---:|---:|---:|
| 3×5，diagonal | 7 | 6 | 1 | 1.25e-7 |
| 3×5，BSGS | 7 | 4 | 1 | 4.87e-7 |
| 7×3，列向量，BSGS | 8 | 4 | 1 | 5.47e-7 |
| 5×7，向量在左，auto | 8 | 4 | 1 | 8.99e-7 |
| 1×9，BSGS | 9 | 5 | 1 | 1.54e-6 |
| 9×1，BSGS | 9 | 5 | 1 | 4.74e-7 |
| 1×1，auto | 1 | 0 | 1 | 2.69e-8 |
| 4×7，仅 2 个非零对角线 | 2 | 1 | 1 | 3.50e-7 |
| 4×7，全零矩阵，有偏置 | 0 | 0 | 0 | 4.28e-10 |
| 4×7，全零矩阵，无偏置 | 0 | 0 | 0 | 1.10e-9 |
| PyTorch Linear(5,3)，Gemm | 7 | 4 | 1 | 1.48e-6 |
| Gemm alpha=-0.5、beta=2、标量偏置 | 7 | 4 | 1 | 4.91e-7 |

所有 16 次 A/B 和边界运行的最大绝对误差不超过 **1.55e-6**。计时是本机单次 evaluator 内部时间，包含 runtime 编码/计算，不包括 keygen、编译和文件传输；不构成统计性能结论。加密随机性使同图误差和时间略有波动。这里验证的是公开仿射算术与实际资源，不是模型数据集准确率实验。

可复核材料：

- [完整 Python 日志](validation/matvec/full.log)、[JUnit](validation/matvec/full-junit.xml)
- [7 项 CTest 日志](validation/matvec/native.log)、[JUnit](validation/matvec/native-junit.xml)
- [16 个加密用例的资源、误差、时间、文件体积](validation/matvec/cases.json)
- [构建日志](validation/matvec/build.log)、[CLI demo 输出](validation/matvec/cli-demo.log)
- 最后只规范化新增 ODS 行的换行符，增量构建结果见 [build-final.log](validation/matvec/build-final.log)；生成内容未改变。
- 完整可执行构建、IR 和运行材料位于本机 `.crossfhe/test-matvec-all/`；生成/验证代码位于 `tests/e2e/test_packed_matvec_execution.py`。未向文档复制密钥、密文或私有输入。

开发中的 `targeted.log` / `targeted-junit.xml` 保留了初轮定向运行（26 通过、1 失败）：新增 codegen 布局检查最初误拒绝已有的 `one-scalar-per-ciphertext` 属性，已修复；最终上面的 87 项完整回归覆盖该标量 A/B 路径并全部通过。

## 当前范围

支持单个秘密向量、一个公开二维矩阵与偏置的单输出仿射变换。支持 `A@x`、`x@A`，一维向量、单行/单列表示，Gemm 的 transB、alpha/beta 和公开广播偏置，以及公开 Constant/Transpose、Identity。需要 ONNX opset 13..21、静态形状和有限浮点系数。默认导入矩阵/系数元素上限沿用 50000，周期上限 16384；实际容量仍由 SEAL 参数决定。

多个 MatVec、非线性、多个秘密向量、秘密矩阵、通用网络布局变换、跨密文分块，以及 OpenFHE packed 路径当前明确拒绝。BSGS 的求和顺序不同于标量展开，会改变 CKKS 近似误差；启用 packing 是显式选择。尚未实现 hoisting、多线程任务调度、parallel BSGS、GALA 协议、旋转密钥组合搜索或噪声/全网络误差证明。
