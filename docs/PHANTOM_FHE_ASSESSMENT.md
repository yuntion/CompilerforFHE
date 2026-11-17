# PhantomFHE 接入评估与 GPU 验证（2026-11-16）

更新：已新增实验性 `phantom-ckks` 编译器后端。当前构建、CLI 和四阶段 runner 使用方式见 [Phantom 后端](PHANTOM_BACKEND.md)。下文保留早期 P0 独立探针的评估记录，文中的“待开发”和“尚未注册”描述的是当时状态。

可以接入，建议新增可选的 `phantom-ckks` CUDA 后端，优先支持纯 CKKS 算术和现有打包 MatVec。现有 diagonal / BSGS pass 的旋转、明文乘法、累加语义可以复用，无需重新编写一套 GPU MatVec pass。

已完成源码核查、上游库构建和隔离适配实验；本次复核确认本地 revision 与 GitHub HEAD 一致，并重新构建检查、运行 GPU 探针通过。**主编译器尚未注册 Phantom 后端**，CLI 仍只支持 `seal-ckks`、`openfhe-cross`。下述实验不等于完整的前端到 GPU 客户端/服务端接入。

## 1. 上游能力与接入边界

- 项目：[encryptorion-lab/phantom-fhe](https://github.com/encryptorion-lab/phantom-fhe)。
- 实测源码：`1f4a198443b3af77118e51f53d5b8f332154b875`（`Fix #15`），CMake 项目版本 `2.1`；未修改上游源码。
- CUDA 加速 BFV、BGV、CKKS，公开版不支持 bootstrapping。
- 该 revision 的 README 将 FHEW/TFHE 标为“不开放源码”，scheme switching 尚未勾选且同样注明不开放源码。不能据此把当前 OpenFHE 的 CKKS↔FHEW 精确比较、ReLU、选择操作映射到公开版 Phantom。
- 主体许可证为 GPLv3。若发布链接 Phantom 的程序，需要评估组合程序的许可兼容性，并履行适用的源码提供等义务；把后端设为可选不自动消除这些义务。

| 当前编译器语义 | Phantom API / 做法 | 判断 |
| --- | --- | --- |
| CKKS 编码与加解密 | `PhantomCKKSEncoder`、`PhantomPublicKey`、`PhantomSecretKey` | 已验证 |
| `heir.packed_rotate` | `phantom::rotate`、`PhantomGaloisKey` | 已验证，正步数为左旋 |
| `heir.packed_mul_plain` | `multiply_plain` + `rescale_to_next_inplace` | 已验证 |
| `heir.packed_add` / `packed_add_plain` | `add` / `add_plain` | 已验证同层累加和偏置 |
| `heir.packed_zero` | 初始层公钥加密零，再 `mod_switch_to_inplace` 对齐层级 | 已验证，见第 3 节 |
| 密文乘法与重线性化 | `multiply_and_relin` + rescale | 已验证一次平方 |
| 标量常量、一般多层算术 | 编码/公钥加密常量、层级和 scale 对齐 | API 可承接，完整 runtime 待实现 |
| 近似 ReLU | 沿用已有多项式近似，消耗 CKKS 深度 | 原理可行，本次未运行该模型 |
| 精确比较、CKKS↔FHEW、逻辑自举 | 公开版没有对应可接入实现 | 保留 OpenFHE 路径 |
| BFV/BGV | 上游提供 | 当前编译器没有对应语义，暂不纳入首版 |

现有 `compiler/CodeGen/FHE/CodeGen.cpp` 已生成调用 `crossfhe::Runtime` 的 C++，适合替换运行库。实际接入还需要改动其后端白名单、runtime include 和能力检查，以及 `front/crossfhe/{cli,pipeline}.py`、`backend/CMakeLists.txt`、依赖构建和运行配置；仅新增头文件不足以成为可用后端。

## 2. 本机真实密文验证

环境：`CompilerforFHE` Conda 环境，NVIDIA L20（compute capability 8.9，46068 MiB），驱动 `580.95.05`，CUDA `13.4` / nvcc `V13.4.59`。上游 README 未列出 L20，本次用本机实际构建和执行确认这个组合可运行。

先独立编译上游 `libPhantom.so`，再构建 [实验程序](../tests/experimental/phantom/main.cu) 与 [打包 runtime](../tests/experimental/phantom/phantom_probe_runtime.hpp)。三个 evaluator fixture 来自现有 `test_periodic_slot_wraparound_with_repeated_encryption` 测试产生的 `model.hpp`，只替换 runtime include 和 namespace；其运算仍由现有编译器生成。这些 fixture 是固定样例，并非新增的 Phantom codegen。

参数为 `N=8192`、4096 个物理槽、逻辑周期 8、scale `2^40`、模数位宽 `{60,40,60}`、一个 special modulus。探针显式检查总位宽 160 不超过上游 tc128 参数表对应的上限；这项检查不代表完整的密码安全审计。

明文参考使用独立的稠密矩阵循环计算 `A*x+b`：

```text
A = [[    0,  .5, 0,   -.25, 2],
     [  -.5,   0, 0,      0, 0],
     [    0, .25, 0,      0, 0],
     [   -2,   0, 0,      0, 0],
     [  .75,   0, 0,  -.125, 0]]
b = [.0625, -.125, .25, -.375, .5]
x = e0、e4、[.875, -.375, .25, -.625, .5]
```

输入补齐到 8 个值后，周期重复到全部物理槽；解密检查全部 4096 槽，包括补零位置和周期重复，绝对误差阈值为 `1e-5`。最后一次运行结果见 [完整日志](../tests/experimental/phantom/results/validation.log)：

| 检查 | 次数 | 最大绝对误差 | 额外检查 |
| --- | ---: | ---: | --- |
| diagonal MatVec | 3 | `1.994e-6` | 每次 5 次旋转、5 次明文乘法 |
| BSGS MatVec | 3 | `9.306e-7` | 每次 3 次旋转、5 次明文乘法 |
| auto MatVec | 3 | `9.450e-7` | 每次 3 次旋转、5 次明文乘法 |
| 初始层公钥零值 | 1 | `1.040e-7` | 序列化后解密 |
| 密文平方、重线性化、rescale | 1 | `1.814e-7` | 与逐元素平方对照 |
| 降层后的公钥零值 | 1 | `1.080e-7` | 初始层加密零后降层 |

**12 项数值检查全部通过**，独立 CTest 为 `1/1 passed`。每次运行重新生成密钥和加密随机数，因此误差会变化；表中是本次复跑结果。公钥、旋转密钥、重线性化密钥及输入/输出密文经过内存流 `save/load` 往返，后续计算仍正确。

实验 runtime 不持有私钥，但测试程序在同一进程中完成密钥生成和解密，尚未验证跨进程客户端/服务端隔离。没有开展公平的 SEAL CPU / Phantom GPU 性能对比；日志中的 `0.43 s` 是整个探针测试时间，不能作为 MatVec 性能或加速比。

## 3. 已发现的适配问题

### 较低层级的公钥零值

第一版适配器直接在乘法 rescale 后的 chain index 2 编码零，再调用 `encrypt_asymmetric(context, plain)`，解密误差约 `2.959e7`，测试失败。[失败日志](../tests/experimental/phantom/results/probe-direct-lower-zero-failed.log) 已保留，不能将这条直接调用路径视为已验证可用。

当前探针改为：

```cpp
auto result = publicKey.encrypt_zero_asymmetric(context);
result.set_scale(value.scale());
if (result.chain_index() != value.chain_index())
  phantom::mod_switch_to_inplace(context, result, value.chain_index());
```

此方式已通过初始层和降层后的零值检查，且不需要私钥。源码 `src/secretkey.cu` 的该路径存在固定使用 context data 1 的 moddown，以及更新公钥 chain index 的行为，值得进一步核查；本次尚未定位上游缺陷根因，也没有修改上游实现。正式 runtime 的常量物化也应优先采用初始层加密后降层，补充多层测试。

### 层级、布局和执行方式

- Phantom 的 chain index 随降层增加；不能照搬 SEAL 按 chain index 比较的方向。正式 runtime 应明确对齐到较低的共同模数层，并验证 scale。
- `set_scale` 只是改元数据，不是 rescale。探针在实际乘法/rescale 后使用接近 `2^40` 的约定；更深乘法链需要参数规划和误差验证。
- 保留现有周期槽布局。仅把逻辑输入放到最前面的槽会破坏跨周期旋转；探针已覆盖首尾基向量。
- 上游 `hoisting_inplace` 返回的是若干旋转的累加结果，不是所有独立旋转密文。不能直接拿它替换每条对角线权重不同的 MatVec 旋转。
- `PhantomCiphertext` 拷贝会产生 GPU 内存分配和设备拷贝。正式 runtime 需要控制临时值生命周期，缓存公开权重的编码结果。
- GPU 运算异步执行；计时使用 CUDA event 或显式同步。编码过程本身也有 CPU/GPU 数据传输，不能假设整个流程完全驻留 GPU。
- Phantom 与 SEAL 的密文/密钥二进制格式不能互换。首版应在 Phantom 内完成密钥生成、加密、求值、解密；不规划 SEAL 密文直接迁入 GPU。
- 上游提供 `save/load` 不等于已经具备可部署的网络协议。正式 runner 还需加入后端/版本/参数标识、长度边界和资源校验。

## 4. 分阶段接入计划

| 阶段 | 开发项 | 完成标准 | 当前状态 |
| --- | --- | --- | --- |
| P0 | 固定上游 revision，构建库，复用现有生成代码验证 GPU 语义 | 真实密文结果与独立参考一致，记录限制 | 本次完成 |
| P1 | 新增可选 `phantom-ckks`，依赖配置与 CUDA CMake；实现 `backend/phantom/phantom_runtime.hpp`、`runner.cu` | 纯 CKKS 常量/add/sub/mul、层级对齐、公开零值可运行 | 待开发 |
| P2 | 接入 codegen/CLI/pipeline 和能力检查，复用现有 MatVec pass | `--backend=phantom-ckks --packing=auto` 可跑；含 logic 的模型明确拒绝 | 待开发 |
| P3 | 完成独立客户端/服务端阶段和密钥/密文序列化 | 服务端无私钥，跨进程重载参数、密钥、输入并正确求值 | 待开发 |
| P4 | 扩展回归与失败路径 | 矩形、稠密、稀疏、全零、偏置、周期边界、多层、缺旋转密钥、深度超限 | 待开发 |
| P5 | GPU 性能与规划 | 同精度/参数比较 SEAL；分别记录冷启动、传输、编码、密钥生成、求值、显存；再调整 BSGS 代价模型 | 待开发 |

首版保留现有 CPU 后端；没有可用 GPU 时不自动把 Phantom 产物当作 SEAL 产物运行。精确跨方案非线性继续使用 OpenFHE，避免在尚无格式与转换协议时混接两套密文。现有 AI 后端生成流程可在后续将 Phantom API、参数契约和探针纳入验证，但它不能补出公开库缺失的 scheme switching。

## 5. 复现指令

以下在仓库根目录执行；不安装 `front/` Python 包。上游源码和构建产物放在被忽略的 `.crossfhe/research/`，实验源文件与两份结果日志保存在 `tests/experimental/phantom/`。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CompilerforFHE
export PYTHONPATH="$PWD:$PWD/front"

# 本次首次获取使用 clone --depth 1；下面额外 fetch 固定 revision，方便未来复现。
git clone --depth 1 https://github.com/encryptorion-lab/phantom-fhe.git \
  .crossfhe/research/phantom-fhe
git -C .crossfhe/research/phantom-fhe fetch --depth 1 origin \
  1f4a198443b3af77118e51f53d5b8f332154b875
git -C .crossfhe/research/phantom-fhe checkout --detach \
  1f4a198443b3af77118e51f53d5b8f332154b875

cmake -S .crossfhe/research/phantom-fhe \
  -B .crossfhe/research/phantom-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPHANTOM_ENABLE_EXAMPLE=OFF \
  -DPHANTOM_ENABLE_BENCH=OFF \
  -DPHANTOM_ENABLE_TEST=OFF \
  -DPHANTOM_ENABLE_PYTHON_BINDING=OFF
cmake --build .crossfhe/research/phantom-build --target Phantom --parallel 8

cmake -S tests/experimental/phantom \
  -B .crossfhe/research/phantom-probe-tracked-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPHANTOM_INCLUDE="$PWD/.crossfhe/research/phantom-fhe/include" \
  -DPHANTOM_LIBRARY="$PWD/.crossfhe/research/phantom-build/lib/libPhantom.so"
cmake --build .crossfhe/research/phantom-probe-tracked-build --parallel 4
ctest --test-dir .crossfhe/research/phantom-probe-tracked-build -V
```

已有该 checkout 时跳过 clone；不同 GPU 需要调整 `CMAKE_CUDA_ARCHITECTURES`，不同 CUDA 安装位置需要调整 nvcc 路径。构建只验证了上述 Linux/NVIDIA 环境，未验证其他平台和参数组合。
