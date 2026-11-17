# Phantom CUDA CKKS 后端

`phantom-ckks` 是可选的实验性 GPU 后端，使用固定 revision `1f4a198443b3af77118e51f53d5b8f332154b875` 的 PhantomFHE 2.1。它沿用现有 MLIR 降级和 C++ evaluator 生成流程，支持标量 CKKS 算术及 `--packing diagonal|bsgs|auto` 的单个公开仿射 MatVec。

## 构建与运行

需要 NVIDIA GPU、兼容的 CUDA toolkit、nvcc 和 CUDA 支持的主机 C++ 编译器。普通 CPU 构建不启用 CUDA；Phantom 的主体许可证为 GPLv3，发布链接产物时需要考虑适用的许可义务。

在已配置的 `CompilerforFHE` Conda 环境及仓库根目录中执行。下面的架构 `89` 和 CUDA 路径对应本机 L20，其他机器应调整：

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PYTHONPATH="$PWD:$PWD/front${PYTHONPATH:+:$PYTHONPATH}"
python scripts/bootstrap.py --enable-cuda \
  --cuda-compiler /usr/local/cuda-13.4/bin/nvcc --cuda-architectures 89 --jobs 8

python -m crossfhe demo affine --backend phantom-ckks --packing auto \
  --output artifacts/phantom-affine --atol 1e-5 --rtol 0
python -m crossfhe demo square-mlp --backend phantom-ckks \
  --output artifacts/phantom-square --atol 1e-4 --rtol 1e-5
```

已有依赖库和编译器时可以使用 `bootstrap.py --skip-build --enable-cuda` 注册现有构建；通过 `--phantom-build` 指定含 `lib/libPhantom.so` 的目录。`--skip-build` 不会更新旧编译器，代码生成器必须先包含本次后端改动。

也可以从 MLIR 生成头文件，并独立编译 runner：

```bash
build/bin/crossfhe-translate model.mlir --backend=phantom-ckks -o model.hpp
cmake -S backend -B artifacts/phantom-runner -G Ninja \
  -DCROSSFHE_BACKEND=phantom-ckks \
  -DCROSSFHE_MODEL_HEADER="$PWD/model.hpp" \
  -DCROSSFHE_PHANTOM_SOURCE="$PWD/third_party/phantom-fhe" \
  -DCROSSFHE_PHANTOM_LIBRARY="$PWD/.crossfhe/phantom-build/lib/libPhantom.so" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build artifacts/phantom-runner --parallel 2
```

## 数值与产物约定

- scale 为 `2^40`，模数位宽为 `[60] + [40] * max(1, depth) + [60]`，一个 special modulus。环维度按上游 `tc128` 总位宽表和槽数需求选择，最大为 32768；超过预算时报错。
- 密文乘法执行重线性化及 rescale，密明乘法执行 rescale。加减与乘法先降到共同层级；Phantom 的 chain index 随降层增加。
- 标量在全部物理槽重复；打包输入补齐到逻辑周期后重复到全部物理槽，保留现有跨周期旋转语义。输出只提取逻辑结果。
- 公开常量在初始层加密。零值在初始层公钥加密后降层，避开已在探针中发现的低层级直接公钥加密问题。
- 客户端解码使用适配层的精确整数 CRT 重构：CUDA 逆 NTT 后将 RNS 系数复制到主机，调用上游主机 CRT，完成带借位的整数符号恢复后再转换为浮点数，最后执行 CUDA FFT。此路径绕开固定版本 GPU CRT 解码在零值和带符号常量上出现的间歇性异常；增加了主机计算和设备传输，计入 decrypt 阶段，服务端同态求值仍在 GPU 执行。
- runner 提供 `keygen / encrypt / evaluate / decrypt` 四个独立进程阶段，参数顺序同 SEAL runner。私钥只写入客户端目录，evaluate 只加载服务端参数、公钥、重线性化密钥和所需旋转密钥。
- `.phantom` 密钥/参数和密文容器包含后端格式版本、上游 revision、模型头文件 SHA256 和产物种类；参数与模型契约匹配。密文加载前检查数量、维度和剩余字节数。原生密钥 payload 仍使用上游研究序列化，仅用于可信本地文件，未实现网络协议、密钥集合认证或完整恶意输入防护。
- evaluator 计时前后同步 CUDA，包含运行库的编码、内存分配和设备操作，不包含加载密钥。尚无公平的 SEAL CPU / Phantom GPU 加速比结论。

## 能力边界与验证

公开库不提供 CKKS bootstrapping 或可接入的 CKKS↔FHEW switching；比较、精确 ReLU、Where、MaxPool 等逻辑模型明确拒绝，应使用 `openfhe-cross`。标量模式可显式选择现有有界多项式 ReLU 近似；其区间和误差契约保持不变。打包模式仍只支持单个公开仿射 MatVec，不支持通用网络布局。不会在无 GPU 时自动切换到 SEAL，也不能混用两套库的密文。

```bash
python scripts/run_tests.py --suite gpu
python scripts/run_tests.py --suite compiler
```

GPU 测试覆盖编译器生成的 diagonal/BSGS/auto、首尾槽基向量及重复加密、矩形和全零矩阵、多层标量运算、公开常量和低层级零值，以及缺旋转密钥、错误容器、截断密文和逻辑模型拒绝。测试执行四个独立进程并核对 NumPy/PyTorch 参考与运行时资源计数。直接 pytest 时使用 `--run-fhe --run-gpu -m gpu`，显式启用后缺少 Phantom 会报错；只加 `--run-fhe` 开启 CPU 测试。统一入口会验证 Phantom 构建缓存的源码目录是否与配置一致。首次隔离探针的结果和适配依据见 [可行性评估](PHANTOM_FHE_ASSESSMENT.md)。

## 本机接入验证（2026-11-16）

环境为 NVIDIA L20、CUDA 13.4、GCC 13.3，编译器使用固定 LLVM 17.0.6（`6009708b4367171ccdbf4b5905cb6a803753fe18`）。重建了默认 `build/bin` 并通过 `bootstrap.py --skip-build --enable-cuda` 注册本机现有 Phantom 库；完整测试使用默认 `.crossfhe/toolchain.json`。

| 验证 | 结果 |
| --- | --- |
| `tests/e2e/test_phantom_execution.py --run-fhe` | 8 passed |
| `tests/compiler tests/front` | 61 passed |
| 默认构建的 CTest | 7/7 passed |
| 现有 SEAL square-MLP、OpenFHE affine 真实加密回归 | 2 passed |
| CLI affine，`phantom-ckks --packing auto`，深度 1 | 最大绝对误差 `7.780e-7` |
| CLI square-MLP，`phantom-ckks`，深度 3 | 最大绝对误差 `1.196e-6` |

两条 CLI 的产物位于 `artifacts/phantom-affine/` 和 `artifacts/phantom-square/`，各自的 `latest_report.json` 保存阶段报告与验证结果。运行会重新生成密钥和加密随机数，误差随运行变化。测试日志及 JUnit 位于本机忽略目录 `.crossfhe/phantom-final-tests.log`、`.crossfhe/phantom-final-junit.xml`、`.crossfhe/phantom-final-compiler-junit.xml`、`.crossfhe/phantom-cpu-junit.xml`。这些结果验证了当前样例和参数组合，不是性能加速比或任意深度模型的精度保证。

2026-11-17 全量验证发现并修复了上述客户端解码异常，包含原失败密文重放和独立密钥重复验证，详见 [本次验证记录](VALIDATION-2026-11-17.md)。
