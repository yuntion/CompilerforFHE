# FHE Backends

`backend/` 实现编译器生成代码调用的密码运算，并构建模型专用 runner。MLIR 方言、方案降级和代码生成在 `compiler/`；本目录不解析 ONNX、不定义优化 pass。

## 库适配

| 目录 | 后端名称 | 库与能力 |
| --- | --- | --- |
| `seal/` | `seal-ckks` | Microsoft SEAL 4.1.2，CKKS 算术及必要的 rescale、level alignment、relinearization |
| `openfhe/` | `openfhe-cross` | OpenFHE 1.4.2，CKKS 算术、CKKS/FHEW 转换、sign bootstrapping、掩码选择 |
| `phantom/` | `phantom-ckks` | 可选 PhantomFHE 2.1 CUDA CKKS 算术与周期布局 MatVec，实验性后端 |

每个目录包含 runtime header、runner 源文件（CPU 为 `.cpp`，Phantom 为 `.cu`）和 `CMakeLists.txt`。密码库完整源码属于外部依赖，由根 `scripts/bootstrap.py` 按 revision 获取到 `third_party/`，或者通过参数引用已有源码。

各库的密文类型不同。跨方案链路完整地在 OpenFHE 中执行；本版不将 SEAL 或 Phantom 密文直接传给 FHEW/TFHEpp。新增 TFHEpp 后端须实现自己的转换键和数值编码。

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

CPU 套件更新 SEAL 和 OpenFHE（含安装），验证秘密矩阵乘、分组/空洞卷积、公开输出与零乘法，以及 OpenFHE 算术、ReLU、ReLU-MLP、MaxPool 和转换调用计数。GPU 套件更新 Phantom，要求已完成 `--with-phantom` 配置。底层 pytest 可通过 `--run-fhe -m seal`、`--run-fhe -m openfhe` 或 `--run-fhe --run-gpu -m gpu` 分别选择后端；底层命令需要先自行构建。默认安全配置须另外运行 demo 测量；回归测试的 OpenFHE 使用显式 `INSECURE_TOY`。详见 [测试说明](../tests/README.md)。

数值范围、精度、安全参数、评估键和转换延迟的完整契约见 [BACKENDS.md](../docs/BACKENDS.md)。Phantom 的依赖、四阶段执行、数值约定与 GPU 测试见 [Phantom 后端](../docs/PHANTOM_BACKEND.md)。
