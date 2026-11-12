# 编译器命令入口

`tools/` 存放可执行程序的入口。方言、pass 和 FHE 代码生成的实现属于 `compiler/`，密码库执行属于 `backend/`。完整 Ubuntu 构建、模型接入和测试步骤在根 [README.md](../README.md)。

## 当前 FHE 工具

| 入口 | 链接的主要库 | 处理过程 |
| --- | --- | --- |
| `crossfhe-opt.cpp` | `CrossFHEPipelines`、`MLIROptLib`、`MLIRParser` | 注册 `arith/func/scfhe/heir`，调用 MLIR `MlirOptMain` 解析 CLI、验证和执行 pass，输出 MLIR |
| `crossfhe-translate.cpp` | `CrossFHECodeGen`、`MLIRParser` | 读取最终 FHE MLIR，调用 `crossfhe::generateCpp`，输出库后端的模型 C++ 头文件 |

两者分别对应 MLIR 工具链常见的 `opt` 和 `translate` 职责。优化和 lowering 保持在 IR 中，库调用代码在最后一个阶段生成。Python 驱动按顺序调用它们，并保存各阶段产物和日志。

在仓库根目录执行以下 Ubuntu/Bash 命令：

```bash
source .venv/bin/activate
cmake --build build --target crossfhe-opt crossfhe-translate --parallel 4
build/bin/crossfhe-opt --show-dialects
build/bin/crossfhe-opt --help
build/bin/crossfhe-translate --help
ctest --test-dir build --output-on-failure
```

新增方言通常需要修改 `crossfhe-opt.cpp` 的 registry；新增 pass 通过 `compiler/Pipelines/FHE/Passes.h` 和 `registerCrossFHEPasses` 接入。扩展生成规则修改 `compiler/CodeGen/FHE/CodeGen.cpp`。具体教程见 [新建方言](../docs/ADDING_DIALECT.md) 和 [新增 pass](../docs/ADDING_PASS.md)。

## 原 NISL 入口

| 文件/函数 | 当前源码行为 | 与 FHE 路径的关系 |
| --- | --- | --- |
| `nisl-opt-main.cc` | 调用 `mlir::compiler::nislOptRunMain` | 原入口壳可以复用 |
| `compiler/Tools/nisl_opt_lib.cc` | 只打印 `nisl-opt-runMain!` 并返回 `1` | 尚未实现解析、方言注册、pass 执行或输出 |
| `nisl-compile-main.cc` | 调用 `mlir::compiler::nislRunMain` | 原编译 driver 入口 |
| `compiler/Tools/nisl_compile_lib.cc` | 注册旧 dialect/pass，调用 `bulidNISLTransformPassPipeline` | 当前走 TOSA/CPU/GPU pipeline，未接入 FHE lowering/codegen |

因此，并非 `nisl-main` 这一入口形式不能使用，而是现有函数实现尚未接上 FHE。直接把 FHE 工具改名为 `nisl-opt` 也不会自动获得这些能力。

`tools/CMakeLists.txt` 已分开构建：默认 `CROSSFHE_ENABLE_LEGACY_COMPILER=OFF` 创建两个 `crossfhe-*` 目标并返回；设为 `ON` 才进入原 NISL 构建逻辑。两条路径使用不同的 build 目录，不混用 CMake cache。旧路径保留供原有工作继续使用，其构建未在本版验证。

如以后决定统一入口，可以把 FHE registry 和 `MlirOptMain` 调用迁入 `nislOptRunMain`，让选定的 main 调用同一个 driver 库；再同步 CMake、`scripts/bootstrap.py`、`front/crossfhe/pipeline.py` 和测试中的工具名。代码生成仍调用现有 `generateCpp`，不需要把实现复制进 main。本版保持分开的命令入口。
