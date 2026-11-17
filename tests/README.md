# Tests

新增或修改测试前，先阅读 [测试编写与生成约束](../docs/TEST_GENERATION_RULES.md)。人工、AI 和脚本生成的测试均适用；本文件说明如何运行测试。

推荐在已激活的 Python 测试环境中使用统一入口：

```bash
python scripts/run_tests.py                         # 默认 quick：增量构建 + CTest + 非加密测试
python scripts/run_tests.py --suite front           # 仅前端，不需要原生工具链或 PyTorch
python scripts/run_tests.py --suite compiler        # 编译器增量构建 + CTest + 编译器 Python 测试
python scripts/run_tests.py --suite cpu             # CTest + 全部 CPU 测试，含 SEAL/OpenFHE 真实加密
python scripts/run_tests.py --suite gpu             # CTest + Phantom GPU 真实加密测试
python scripts/run_tests.py --suite all             # 全部测试，要求 CPU 和 GPU 后端均已配置
python scripts/run_tests.py --suite scripts         # 测试基础设施回归，只需要 pytest
python scripts/run_tests.py -- -k matvec             # 筛选默认 quick 中的 Python 用例，CTest 仍完整运行
```

首次准备完整 Python 测试依赖：`python -m pip install -r requirements.txt`。仅前端可使用 `python -m pip install '.[dev]'`，不安装 PyTorch；仅测试脚本回归只需 pytest。除 `front` 和 `scripts` 外，先用 `python scripts/bootstrap.py --jobs 4` 构建和注册工具链；GPU / all 还需要使用 `--with-phantom` 完成 CUDA 后端配置。测试入口和仓库 pytest 配置均自动加载源码，无需额外 editable install 或手动设置 `PYTHONPATH`。测试入口不会自动安装依赖或调用 bootstrap。

已有工具链时直接运行测试入口即可。重新 bootstrap 会覆盖 `.crossfhe/toolchain.json`；GPU 环境须继续传入 `--with-phantom` 及对应路径、CUDA 参数，否则原有 Phantom 注册会被移除。

`--jobs 4` 控制增量构建并发；`--skip-build` 显式复用现有二进制，并记录到 `summary.json`。默认更新原生编译器；cpu / all 还更新 SEAL、OpenFHE 构建和安装；gpu / all 还更新 Phantom。`--config /path/to/toolchain.json` 可以指定配置，否则使用 `CROSSFHE_CONFIG` 或仓库 `.crossfhe/toolchain.json`。构建保留缓存中的编译器设置，并核对源码目录；源码不匹配时使用新的构建目录重新 bootstrap，避免在原缓存上切换源码。

bootstrap 会记录 `openfhe_build`。旧配置缺少此字段时，脚本仅在安装前缀的同级目录中查找缓存，并要求源码和安装前缀同时匹配且结果唯一；否则提示重新 bootstrap。预先安装的外部库没有本地构建目录时，可明确选择 `--skip-build`。

每次运行会在 `.crossfhe/test-runs/` 下新建独立目录，包含构建/测试日志、实际命令及退出码、JUnit XML、`summary.json` 和 pytest 用例产物。`--output-dir /path/to/reports` 可更改父目录；脚本只在新建的子目录中运行 pytest。命令始终在仓库根目录执行，可以从其他目录通过脚本绝对路径调用。报告时间使用 UTC+08:00。

| 文件 | 用途 |
| --- | --- |
| `summary.json` | 是否通过、测试数量、跳过数量、各步骤退出码与耗时 |
| `build*.log`、`install-openfhe.log` | 原生工具及后端构建、安装日志 |
| `ctest.log`、`pytest.log` | 测试输出 |
| `ctest-junit.xml`、`pytest-junit.xml` | JUnit 测试报告 |
| `*-command.json` | 实际命令、开始/结束时间与退出码 |
| `cases/` | pytest 用例生成的模型、构建和执行产物 |

构建失败、测试失败、未选中用例或出现跳过时，脚本返回非零退出码。CPU 通过 `gpu` 标记排除 GPU 测试；quick 通过 `fhe` 标记排除加密测试，这些排除不算跳过。显式选择 gpu / all 时，不会把缺少 Phantom 配置当作可接受的跳过。

`--` 后的参数传给 pytest，例如 `-- -k matvec -x` 或 `-- --durations=10`。套件的 `-m`、`--run-fhe`、`--run-gpu` 和报告路径参数由脚本管理，不能透传覆盖；`--collect-only` 也不能透传，因为统一入口要求实际执行测试。只查看收集结果时直接使用 pytest。

下面是直接调用底层工具的方式，所有命令在仓库根目录执行。直接 pytest / CTest 不做增量构建，须先自行构建。仅选择前端或脚本测试时不要求原生工具链。

| 目录 | 验证内容 | 是否执行密码运算 |
| --- | --- | --- |
| `front/` | ONNX 算子、类型、静态形状、限制与拒绝诊断 | 否 |
| `compiler/` | 原生工具注册、非法 MLIR、编译产物状态、导出 tracing 拒绝 | 否 |
| `compiler/inputs/` | 可直接交给 C++ 编译器的 MLIR fixture | CTest 不执行密码运算 |
| `backend/` | SEAL 上的密文边界和复合计算 | 是，需 `--run-fhe` |
| `e2e/` | 模型到生成代码、密文推理、精度验证、资源契约与真实加密 A/B 检查 | CPU 需 `--run-fhe`，GPU 还需 `--run-gpu` |
| `scripts/` | 测试筛选、依赖诊断、构建阻断、日志与报告契约 | 否，不需要工具链 |

```bash
ctest --test-dir build --output-on-failure
python -m pytest -q --basetemp .crossfhe/test-unit
python -m pytest --run-fhe -q --basetemp .crossfhe/test-cpu
python -m pytest --run-fhe --run-gpu -q --basetemp .crossfhe/test-all
python -m pytest --run-fhe -m seal -q                    # 仅 SEAL 加密测试
python -m pytest --run-fhe -m openfhe -q                 # 仅 OpenFHE 加密测试
python -m pytest --run-fhe --run-gpu -m gpu -q           # 仅 GPU 加密测试
python -m pytest tests/compiler -m 'not native and not torch' -q  # 不需要原生工具或 PyTorch
```

`--basetemp` 是 pytest 专用的可重新生成目录，不要指向自己的数据、模型或长期保存的密钥。各组命令使用不同目录，便于保留本轮报告。Python 默认只收集上表的五组测试；原仓库的 `test.py`、`tosa-mlir` 和 `gpu-module` 是旧示例，不属于 FHE 测试结果。

默认 pytest 排除未启用的加密和 GPU 用例，显示为 deselected；显式启用后的缺依赖统一为 setup 错误。检查在 `-k` / `-m` 筛选之后进行，多个测试文件共用 session 工具链 fixture，不再分别跳过。`native`、`torch`、`seal`、`openfhe`、`fhe`、`gpu` 对应真实依赖，开启严格标记检查。OpenFHE 回归测试使用 TOY 参数，安全配置与论文性能实验单独执行，详见 [VALIDATION.md](../docs/VALIDATION.md)。修复步骤与验证结果见根目录 [TESTING_REPAIR_PLAN.md](../TESTING_REPAIR_PLAN.md)。

新增专项测试见 `compiler/test_optimizations.py`、`compiler/test_ai_backend.py` 和 `e2e/test_optimization_execution.py`；使用方式和实测结果见 [优化开发报告](../docs/development/RESULTS.md)。
