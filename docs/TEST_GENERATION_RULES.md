# 测试编写与生成约束

本文约束本项目新增或修改的测试，包括人工编写、AI 生成和脚本生成的用例。生成测试前须遵循本文；运行方式见 [测试说明](tests/README.md)，现有流程及边界见 [修复方案](TESTING_REPAIR_PLAN.md)。本文是编写规则；标记、依赖检查和运行结果判断由现有 pytest 配置与测试入口执行，其余要求通过代码审查和实际验证落实。

## 1. 先确定验证目标，再生成代码

每个用例必须能说明：触发条件是什么、预期行为是什么、依据哪个功能契约判定结果。函数名、参数名和必要的注释应表达这些信息，不要求额外生成一套用例登记系统。

- 优先补充能复现具体缺陷、覆盖行为边界或验证功能契约的最小用例。缺陷回归在条件允许时验证修复前失败、修复后通过；无法验证旧版本时说明限制。
- 只修改文档、注释或其他低影响内容时，不强制增加测试；选择与改动风险相称的验证。
- 不为增加数量而复制同类测试、覆盖每个私有函数或重复实现待测算法。测试不能把待测函数的返回值直接作为自身期望值。
- 同类输入使用 `pytest.mark.parametrize`，参数应覆盖不同语义或边界；不堆积没有新覆盖的随机样本。
- 修改已有测试前先核对原契约。不得通过删除断言、绕开失败分支、放宽容差或改写基准结果来掩盖缺陷；契约确需改变时说明依据并同步文档。

## 2. 使用现有分层和框架

| 位置 | 生成内容 | 不应承担的职责 |
| --- | --- | --- |
| `tests/front/` | ONNX 算子、类型、形状、公开参数及拒绝诊断 | 原生构建和真实加密 |
| `tests/compiler/` | 方言、pass、资源契约、代码生成及编译器接口的正反例 | 没有声明依赖的密码库执行 |
| `tests/compiler/inputs/` | 小型、可读、可复用的 MLIR fixture | 模型 runner、密钥或运行报告 |
| `tests/backend/` | 后端运行时接口、密文边界与复合计算 | 用模拟结果替代密码库行为 |
| `tests/e2e/` | 模型编译、真实加密执行、精度和资源契约 | 把只生成代码报告为加密成功 |
| `tests/scripts/` | 测试入口、依赖诊断、用例筛选、构建阻断和报告契约 | 要求真实密码库或 GPU 才能验证脚本逻辑 |
| `tests/CMakeLists.txt` | 小型、快速的原生 CTest 检查 | 重复编排全部 Python / 加密测试 |

Python 测试采用 pytest，文件命名为 `test_*.py`，用例命名为 `test_<行为或条件>_<预期结果>`，或沿用同文件中同样清晰的命名。原生工具检查复用 CTest。不得另建一套测试框架、批处理入口或独立的全量测试脚本；统一编排由 `scripts/run_tests.py` 承担。

不要把新测试放入旧示例 `tests/test.py`、`tests/tosa-mlir/`、`tests/gpu-module/`，也不要为本项目测试修改 `third_party/` 上游用例。新增测试目录、标记或套件时，必须同步 `pyproject.toml`、测试入口和使用说明，确认实际能够收集到。

## 3. 按真实依赖标注测试

| 标记 | 必须使用的情况 |
| --- | --- |
| `native` | 调用原生编译器工具，或使用共享工具链 `config` fixture |
| `torch` | 使用 PyTorch 导出、模型执行或参考计算 |
| `fhe` | 需要显式开启的加密集成用例，包括后端执行前的参数预算拒绝检查 |
| `seal` | 成功编译模型 runner 或执行测试需要 SEAL 库 |
| `openfhe` | 成功编译模型 runner 或执行测试需要 OpenFHE 库 |
| `gpu` | 需要 Phantom CUDA 库和 GPU 环境 |

成功执行的 SEAL / OpenFHE 加密用例至少包含 `native`、`fhe` 和对应后端标记；Phantom 加密用例至少包含 `native`、`fhe`、`gpu`。用到 PyTorch 时再加 `torch`。纯 Python 测试不因位于 `tests/compiler/` 就添加 `native`；只验证原生代码生成器拒绝某个后端，不额外声明该密码库依赖。

混合后端参数化测试用 `pytest.param(..., marks=...)` 分别标注，不给全部参数统一加上所有后端标记。可选依赖的导入放在需要它的函数或 fixture 内，避免只收集前端用例就要求 PyTorch / CUDA。标记须在 `pyproject.toml` 注册，保留 `--strict-markers`。

默认 pytest 排除加密测试；`--run-fhe` 开启 CPU 加密测试，`--run-fhe --run-gpu` 额外开启 GPU。新增用例必须遵守这套选择规则，不得通过修改标记或 pytest 配置隐式开启昂贵测试。

## 4. 共用依赖检查，不在用例中准备环境

- 工具链复用 `tests/conftest.py` 的 session `config` fixture，公共检查复用 `scripts/test_support.py`。不得为新文件复制一份读取 `.crossfhe/toolchain.json`、搜索编译器或检查后端的代码。
- 缺少必需依赖由筛选后的统一检查报错。不得使用 `pytest.skip`、`pytest.importorskip`、`try/except: return` 或捕获错误后改为通过，来掩盖所选用例缺失的工具链。
- `tests/scripts/` 为验证“跳过是否导致入口失败”等行为，可以在隔离的临时测试项目中构造 skip / fail；这些是受测输入，不能让外层回归用例无条件通过。
- 测试不得自动安装包、联网下载依赖、运行 bootstrap、切换 Conda、改写真实工具链配置或修改上游 checkout。首次准备由调用者完成，增量构建由统一入口完成。
- 工具位置从配置取得，不硬编码开发机路径、Conda 环境路径或 GPU 编号。调用原生工具时处理平台可执行文件后缀，复用 `crossfhe.pipeline.command`；脚本测试可直接使用标准库 subprocess 验证退出码与输出。

## 5. 断言必须验证真实行为

正例验证可观察结果，例如输出数值与形状、IR 中关键语义、资源契约或进程退出码。不能仅凭文件存在、命令被调用、日志出现 PASS 就宣称功能正确。测试目标本来就是产物创建、调度或报告时，可以断言对应文件、命令和状态，但应同时核对其关键内容或失败行为。

反例应构造明确的非法条件，使用具体异常类型和稳定的诊断片段；执行子进程时检查非零退出码及目标诊断。不得用 `raises(Exception)` 接受任意错误，也不得把“缺依赖”误当成“非法模型被正确拒绝”。有未完成产物或不可运行状态契约时，同时核对其状态。

编译器测试优先检查关键操作、类型、属性和资源关系，不对完整 MLIR 文本、临时路径、SSA 编号或 CMake 日志做脆弱的全文匹配。优化测试应同时验证语义保持和目标资源变化；只有契约保证精确数量时，才断言固定计数。

Mock / monkeypatch 可用于脚本调度、配置失败和报告处理等隔离测试。不得 mock 待验证的原生 pass、模型 runner、密码库执行或解密输出后，声称通过真实编译或真实加密验证。临时诊断探针可以辅助排查，但不能代替正式回归用例或计入独立测试数量。

## 6. FHE 用例的额外要求

成功路径必须实际完成模型编译、密钥生成、加密、密文求值和解密，调用现有 `compile_onnx` / `compile_torch`、`run_encrypted` 或对应真实后端入口。若用例目标是在执行前拒绝非法模型或超预算参数，应明确写成拒绝测试，不宣称完成加密推理。

- 参考结果来自独立的 NumPy / PyTorch 计算、公开公式或人工可核对的小型固定值，不从待比较的密文输出反推。
- 明确输入 dtype、shape、数值范围和比较边界；采用小型模型，控制编译、内存、密钥和执行成本。
- 指定有依据的 `atol` / `rtol`，检查输出形状和全部目标输出。近似 ReLU 按其近似误差预算验证，不沿用普通 CKKS 算术阈值。
- 精度、资源计数和执行报告应相互对应。涉及客户端/服务端隔离的成功路径，核对服务端无私钥的执行结果。
- 随机输入固定局部随机种子；不依赖测试顺序或其他用例留下的模型、密钥和密文。只有验证随机性相关缺陷时才增加多轮独立密钥执行，并说明轮数依据。
- 不用“失败后重跑直到通过”消除间歇性错误，不通过降低安全参数、修改模数链或放宽阈值隐藏失败。OpenFHE 的 TOY 回归必须显式标记其配置，不将其结果描述为默认安全参数验证或性能结论。

## 7. 产物和进程必须隔离

所有动态生成的 ONNX、MLIR、C++、runner、日志、密钥和密文使用 pytest `tmp_path` / `tmp_path_factory`。可复用的 fixture 源码可提交到测试目录；运行产物不得写入源码目录或覆盖已有模型、报告和工具链。

环境变量和全局对象的修改使用 pytest `monkeypatch` 或隔离子进程，测试结束后恢复。禁止删除或清空用户目录；`--basetemp` 只由测试入口分配到新建运行目录，测试不得覆盖该设置。手动调用 pytest 时也只能指定可重新生成的测试目录。

子进程须等待结束、检查退出码并保留有助于诊断的输出，使用明确且合理的超时；不把进程已启动当作运行成功，不用固定 sleep 代替完成信号。测试不要断言机器相关的绝对耗时或加速比，性能实验另行记录环境和方法。

## 8. 最小标记示例

下面仅演示依赖标记和共享 fixture，不代表每项检查都要另建文件；优先扩展已有合适用例。

```python
import os
from pathlib import Path

import pytest
from crossfhe.pipeline import command


@pytest.mark.native
def test_undefined_mlir_value_is_rejected(tmp_path, config):
    source = tmp_path / "invalid.mlir"
    source.write_text(
        "module { func.func @model() -> f64 { return %missing : f64 } }",
        encoding="utf-8",
    )
    suffix = ".exe" if os.name == "nt" else ""
    tool = Path(config["native_build"]) / "bin" / ("crossfhe-opt" + suffix)
    with pytest.raises(RuntimeError, match="undeclared SSA value"):
        command([tool, source, "--ppml-to-scfhe"], config)
```

## 9. 生成后的验收与结果陈述

先确认新用例可被收集、标记正确，再根据影响范围运行最小充分验证：

| 修改范围 | 首选验证 |
| --- | --- |
| 前端行为 | `python scripts/run_tests.py --suite front` |
| 方言、pass、原生诊断 | `python scripts/run_tests.py --suite compiler`，必要时增加受影响后端的加密用例 |
| SEAL / OpenFHE 运行时、生成代码或加密链路 | `python scripts/run_tests.py --suite cpu -- -k '<受影响用例>'`，影响广泛时运行完整 CPU 套件 |
| Phantom 运行时、CUDA 生成代码 | `python scripts/run_tests.py --suite gpu` |
| 测试入口、依赖检查、标记选择和报告 | `python scripts/run_tests.py --suite scripts`；同时改动真实用例调度时验证受影响套件 |

`--collect-only` 仅用于直接 pytest 检查收集，不能作为用例已通过的证据。直接 pytest / CTest 不触发构建；修改原生实现后使用统一入口更新程序，不能用旧二进制证明新实现正确。

交付说明必须区分已生成、已收集、已执行、已通过和未验证，给出实际执行命令、结果及必要的日志位置。缺环境而未执行时如实写明，不伪造 PASS、JUnit 或测量数据，不把 deselected 算作通过，也不重复累加复跑数量。

所选正式测试出现失败、空选择或跳过时，不得将本次验证报告为完整通过。测试数量变化来自真实用例增删，不能修改历史报告或硬编码现有测试总数来迎合预期。新增规则若需要机器强制执行，应另行修改公共检查并验证；仅补充本文不会自动实现新的检查机制。
