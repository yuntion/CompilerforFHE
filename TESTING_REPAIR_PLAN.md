# 测试流程修复方案

## 目标与现状

以 `python scripts/run_tests.py` 作为日常测试入口，保留可直接使用的 pytest / CTest；测试前准备、构建、用例选择和结果判断应有一致规则。

修复前，脚本只完成了基础编排：直接 pytest 仍可能找不到源码；工具链缺失时混用跳过和报错；只有 `fhe/gpu` 标记，无法准确描述原生工具和 CPU 后端依赖；OpenFHE 没有随测试增量更新。下表记录当前状态，不重写编译器或密码库实现。

## 当前修复状态

**原方案范围内的修复和验收均已完成，没有尚未完成的修复项。** “已修复”适用的入口和保留边界如下，不能理解为所有底层命令都会自动准备环境和构建。

| 原问题 | 状态 | 已落实内容与实现位置 | 验证依据 |
| --- | --- | --- | --- |
| 入口分散 | **已修复** | [run_tests.py](scripts/run_tests.py) 提供七种套件；根 README、模块 README 和 [测试说明](tests/README.md) 统一推荐该入口 | quick、front、scripts、CPU、GPU 实跑通过；all 的构建编排由回归测试覆盖 |
| 环境准备容易遗漏 | **已修复（源码路径、检查和说明）** | [pyproject.toml](pyproject.toml) 配置 pytest 源码路径；[test_support.py](scripts/test_support.py) 检查 Python 依赖和工具链；README 改用 Conda 建环境、pip 安装依赖 | 清除 PYTHONPATH 后前端 15/15 通过；仓库外脚本调用通过；错误配置有回归覆盖 |
| 可能测到旧程序 | **已修复（统一入口默认行为）** | 脚本先更新原生工具及所选后端；CPU 更新 OpenFHE 并安装；[bootstrap.py](scripts/bootstrap.py) 记录 OpenFHE 构建目录；显式 `--skip-build` 写入报告 | CPU 报告确认 build → build-openfhe → install-openfhe → CTest → pytest；GPU 确认先构建 Phantom；构建失败阻断有回归覆盖 |
| 依赖缺失的行为不一致 | **已修复** | [conftest.py](tests/conftest.py) 在筛选后统一检查依赖，共享 session 配置；移除原生测试和 Phantom 测试各自的跳过逻辑 | 原生、优化、GPU 三组缺配置均返回 setup 错误，退出码 4，无跳过 |
| 测试分类不足 | **已修复** | 补齐 native、torch、seal、openfhe 标记；保留 fhe、gpu；严格标记检查；用 `--run-fhe`、`--run-gpu` 显式启用加密用例 | SEAL 26、OpenFHE 5、GPU 8 项收集正确；完整 CPU / GPU 套件分别通过 |
| 日志和验证结果需手工整理 | **已修复** | 每次运行创建独立目录，自动保存命令、退出码、日志、JUnit 和 summary；空选择、失败、跳过均返回非零状态 | 新增测试覆盖报告隔离、参数透传和失败/跳过判定 |
| 本机 Phantom 构建缓存引用错误源码副本 | **已修复（代码校验与本机配置）** | 添加缓存来源校验；从 `third_party/phantom-fhe` 新建匹配构建，更新本机工具链，保留旧构建和配置备份 | 当前 `.crossfhe/phantom-build` 缓存来源与配置一致；8/8 GPU 测试通过 |

## 未修复项与保留边界

**原方案内未修复项：无。** 以下行为尚未实现，按原方案作为使用边界保留，不计作已修复功能：

| 行为 | 状态 | 当前使用方式 |
| --- | --- | --- |
| 直接 pytest / CTest 自动重编译原生程序 | **未实现，保留底层入口边界** | 先自行构建，或使用 `python scripts/run_tests.py`；显式 `--skip-build` 也不会保证二进制已更新 |
| 自动创建或激活 Conda 环境、安装 Python/系统依赖、首次 bootstrap | **未实现，保留首次准备步骤** | 按 README 准备环境并执行 bootstrap；测试入口检查缺失项并给出诊断 |
| 自动改写不匹配的构建缓存或替其他机器修复工具链配置 | **未实现，保留手动选择构建目录** | 校验失败时提前停止，使用匹配的新构建目录重新注册；本机 Phantom 配置已单独修复 |

验证边界：本轮实际执行完整 CPU 与 GPU 两组，合计覆盖全部 127 项 Python 测试；没有再用 `--suite all` 重复执行一次完整组合。实际验证环境为下文记录的 Linux / Conda 环境，不能据此宣称其他平台已实测通过。

## 实施顺序与验收标准

### 1. 统一入口和使用说明（已修复）

- [x] 保留 quick / front / compiler / cpu / gpu / all 六种套件，增加独立的 `--suite scripts` 测试基础设施回归测试入口。
- [x] 用标记选择 GPU 测试，避免依赖单个测试文件名；明确 pytest 参数透传和报告目录。
- [x] 根 README、各模块 README 和测试说明统一推荐脚本；底层命令注明用途和构建责任。
- 验收：从仓库外通过脚本绝对路径调用成功；每种套件有明确覆盖范围。

### 2. 修复环境准备容易遗漏（已修复）

- [x] 配置 pytest 源码搜索路径，仓库内直接运行 pytest 不要求 editable install 或手动设置 PYTHONPATH。
- [x] 提取脚本和 pytest 共用的工具链读取、路径规范化及依赖检查到 `scripts/test_support.py`，给出当前解释器和具体修复命令。
- [x] Python 依赖安装步骤使用 pip requirements，避免把 pip 版本清单交给 Conda。
- 验收：清除 PYTHONPATH 后前端测试仍通过；缺少配置、损坏 JSON、无效工具路径时给出明确诊断。

### 3. 防止测试使用旧程序（已修复，适用于统一入口默认构建）

- [x] 日常入口先更新原生工具；cpu / all 更新 SEAL 和 OpenFHE 构建及安装；gpu / all 更新 Phantom。
- [x] bootstrap 在工具链中记录 OpenFHE 构建目录；旧配置从安装前缀同级目录查找缓存，核对源码和安装前缀且要求结果唯一，不写死构建目录。
- [x] 检查构建缓存所属源码目录，阻止误用其他源码树的构建；记录是否显式跳过构建。
- 验收：构建步骤先于 CTest/pytest；构建失败立即停止；`--skip-build` 仅作为明确选择记录在报告中。
- 边界：直接 pytest / CTest 是底层入口，不触发构建；需先自行构建或使用统一脚本。

### 4. 统一依赖缺失行为（已修复）

- [x] pytest 在筛选后的用例集合上统一检查依赖，缺少必需工具链时提前报错，不再由各文件选择跳过或报错。
- [x] 共享 session 配置 fixture，移除重复的工具链读取和可选 Phantom 跳过。
- [x] 未开启的加密/GPU 用例作为 deselected 处理；显式开启后缺依赖属于错误。
- 验收：原生测试、优化测试、GPU 测试在缺少依赖时具有同一种失败规则；纯前端和脚本回归不需要密码库工具链。

### 5. 补齐测试分类（已修复）

- [x] 注册并标注 `native`、`torch`、`seal`、`openfhe`，保留 `fhe`、`gpu`；标记对应真实依赖，拒绝测试和纯前端测试不额外要求密码库。
- [x] 直接 pytest 默认运行非加密测试；`--run-fhe` 开启 CPU 加密测试，`--run-fhe --run-gpu` 额外开启 GPU。
- [x] pytest 使用严格标记检查，避免拼写错误造成漏选。
- 验收：CPU 套件排除 GPU，GPU 套件只选择 GPU；后端标记可独立选择 SEAL / OpenFHE；前端筛选不检查原生依赖。

### 6. 回归验证与结果记录（已完成）

- [x] 增加 27 项回归测试：依赖诊断、用例筛选、构建顺序/失败阻断、独立报告、参数透传和跳过判定。
- [x] 执行脚本自身回归、完整 quick、完整 CPU / GPU 测试，检查无意外跳过；记录实际结果与日志目录。
- [x] 将本文件逐项更新为完成，并补充实际验证结果和仍需明确的边界。

## 验证记录

方案已先写入，随后按上述顺序实施。使用 Conda `CompilerforFHE`、Python 3.12 和现有固定 revision 的工具链验证。

| 检查 | 实际结果 | 本机日志目录（`.crossfhe/test-runs/` 下） |
| --- | --- | --- |
| `--suite scripts` | 27/27 通过，零跳过 | `2026-11-17-scripts-9cn2ebdm` |
| 默认 quick | 7/7 CTest、88/88 Python 通过，39 项加密用例排除，零跳过；pytest 4.13 秒 | `2026-11-17-quick-ptybahpt` |
| `--suite gpu` | 7/7 CTest、8/8 GPU Python 通过，119 项其他用例排除，零跳过；pytest 90.29 秒 | `2026-11-17-gpu-dy6bz2_b` |
| `--suite cpu` | 7/7 CTest、119/119 Python 通过，8 项 GPU 用例排除，零跳过；pytest 395.84 秒 | `2026-11-17-cpu-3ml6qyql` |
| 仓库外绝对路径调用 front | 15/15 通过；清除 PYTHONPATH 并指定不存在的工具链配置 | `2026-11-17-front-ugjzs2p4` |
| 直接 pytest 的独立前端 / 纯 Python 编译器筛选 | 15/15、5/5 通过，均未配置原生工具链且清除了 PYTHONPATH | 终端验证 |
| 三类真实测试缺少工具链 | 原生、优化、GPU 均在用例执行前返回同一种 setup 错误，退出码 4，无跳过 | 终端验证及新增自动回归 |
| 后端标记收集 | SEAL 26、OpenFHE 5、GPU 8；共 127 项 Python 测试（原有 100 + 新增 27） | `pytest --collect-only` |

完整 CPU 与 GPU 两组互不重叠，合计 **127 项独立 Python 测试全部通过、零跳过**，包括新增的 27 项测试基础设施回归；原生 CTest 为 **7 项全部通过**。本轮按 CPU / GPU 分组执行完整覆盖，没有重复执行整套 all。

每个脚本运行目录包含命令、退出码、日志、JUnit 和 `summary.json`；日志和用例产物只保存在本机忽略目录中。quick、CPU、GPU 有重复覆盖，表中通过数不能直接相加作为独立测试总数。直接 pytest / CTest 仍由调用者负责先构建；自动增量更新属于统一脚本的职责。

本机另发现已有配置中的 Phantom 源码为 `third_party/phantom-fhe`，但旧构建缓存使用 `.crossfhe/research/phantom-fhe`。已保留旧构建和配置备份，从所配置的子模块新建 `.crossfhe/phantom-build` 并注册匹配的库路径。本机修改仅在忽略的 `.crossfhe/` 中，其他机器通过新的缓存校验获得明确诊断。
