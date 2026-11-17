# CrossFHE 冗余文件与代码清理候选清单

生成时间：2026-11-17 17:35（GMT+8）
勘察对象：`root@10.15.0.9:30008:/shizy/CompilerforFHE` 工作区快照
勘察方式：只读（文件枚举、源码检索、行数/体积统计、MD5 重复检测、git 状态核对）

> **本清单只是建议，没有执行任何删除或修改。** 除新增本文件外，仓库内容保持原样。
> 当前工作区有约 30 个已修改（M）、5 个已删除（D）和若干未跟踪（??）的未提交改动，属于他人正在进行的工作。执行任何清理前，请先 `git status` 确认、提交或 stash，避免把别人的改动一起清掉。

## 0. 判读约定

| 等级 | 含义 | 典型动作 |
| --- | --- | --- |
| R0 | 零风险：未跟踪生成物、空文件、已删除文件 | 可直接删，不影响构建/测试 |
| R1 | 低风险：需保留锚点或同步改 1-2 处引用 | 改完静态核对即可 |
| R2 | 中风险：需同步改配置/测试/文档 | 改完需跑相应测试套件 |
| R3 | 高风险：涉及功能取舍、成片联动删除 | 先做产品决策，再整体下线 |

「删除」对 R3 条目指**整块下线**（含引用、测试、文档），不是直接 `rm`。体积与行数为实测值。

---

## 1. 空文件、占位文件、已删除但仍被 git 跟踪（R0）

| 路径 | 现状 | 删除原因 |
| --- | --- | --- |
| `compiler/Utils/CMakeLists.txt` | 0 字节 | 空文件。默认构建路径 `compiler/CMakeLists.txt:11-14` 只进入 `Dialect`、`Conversion/FHE`、`Pipelines/FHE`、`CodeGen/FHE`；`Utils` 仅在 legacy 分支 `:7` 被引用，而该分支默认关闭 |
| `compiler/Utils/OptionUtils.cpp` | 4 字节，无有效代码 | 同上；留着只会让人误以为这里有工具实现 |
| `bootstrap.log` | 0 字节，工作区已删除但仍在 index | `.gitignore` 已忽略 `*.log`；用 `git rm --cached` 让索引与工作区一致 |
| `tests/test.py` | 0 字节，已删除仍跟踪 | 空占位文件，无内容、无引用 |
| `tests/gpu-module/gpu-moudle.mlir` | 0 字节，已删除仍跟踪 | 空文件且文件名拼写错误（`moudle`） |
| `tests/tosa-mlir/add_llvm_cpu_codegen.ll`、`tests/tosa-mlir/tosa-add.mlir` | 0 字节，已删除仍跟踪 | 属于 legacy TOSA 路径（见第 6 节），内容为空 |
| `tests/gpu-module/`、`tests/tosa-mlir/` | 遗留空目录 | 文件已删，目录无内容 |
| `docs/validation-junit.xml`（4330 字节单行）、`docs/native-validation-junit.xml`（1950 字节单行） | 测试报告放在 `docs/` 根目录 | 是机器生成的 JUnit XML，不是文档；`docs/validation/` 已被 `.gitignore` 忽略，但放根目录反而被 git 跟踪。应删除或移入被忽略目录 |

`git ls-files --deleted` 当前正好列出上述 5 个已删除文件，可一次性从索引移除。

---

## 2. 生成产物与历史归档（R0，均不在 git 中）

| 路径 | 体积 | 内容与删除原因 |
| --- | --- | --- |
| `.crossfhe/` | **23 GB** | 历史测试运行、验证快照、AI 迁移包、源码备份。大头：`test-runs` 3.3G、`ai-v1-tests` 3.2G、`validation-2026-11-16` 3.0G、`validation-2026-11-17-094812` 2.9G、`test-matvec-all` 2.5G、`test-optimized-all` 2.4G、`source-backups` 1.7G、`research` 988M、`test-pass-ab` 943M、`cleanup-tests` 731M、`ai-v1-final-cases` 578M、`phantom-native` 210M。全部可由 bootstrap / 测试重跑再生 |
| `docs/validation/` | 8.5 MB / 1523 文件 | 其中 `2026-11-17-gpu-matvec/` 占 1494 文件，已被 `.gitignore:14 validation/` 忽略。顶层与 `raw/` 存在 22 组同名同内容文件；全项目 MD5 检测出 197 组完全重复文件，绝大多数在这里 |
| `build/` | 230 MB | CMake/Ninja 构建目录，已忽略 |
| `artifacts/` | 50 MB | demo 产物、密钥/密文、runner 构建目录，已忽略 |
| 129 个 `__pycache__/`、`.pytest_cache/` | 小 | Python 字节码与 pytest 缓存，已忽略 |

**必须保留（否则测试跑不起来）**：`.crossfhe/toolchain.json`，以及它直接引用的 `.crossfhe/openfhe-build/`、`.crossfhe/openfhe-install/`、`.crossfhe/phantom-build/`。`backend/toolchain.py:14`、`scripts/test_support.py:47`、`scripts/run_tests.py:140` 都以该 JSON 为默认配置。

**不建议删**：`articles/`（375 MB 文献 PDF，被 `docs/development/ARTICLES.md` 与 `literature-index.csv` 引用，是研究资料而非构建产物）。

---

## 3. 超大验证报告与重复内嵌的 transcript（R1）

| 路径 | 规模 | 删除原因 |
| --- | --- | --- |
| `docs/VALIDATION-2026-11-17.md` | **40953 行 / 1.56 MB** | 正文摘要只到约第 74 行，之后把整段工具调用 transcript、日志、MLIR、JSON 原样内嵌（21489 行缩进块 + 16898 行日志文本 + 2512 个代码围栏）。同一批内容另存为 `docs/validation/2026-11-17-gpu-matvec/tool-transcript.md`（435 KB）和 `tool-transcript.jsonl`（465 KB），等于三份拷贝；不可读且拖慢 git/编辑器 |
| `docs/validation/2026-11-17-gpu-matvec/tool-transcript.md` 与 `.jsonl` | 435 KB / 465 KB | 同一 transcript 的两种格式，留一份即可 |
| `docs/validation/2026-11-17-gpu-matvec/INDEX.md`（327 KB / 1461 文件 SHA256 清单）、`artifact-inventory.json`（1.35 MB / 3883 文件清单） | 1.7 MB | 逐文件归档索引，仅审计时需要；日常开发用不到 |

**注意**：`docs/PHANTOM_BACKEND.md:78` 和 `docs/validation/.../INDEX.md:3` 链接了 `VALIDATION-2026-11-17.md`。建议**瘦身**（保留文件、摘要与链接，删掉附录里的原始 transcript），而不是直接删文件，以免留下坏链。`docs/VALIDATION-2026-11-16.md`（338 行）体量合理，建议保留。

---

## 4. 对 scripts 的测试（用户点名，R2）

| 路径 | 规模 | 删除原因 |
| --- | --- | --- |
| `tests/scripts/test_testing_workflow.py` | 194 行 / 12 个用例 | 全部是**元测试**：测的是 `scripts/test_support.py` 与 `scripts/run_tests.py` 自身，而不是编译器、前端或后端的产品行为。具体测「配置错误信息文字」「CMakeCache 源目录不匹配」「pytest 标记筛选」「runner 生成 summary.json」「拦截用户传 `-m`」等。`isolated_pytest` fixture 起子进程跑 pytest、`mini_repo` 造临时仓库并 monkeypatch 内部函数（:77-99、:135-149），维护成本明显高于它保护的价值；runner 是内部开发工具，出错看日志即可 |
| `tests/backend/ai/test_ai_validation_failures.py` 等专测错误文案的用例 | 17 行等 | 与第 5 节的防御性校验配套，校验简化后这些用例自然失去意义 |

**删除时必须同步修改的引用**（否则 `pytest` 与 `--suite scripts` 报错）：

- `pyproject.toml:27` — `testpaths` 中的 `"tests/scripts"`
- `scripts/run_tests.py:23, 35, 39, 173, 187-188, 206` — `SUITES` 里的 `"scripts"` 及 6 处分支判断
- `docs/TEST_GENERATION_RULES.md:25, 55, 124` — 把 `tests/scripts/` 列为正式分层并规定其契约
- `tests/README.md`、`docs/development/AI_BACKEND_ASSESSMENT.md` 中对 `TEST_GENERATION_RULES.md` 的引用段落
- `tests/conftest.py` 保持不动（它依赖 `scripts.test_support`，不依赖该测试文件）

客观说明：`docs/TEST_GENERATION_RULES.md` 目前**明确要求**保留这一层，所以这不是「无主死代码」，而是规范层面的过度工程。删除等于同时简化规范，需团队确认。

---

## 5. 过多的防御性编程（用户点名，R2）

按证据密度排序（括号内为「raise / isinstance / 行数」实测）：

| 位置 | 实测 | 冗余表现 |
| --- | --- | --- |
| `backend/ai/protocol.py` | 78 / 36 / 425 行 | 对**自己生成的本地离线 JSON**做了防恶意输入的全套校验：`MAX_JSON_BYTES`、`MAX_CONTEXT_BYTES`、`MAX_NODE_ID_LENGTH`、`MAX_RATIONALE_ITEMS`（:20-25）、手写 JSON Schema（:43-78）、递归 `_json_values`（:81-95）、重复键检测（:116）、symlink/普通文件/大小检查（:134-152）、`parse_json` 与 `read_json` 各查一次 `max_bytes`（:134、:158）。同一约束被检查 2-3 遍 |
| `backend/ai/validation.py`、`context.py`、`provider.py`、`task.py` | 36/343、28/221、29/166、18/138 | 同一份 task/contract 在 prepare → verify → validate → driver 各阶段反复校验；4 个模块合计 111 处 raise |
| `scripts/test_support.py` | 11 / 4 / 130 行 | `require_file`、`config_path`、`executable`、`load_toolchain`（:14-58）四处几乎相同的 `isinstance(value, str) or not value` + 「run bootstrap.py first」长提示；`openfhe_build`（:107-130）为「旧配置」做 glob 搜索兼容，而当前 `toolchain.json` 已明写 `openfhe_build`，该分支实际不可达 |
| `scripts/run_tests.py` | 6 / 0 / 234 行 | `managed_options`（:151-156）拦截用户传 `-m/--basetemp/--junitxml`；`check_report`（:121-129）与 `finally`（:217-225）两处解析同一份 JUnit；`run_step`（:103-111）三层 KeyboardInterrupt 处理；`main`（:210-216）四类异常分支 |
| `compiler/CodeGen/FHE/CodeGen.cpp` | 482 行，30+ 处 `emitError` | `validateEvaluatorImpl` 在 :232、:332、:450 被调用 3 次；:279 又完整重写了 `PackedOps.cpp:8-10` 的「周期是 2 的幂且 ≤16384」检查 |
| `front/crossfhe/frontend.py`、`affine.py` | 40 / 401 行、21 / 110 行 | `import_onnx`（:398-399）与 `affine.py:18` 重复检查 ONNX opset 13..21；`affine.py` 的有限系数、维度检查随后又被 MLIR verifier 与 CodeGen 各查一遍 |

**跨层重复校验是最典型的一处**：同一份打包 IR 的「周期合法、系数有限、矩阵维度、偏置匹配」在 `front/crossfhe/affine.py` → `compiler/Dialect/HEIR/PackedOps.cpp:6-43` → `compiler/Pipelines/FHE/MatVec.cpp:24-45` → `compiler/CodeGen/FHE/CodeGen.cpp:228-316` 被检查 4 次。

建议保留 **MLIR verifier**（`PackedOps.cpp`，IR 正确性的边界，删了会让手写 `.mlir` 失去保护），收敛 Python 前端与 C++ CodeGen 里的重复检查；AI 后端用一个 `validate_task()` 统一入口替代分散校验。

---

## 6. Legacy 编译器死代码（R3，需决策）

`CMakeLists.txt:4` 的 `CROSSFHE_ENABLE_LEGACY_COMPILER` 默认 **OFF**，README:110 明写「旧仓库的 TOSA/CPU/GPU 文件仍保留在原目录……**本版未验证旧路径**」。默认构建只走 `compiler/CMakeLists.txt:11-14`，以下约 **2955 行** C++/CMake 从未被编译、也无人验证：

- `build-tools/cmake/` 全部 10 个 `.cmake`（含 `LegacyCompiler.cmake` 210 行、`nisl_llvm.cmake` 222 行）
- `compiler/CodeGen/CPU/`、`compiler/CodeGen/GPU/`（约 500 行）
- `compiler/Conversion/TOSA/`、`compiler/Conversion/ConvertToMlirLLVM/`（约 500 行）
- `compiler/Tools/`（约 800 行，含 `nisl_compile_lib.cc` 531 行）、`tools/nisl-compile-main.cc`、`tools/nisl-opt-main.cc`
- `compiler/Utils/`（第 1 节的空文件）、`tests/CMakeLists.txt:1-4`、`tools/CMakeLists.txt` legacy 分支、`tools/README.md:38`

佐证：`LLVMCPUCodeGenpass.cpp:115` 与 `LLVMGPUCodeGenpass.cpp:189` 写死 `/home/MLIR-Compiler/tests/tosa-mlir/`，本机没有该路径，旧路径即使打开也跑不通。

删除原因：近 3000 行不可构建、不可验证、与 FHE 主线无关的上游遗留代码，干扰阅读和构建配置。若需保留作历史参考，建议移到独立 tag/分支而非主目录。删除时须同步移除 `CROSSFHE_ENABLE_LEGACY_COMPILER` 开关及相关文档说明。

---

## 7. 兼容转发层与孤立文件（R1-R2）

| 路径 | 现状 | 删除原因 |
| --- | --- | --- |
| `front/crossfhe/packed.py` | 5 行 | 纯转发：`from backend.packed import import_packed_onnx`。生产路径 `front/crossfhe/pipeline.py:15` 已经直接 import `backend.packed`，唯一使用者是 `tests/compiler/test_matvec_pass.py:11`。把测试导入改成 `backend.packed` 后即可删 |
| `scripts/prepare_ai_backend.py` | 18 行 | 纯转发到 `crossfhe.cli.main(["ai", "prepare", ...])`；`backend/ai/README.md:22` 自己注明「仅转发」。可直接用 `python -m crossfhe ai prepare` |
| `examples/user_model.py` | 17 行 | 全仓库检索无任何引用（README、测试、脚本均未提及），是孤立示例 |
| `docs/MIGRATION.md` | 54 行 | 记录一次**已完成**的目录迁移历史（旧路径 → 新路径对照表），当前结构已稳定；唯一入链是 README 的文档地图。属过程文档，可归档 |

---

## 8. AI 后端整块（R3，需决策）

与核心链路（ONNX → MLIR → FHE 代码生成 → 加密执行）无关的实验性「AI 候选调度生成/验证」子系统，却贡献了全项目最高的校验密度和大量配套文件：

| 组成 | 规模 |
| --- | --- |
| `backend/ai/` 8 个模块 + `backend/api/` 3 个 JSON 与 README | 1699 行 Python |
| `tests/backend/ai/`（4 文件）、`tests/compiler/test_ai_backend.py`、`test_ai_schedule.py`、`tests/e2e/test_ai_backend_execution.py` | 1111 行测试 |
| `docs/development/AI_BACKEND.md`、`AI_BACKEND_ASSESSMENT.md`、`AI_BACKEND_VALIDATION.md`、`docs/development/validation/ai-prepare.log` | 3 份文档 |
| `.crossfhe/ai-*`、`artifacts/ai-v1-demo/` | 数 GB 归档（见第 2 节） |

接入点仅 `front/crossfhe/cli.py:37-38, 52-56` 的 `ai` 子命令；`pyproject.toml:24` 把 `backend.ai` 声明为分发包。

删除原因：若不再做 AI 候选生成研究，整块下线可同时消除第 5 节的主要冗余、7 个测试文件、3 份文档和数 GB 归档。风险是联动面大，须先确认该功能是否还在路线图内；若保留，则建议至少执行第 5 节的校验收敛。

---

## 9. 建议执行顺序

1. **第 1 节**：清空文件与已删除索引项（零风险，先让 git 状态干净）。
2. **第 2 节**：清理生成产物；**先备份 `.crossfhe/toolchain.json`**，并保留其引用的 3 个 build/install 目录。
3. **第 3 节**：瘦身超大报告、删重复 transcript 与归档索引（保留链接锚点）。
4. **第 7 节**：改 2 处 import / 1 处文档链接后删转发层。
5. **第 4 节**：团队确认后整体下线 `tests/scripts/`，同步改 `pyproject.toml`、`run_tests.py`、`TEST_GENERATION_RULES.md`。
6. **第 5 节**：按「保留 MLIR verifier、收敛前后端重复校验」逐个简化，每改一处跑对应套件。
7. **第 6 / 8 节**：最后做产品决策，整块处理 legacy 编译器与 AI 后端。

### 明确不建议删除

`third_party/`（5 个 gitlink 子模块，构建依赖）、`articles/`、`docs/development/literature-index.csv`、`docs/measurements/results.csv` 与 `results.json`（被 `RESULTS.md`、`VALIDATION.md` 等引用）、`.crossfhe/toolchain.json` 及其 3 个构建目录、`docs/ADDING_DIALECT.md` / `ADDING_PASS.md`（扩展教程）、`tests/compiler/inputs/`（CTest fixture）、`README.md`、`docs/VALIDATION-2026-11-16.md`。

---

## 10. 本次操作声明

- 只新增本清单文件（`docs/CLEANUP_CANDIDATES.md`），本地保留一份同名副本用于预览。
- 未删除、重命名或修改任何现有文件；未执行 `git add` / `git rm` / `commit`；未运行测试或构建。
- 清单中的体积、行数、引用位置均来自当前工作区实测；若期间有人继续改动项目，执行前请重新核对。
