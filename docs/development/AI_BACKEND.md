# AI 辅助后端：首版调度与验证闭环

开题报告要求 AI 限定在后端辅助，FHE 算法映射和专用 pass 由编译器完成。本版实现 `schedule-v1`：模型只返回现有 typed DAG 节点的串行排列，编译器检查数据/效果依赖、保持参数与 ABI、计算 last use 并生成 C++，随后真实构建和加密差分。

已提供离线响应回放和显式配置的 OpenAI 兼容模型接口。开发依据与后续研究建议见[代码与提示词评估](AI_BACKEND_ASSESSMENT.md)。

实现集中在 [backend/ai](../../backend/ai/README.md)（Python 包 `backend.ai`）：[task.py](../../backend/ai/task.py) 准备公开任务，[context.py](../../backend/ai/context.py) 读取函数映射，[protocol.py](../../backend/ai/protocol.py) 与 [prompts.py](../../backend/ai/prompts.py) 定义候选和消息，[driver.py](../../backend/ai/driver.py)、[provider.py](../../backend/ai/provider.py)、[validation.py](../../backend/ai/validation.py) 编排生成、构建与验证。统一 CLI 位于 `front/crossfhe`，`python -m crossfhe ai prepare|baseline|run` 用法不变。AI 实现直接依赖 `backend.toolchain` 与 `backend.execution`，函数映射资产继续位于 `backend/api/`；纯模型接入包位于 `front/crossfhe`。

## 本项目中的密码库函数映射

执行映射位于本仓库：[CodeGen.cpp](../../compiler/CodeGen/FHE/CodeGen.cpp) 将 FHE IR 生成 runtime 调用，`backend/<库>/<库>_runtime.hpp` 再调用固定版本密码库。[backend/api](../../backend/api/README.md) 提供可直接查阅的对应表，记录 **IR → runtime 签名 → `library_calls`**、适用约束和库 revision。

| 后端 | 当前映射 | 本版候选能力 | 本地函数映射 |
| --- | --- | --- | --- |
| `seal-ckks` | SEAL CKKS 标量算术、周期打包 MatVec | 合法独立节点重排、last-use 释放 | [seal.json](../../backend/api/seal.json) 和 [SEAL runtime](../../backend/seal/seal_runtime.hpp) |
| `openfhe-cross` | OpenFHE CKKS 和 CKKS↔FHEW sign/选择 | 保守保持原顺序，允许 last-use 释放 | [openfhe.json](../../backend/api/openfhe.json) 和 [OpenFHE runtime](../../backend/openfhe/openfhe_runtime.hpp) |
| `phantom-ckks` | Phantom CUDA CKKS 标量算术、周期打包 MatVec | 合法独立节点重排、last-use 释放 | [phantom.json](../../backend/api/phantom.json) 和 [Phantom runtime](../../backend/phantom/phantom_runtime.hpp) |
| 独立 TFHE | 尚未实现 | 拒绝作为编译目标 | OpenFHE/FHEW 使用 OpenFHE 的实现与映射 |

AI 默认读取这些本地映射。准备器将目标后端的完整映射写入 `api_context.json` 的 `function_mapping` 字段，并将该文件和实际 runtime 头文件一同封存。默认流程不需要 `/shizy/fhe-coder`，也无需复制密码库的完整上游文档。映射是现有实现的可读说明；执行行为由 CodeGen 和 runtime 决定，首版候选仍只能重排现有节点。

需要额外说明时，可通过 `--documentation-root` 显式读取 FHE-Coder 格式的 SEAL/OpenFHE 旧摘要。辅助检索条目带库名、来源文件 hash、签名、说明、截断标记和兼容性标记；未记录上游 revision 的资料仍标为 `unknown/auxiliary`，不能覆盖任务中的固定 runtime 与函数映射。Phantom 使用自身本地精确 API 映射，HEonGPU 是另一套库；独立 TFHE 不作为编译目标。显式指定的文档目录无效会报错。

## 快速运行

在仓库根目录、既有 `CompilerforFHE` 环境执行；修改原生生成器后先增量构建，不用旧二进制验证新实现。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CompilerforFHE
export PYTHONPATH="$PWD:$PWD/front${PYTHONPATH:+:$PYTHONPATH}"
cmake --build build --target crossfhe-opt crossfhe-translate --parallel 4

# 生成真正可执行的参考 artifact。也可使用 compile 导入自己的 ONNX。
python -m crossfhe demo affine --backend seal-ckks --packing auto \
  --output artifacts/ai-v1-reference

# 冻结精度、安全策略、公开输入范围、版本和最多 10 次总生成预算。
python -m crossfhe ai prepare artifacts/ai-v1-reference \
  --output artifacts/ai-v1-task \
  --input-bound 0.5 --atol 1e-4 --rtol 1e-5

# 确定性竞争基线，不请求模型。输出仍需接受独立验证。
python -m crossfhe ai baseline artifacts/ai-v1-task \
  --strategy depth-first --output artifacts/ai-v1-candidate.json

python -m crossfhe ai run artifacts/ai-v1-task \
  --artifact artifacts/ai-v1-reference --candidate artifacts/ai-v1-candidate.json \
  --output artifacts/ai-v1-run
```

目录/候选文件必须新建，准备器不会覆盖非空任务目录。任务准备统一使用 `python -m crossfhe ai prepare`。旧 artifact 缺少 DAG 或版本上下文时应重新编译。

上述准备命令已包含本地函数映射。如需追加已有的辅助摘要，可在准备命令中增加 `--documentation-root /path/to/fhe-coder`；该选项不是运行前提。

`baseline --strategy reference` 生成原顺序，仍由候选 codegen 插入 last-use 释放；`depth-first` 采用确定性拓扑启发式，不保证最小内存。两者用于离线打通流程和对照，不算 AI 生成结果。

### 调用模型

仅在显式传入 `--model` 和 `--api-base` 时发起请求。凭据从指定环境变量读取；不写进任务包、messages 或报告。

```bash
# 预先在环境中设置 OPENAI_API_KEY；命令中不填写密钥。
python -m crossfhe ai run artifacts/ai-v1-task \
  --artifact artifacts/ai-v1-reference --output artifacts/ai-v1-model-run \
  --model YOUR_MODEL --api-base https://api.openai.com/v1
```

可用 `--api-key-env` 指定凭据变量名、`--timeout` 指定 HTTP 超时。provider 使用标准库，无需新增模型 SDK。当前实现使用 chat/completions 文本接口；模型必须输出严格 JSON，围栏、额外文字、重复键、非有限数字和未知字段都会被拒绝。

修复队列可离线回放：把响应放入 `{"responses": [候选对象或原始响应字符串, ...]}` 文件，使用 `ai run --responses FILE`。`--candidate`、`--responses`、`--model` 互斥。

## 任务与候选协议

任务 schema 为 v2，包含 typed DAG、资源/ABI 契约、tensor mapping、近似策略、验证策略、库 revision，以及原生生成器、C++ 工具链、实际库文件和后端源码指纹。模型 ONNX hash、参考源码 hash、候选源码/binary hash 分别记录。

任务准备与加载共用 `protocol.validate_task()` 检查策略、资源契约、DAG 与 tensor mapping。本地 JSON 使用普通文件读取；模型 HTTP 响应的大小限制由 provider 在读取时执行。任务与上下文的 hash、候选调度合法性和真实加密差分仍在各自入口检查。

准备器白名单导出 `task.json`、`dag.json`、`contract.json`、`program.mlir`、`reference.hpp`、目标 runtime、`api_context.json`、`candidate.schema.json`、`PROMPT.md`。不导出输入明文、密钥、密文或可执行文件。

要查看模型收到的密码库映射，直接读取任务的 `api_context.json.function_mapping`；运行时的 `messages.json` 还包含固定 runtime 头文件。辅助摘要按需检索，本地函数映射完整保留。

模型返回：

```json
{
  "schema_version": 1,
  "status": "proposal",
  "mode": "schedule-v1",
  "task_hash": "<task.json 中的 hash>",
  "schedule": ["n0", "n2", "n1", "n3"],
  "rationale": ["先完成可提前释放中间密文的独立分支。"]
}
```

实际节点集合来自任务，示例顺序不适用于任意图。也可返回 `{"schema_version":1,"status":"no_candidate","task_hash":"...","reason":"..."}`。

Python 检查外层 schema、身份、节点全集和依赖；原生工具从原 MLIR 独立重建依赖并再次检查。随机加密/零值、方案转换等操作为保守效果屏障；OpenFHE 首版全部保持顺序。候选不能新增操作、改变操作数、重关联、删除 rescale、重选 BSGS、修改参数/布局或提交 C++。

原生命令也可独立调用：

```bash
build/bin/crossfhe-translate model.mlir --backend=seal-ckks --emit-dag -o dag.json
build/bin/crossfhe-translate model.mlir --backend=seal-ckks \
  --schedule=candidate.json -o candidate.hpp
```

默认不传 `--schedule` 的代码生成保持原有行为。候选代码生成在保持 ABI 的基础上释放最后一次使用后的中间密文；这不等于已经测得 RSS/显存降低。

## 真实验证、反馈与报告

1. 核对封存任务、原 artifact 的完整 hash、版本/实际库文件和 ABI/近似元数据；重新生成 DAG 与参考 C++，拒绝不一致上下文。
2. 只将严格校验的排列交给原生生成器，生成 `candidate.hpp` 后执行真正的 CMake/C++ 构建，保留命令日志、失败阶段与 hash。
3. 可信参考 runner 负责 keygen/encrypt/decrypt，候选 runner 只执行 evaluate。两者在同一后端内使用同一密钥集合、实际参数和同一份加密输入；SEAL/OpenFHE/Phantom 容器不能跨库互换。
4. 同时比较候选/冻结 FHE 参考、候选/独立 ONNX ReferenceEvaluator、FHE 参考/明文，检查全部输出的形状、有限性、绝对/相对误差以及操作计数。
5. 默认使用公开范围内的零、正负边界、交替与固定 seed 随机输入。OpenFHE 逻辑路径避开默认的零输入用例，但不能保证任意内部比较的 margin；使用者仍需校准 comparison scale 和应用输入范围。
6. 数值通过后，交替执行参考/候选，记录预热、重复样本、median/p95 evaluator 时间及进程墙钟时间。当前比较对象为原 codegen；候选同时包含确定性的生命周期管理，计时差异不能单独归因于 AI。未测量峰值 RSS 或 GPU 显存，不报告已证明的内存收益或加速比。

Phantom 保留容器身份检查：调度候选构建使用可信参考头的 `CROSSFHE_CONTRACT_HEADER` 计算计算/参数身份，候选自身源码和 binary hash 另记。默认普通构建仍以自身模型头为身份。只有已经验证保持原图的调度候选才使用共享契约路径。

每轮保存 `messages.json`、原始响应、候选、源码、构建日志、差分结果和 timing。修复反馈带上一版完整响应/候选、失败阶段和诊断；不包含客户端输入数组、解密结果或密钥。同一合法排列不会再次编译/生成新密钥重试，改变理由也不构成新候选。总生成次数最多 10 次，包含首轮；无候选、provider 错误或基础设施超时会终止，不能伪装成通过。

运行结果在 `RUN/report.json`：`validated` 表示本次功能与固定策略检查通过，`functional_only` 表示显式 TOY 功能检查，`rejected`/`infrastructure_error`/`provider_error` 分别记录对应失败。TOY 永远不能 `policy_passed`。本版仅归档供审阅，`promotion_allowed=false`、`selectable=false`，不覆盖原参考 artifact。

## 当前边界

本版仅执行可信原生生成器从受限 JSON 生成的代码，采用独立客户端/求值进程和目录；尚未提供运行任意 AI C++ 的操作系统沙箱。多线程、多 stream、buffer pool、任意 API 绑定、自动参数优化、bootstrap/rescale 重排和自动晋升均未开放。

有限差分测试不构成全输入的形式化证明或全应用安全证明。输入公开区间不保证所有内部比较合法，多项式局部近似界也不是网络总误差界；阈值由任务创建者明确给定，模型无权放宽。哈希核对用于内容完整性，不是来源签名。

测试复用现有 pytest/CTest 与统一入口；不请求在线模型即可验证协议、反馈、真实编译和 SEAL/OpenFHE/Phantom 加密链路：

```bash
python scripts/run_tests.py --suite all
```

本次实际运行与报告见 [AI 首版验证记录](AI_BACKEND_VALIDATION.md)。
