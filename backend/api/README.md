# 密码库函数映射

本目录保存本项目维护的 **FHE IR → runtime 方法 → 真实密码库 API** 映射，是 AI 后端任务的默认函数资料。映射描述当前适配器已经实现的路径，不是密码库的完整功能列表；它不会授予调度候选新增 API、修改算法或改变参数的权限。

AI 实现位于 [backend/ai](../ai/README.md)（Python 包 `backend.ai`），由 [context.py](../ai/context.py) 读取本目录，[task.py](../ai/task.py) 将完整映射封存到任务的 `api_context.json.function_mapping`。本目录继续只保存映射资产；生成、提示词与验证驱动归 `backend/ai/`，`front/crossfhe` 通过统一 CLI 保留 `python -m crossfhe ai prepare|baseline|run`。

| 文件 | 后端 | 库及审查 revision | 实际适配器 |
| --- | --- | --- | --- |
| [seal.json](seal.json) | `seal-ckks` | Microsoft SEAL 4.1.2，`119dc32e135cb89c1062076a69310d4413ebc824` | [seal_runtime.hpp](../seal/seal_runtime.hpp) |
| [openfhe.json](openfhe.json) | `openfhe-cross` | OpenFHE 1.4.2，`aa391988d354d4360f390f223a90e0d1b98839d7` | [openfhe_runtime.hpp](../openfhe/openfhe_runtime.hpp) |
| [phantom.json](phantom.json) | `phantom-ckks` | PhantomFHE 2.1，`1f4a198443b3af77118e51f53d5b8f332154b875` | [phantom_runtime.hpp](../phantom/phantom_runtime.hpp) |

版本核对了各上游 `CMakeLists.txt` 与 [UPSTREAM.md](../../docs/UPSTREAM.md)。[bootstrap.py](../../scripts/bootstrap.py) 初始化父仓库固定的 submodule gitlink 并记录实际 revision；其中 Phantom 另有显式 revision 检查。自定义源码目录的 revision 不一定等于上表，必须分别核对，不能只凭库名推断兼容性。

## 对应关系

以下表格是索引。重载签名、上游限定名称、零值分支和前置条件以各 JSON 的 `mappings` 为准。

| IR 或路径 | SEAL | OpenFHE | Phantom |
| --- | --- | --- | --- |
| `arith.constant` | 生成公开 `const double`，不调用密码库 | 同左 | 同左 |
| `heir.materialize` | `constant` → encode + 公钥 encrypt | `constant` → MakeCKKSPackedPlaintext + Encrypt | `constant` → encode + encrypt_asymmetric |
| `heir.rlweadd` | `add` 的密密/密明/明密重载 | `add` → EvalAdd 相应重载 | `add` → add / add_plain |
| `heir.rlwesub` | `sub` → sub / negate + add_plain；自减公钥加密零 | `sub` → EvalSub / EvalNegate + EvalAdd | `sub` → sub / negate + add_plain；自减转 zero_like |
| `heir.rlwemul` | `mul` → multiply + relinearize / multiply_plain，随后 rescale；公开零特殊处理 | `mul` → EvalMult 相应重载 | `mul` → multiply_and_relin / mul_periodic，随后 rescale；公开零特殊处理 |
| `heir.packed_add` | 密密 `add` | 未接入 | 密密 `add` |
| `heir.packed_rotate` | `rotate` → rotate_vector | 未接入 | `rotate` → phantom::rotate |
| `heir.packed_zero` | `zero_like` → 目标层 encrypt_zero | 未接入 | `zero_like` → 初始层 encrypt_zero_asymmetric，再降层 |
| `heir.packed_mul_plain` | `mul_periodic` → 周期扩展/编码 + multiply_plain + rescale | 未接入 | `mul_periodic` → 周期扩展/编码 + multiply_plain + rescale |
| `heir.packed_add_plain` | `add_periodic` → 周期扩展/编码 + add_plain | 未接入 | `add_periodic` → 周期扩展/编码 + add_plain |
| `heir.depack` | 不支持 | `to_logic` → EvalCKKStoFHEW(value,1) | 不支持 |
| `heir.lut_lt` | 不支持 | `sign` → BinFHEContext::EvalSign(value[0],true) | 不支持 |
| `heir.repack` | 不支持 | `to_ckks_mask` → EvalFHEWtoCKKS 的固定标量/掩码编码 | 不支持 |
| `heir.select` | 不支持 | `select` 组合计算 `mask*(a-b)+b` | 不支持 |
| 层级与 scale 维护 | 私有 `align`、`refreshScale` | 由固定上下文及 EvalAdd/EvalSub/EvalMult 的库实现处理 | 私有 `validate`、`align`、`switchTo`、`requireNext`、`refreshScale` |

三个后端的 evaluator 都不需要私钥。公钥常量物化及零值加密属于合法的服务端操作，不能据此向候选开放客户端明文或解密。客户端 keygen/encrypt/decrypt 的完整接口不在本目录的 evaluator 映射范围内。

SEAL 与 Phantom 的 packed 路径沿用编译器限定的单输入/输出、periodic 布局公开仿射 MatVec，不能据此推断支持任意网络布局。Phantom 当前适配器没有公开 stream/event 调度接口，也没有接入 CKKS bootstrapping 或 CKKS↔FHEW 转换。OpenFHE 的 sign 使用该库内置 FHEW 路径，当前项目没有独立 TFHE 后端；`heir.lut_lt` 也不代表已经实现任意 LUT。

SEAL 的 chain index 随降层减小，Phantom 随降层增大。两者不能直接互换层级比较条件。OpenFHE 的 `select` 是 runtime 组合，不是一个可以凭名称直接调用的库函数；其中公开分支可能使用普通 C++ double 辅助重载，秘密条件仍由加密掩码参与计算。

## JSON 字段

顶层固定记录 `schema_version`、后端/库标识、版本/revision、runtime 相对路径及其原始字节 SHA-256、codegen 来源和审查说明。`library` 使用 `seal`、`openfhe`、`phantom` 三个机器标识。每个映射项包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `operation` | 文件内唯一的稳定标识；不同重载使用不同标识 |
| `ir_ops` | 生成该路径的 IR 名称；私有辅助或公开 double 辅助可为空 |
| `runtime` | 当前适配器的限定签名；`public_constant` 明确标记没有 runtime 调用 |
| `library_calls` | 当前实现直接或经转发使用的真实上游限定入口；分支/重载的并集不表示每次全部执行 |
| `description` | 当前实现的计算语义、调用次序及必要的分支说明 |
| `constraints` | 参数、布局、评估键、效果与可访问性边界 |

私有辅助路径单独列出，便于识别复制、编码和维护成本；不表示候选能够直接调用这些私有方法。标准容器、计数器和计时函数不作为密码库功能登记。`library_calls` 不是完整的递归调用图，例如乘法后执行的 rescale 应结合 `refresh_scale` 项阅读。

## 维护与核验

1. 修改 [CodeGen.cpp](../../compiler/CodeGen/FHE/CodeGen.cpp) 或 runtime 时，逐条核对 IR 名称、公开/密文重载、常量零分支和辅助调用。新增能力必须先有实际实现与验证，再补映射；不能由外部摘要推导出项目不存在的接口。
2. 对照本机固定 revision 的上游头文件核验限定名称、实参及语义。保留 `operation` 稳定标识，必要时更新版本/revision；发生格式不兼容时再升级 `schema_version`。
3. 按原始文件字节重新计算并更新对应 `runtime_sha256`，例如在仓库根执行 `sha256sum backend/seal/seal_runtime.hpp`。换行和任何字节变化都会改变 hash，不能只改 hash 而省略重新审查。
4. 运行现有 AI 上下文/任务准备测试，确认三个固定 JSON 均能读取，映射 hash 与任务中的 runtime 相符，默认资料无需 `/shizy/fhe-coder` 等外部目录。涉及实际 API 行为的改动，再运行对应原生/加密测试。
5. 新任务重新封存映射与源码上下文。既有任务的 hash、原始响应和验证证据保持可回放，不能通过改写旧任务绕过版本或完整性检查。

任务默认包含完整的本地函数映射，并可按符号检索其中片段。显式指定的 FHE-Coder `summaries_db.json` 只作可选辅助摘要，其上游 revision 未知，不能覆盖本地映射、已核验 runtime 或编译器契约。读取 JSON/提供提示词上下文本身不等于候选已经编译或验证通过。
