# 文献阅读与实现边界

后续 MatVec 专题已完成：新增阅读 GALA、LinearTransform-SmallKey 的 diagonal/BSGS 章节和 Coyote 的向量化论述，并实现 SEAL CPU 周期槽位 MatVec pass。详见 [专题研究、代码映射和实测](MATVEC.md)。下文保留前一轮标量优化的文献与边界记录。

## 阅读范围

`articles/` 共 253 篇 PDF，分为 Compiler 22、CNN 29、GPU 15、Hardware 22、Matrix Multiplication 20、PPML 80、Retrieval 17、Scheme 27、Training 21 篇。已全量提取文本、记录页数与 SHA-256，并对题名、摘要和关键字做相关性筛选；重点阅读下表中与本轮变换直接相关的内容。**这不表示已逐页精读全部 253 篇。** 文件清单见 [literature-index.csv](literature-index.csv)，本机检索文本位于 `.crossfhe/articles/`。

开题报告通过 antiword 读取。实现依据为第 3.1、4.1 节与答辩改进意见：手写 FHE 场景专用 pass；只对需要 FHE 适配的模块触发；后端 AI 负责适配辅助；先真实编译和比较结果，再反馈修复，最多 10 轮。报告中有关整体模型精度低于 0.1%、全量算子支持等是研究目标，不作为当前已完成能力。

## 论文与代码的对应关系

| 本地论文 | 关注点 | 本轮处理 |
| --- | --- | --- |
| `FHE-Compiler/2026-Libra.pdf`，摘要与 cross-scheme/pattern-scheduling 设计 | 按 FHE 原语和跨方案模式评估成本，并与硬件调度联动 | 共享完整的重复比较/转换链；导出实际原语计数。没有 GPU 实测成本，因此不做 Libra 式 GPU 最优模式选择 |
| `FHE-Compiler/2024-CHLOE.pdf`，transparent/oblivious loops 与 control-path reduction | 控制路径和多级向量化 | 当前 IR 无 loop/tensor 结构，先裁剪不可达于输出的标量密文链，保留未来控制路径分析接口；不声称复现 CHLOE 向量化 |
| `FHE-Compiler/2026-Ciphertext-Poly-Opt.pdf`，引言中的 CSE、ModDown hoisting 与 liveness scheduling | 密文层与多项式层冗余、管理操作与生命周期 | 手写密文层 CSE/DCE。当前无 NTT/ModDown IR，故没有实现论文的多项式层优化或 hoisting |
| `FHE-Compiler/2020-Efficient Homomorphic Computation.pdf`（EVA） | 深度、scale 与 rescale 管理 | 基于现有库契约重算深度，显式启用乘法树平衡；rescale 仍由 runtime 执行 |
| `FHE-Compiler/2023-Error-Latency-Aware Scale Management.pdf`（ELASM），摘要及误差/延迟权衡 | 输出误差必须参与优化决策 | 不把深度估计当成噪声证明；近似和重关联单独 opt-in，保存误差和参数信息 |
| `FHE-Compiler/2026-Orbit.pdf`，ILP 联合 rescale/bootstrap 放置 | level budget 将维护操作耦合 | 当前无显式维护 IR 和误差模型，列为后续；未添加只有名字的 ILP pass |
| `FHE-CNN/2024-PolyApprox-NonPoly.pdf`（SMART-PAF），输入范围与 progressive approximation | 多项式替换必须关注输入分布、范围、精度恢复 | 本轮只实现可解析验证的有界二次 ReLU；要求显式范围和局部预算，不冒充 SMART-PAF 的校准/微调，也不推广到整个网络的误差上界 |
| `FHE-Compiler/2025-TensorCompiler.pdf`（Rotom），layout assignment/ApplyRoll | 保留 tensor 结构，搜索 packing/layout 和布局变换 | 标量化已丢失矩阵/通道结构，必须先补 tensor/slot 层和 rotate 后端，再实施 |
| `FHE-Compiler/2021-Porcupine-A Synthesizing Compiler.pdf` | 基于合成的向量化、参考语义验证 | 用确定性参考约束候选，不把源码能编译等同于语义正确；向量化合成暂未实现 |
| `FHE-Compiler/2025-ILA.pdf` | 通过类型表达 FHE 可执行性条件 | 当前资源契约只有结构性预算，不宣称拥有 ILA 的类型正确性证明 |
| `FHE-Compiler/2025-HEIR.pdf` | 多层抽象、可组合优化与后端 | 参考其分层思想。该 PDF 是 Google 等团队的 Universal HEIR；本仓库沿用的 Bian/CHLOE HEIR 方言是另一个来源，不能混称已移植该 PDF 的实现 |
| `FHE-Compiler/2026-KeyMemRT.pdf` | rotation-key 内存占用与生命周期 | 当前标量路径没有 rotation keys，待 packing 后再开展这类优化 |

其他 Scheme、GPU、Hardware、Matrix Multiplication、Training 和 Retrieval 论文用于确定研究边界。不能把 BFV/BGV 的精确整数恒等式、TFHE 的 bit/LUT 变换或 GPU 内核融合直接套到当前 CKKS 实数近似和 FHEW sign 接口。

## 本轮具体数学及工程约束

1. 默认优化使用同一 SSA 值的结构共享和局部恒等式，不交换加乘的操作数，不默认重关联。`ppml.source` 仅用于诊断，可在 CSE 等价判定时忽略；编码、类型、noise-status 等其他属性仍须相同。
2. `heir.materialize` 调用公开密钥加密，带有随机性，不能作为普通纯函数做 CSE；没有输出使用的 materialization 可以被 DCE 删除。
3. 局部简化会改变 CKKS 的 rescale 次数和近似误差，依靠与明文参考的密文回归验证应用精度；它不承诺密文字节或噪声分布相同。
4. 平衡仅处理单次使用、纯密文乘法子树，优先结合较浅的输入，只有预测深度下降才改写。不会复制共享子树；仍保留既有逐次重线性化/rescale runtime。
5. 二次 ReLU：`p(x)=x/2+x²/(2B)+B/16`。设 `t=|x|/B∈[0,1]`，则误差为 `(B/2)(t²-t+1/8)`，取值位于 `[-B/16,B/16]`。这是**声明区间内、实数算术、单个激活**的近似界，不包含系数量化、浮点、CKKS 或下游误差放大。输入区间是调用者给定的前提，当前没有静态范围证明。
6. 保留 `ppml.noise_status = "unanalysed"`。没有真实噪声传播模型时，不写入“噪声已优化”“128 位安全已证明”或预测加速比。

## fhe-coder 参考代码审阅

阅读了 `/shizy/fhe-coder/README.md`、其 ICLR 2026 PDF 摘要、`agents/react_agent.py` 以及 `tools/compiler.py`、`executor.py`、`security_check.py`、`summary_rag.py`。

可借鉴 Prompt Formalizer → 版本化 API RAG → 候选生成 → 校验反馈的结构。本地 checkout 的 `Compiler.compile()` 直接返回 `Compilation successful.`；`Executor.execute()` 忽略退出码和数值结果，并使用 `random.choice([True, True])`；`security_check.py` 没有实际执行 minimum_lambda 估计，最终返回 `Code is secure!`。这是**本地文件的事实**，不能据此断言论文或所有上游版本使用相同实现。

因此本项目不导入这些工具作为验证器。AI 模块设计、离线原型及实际边界见 [AI_BACKEND.md](AI_BACKEND.md)。
