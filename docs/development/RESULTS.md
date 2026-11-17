# 文献驱动优化开发与验证结果

日期：2026-11-16（Asia/Shanghai）。本轮完成 [开发计划](PLAN.md) 的 P0—P6：新增 6 个可独立调用的专用 pass、默认优化组合、原生资源 JSON 导出、Python/CLI 选择与 manifest 契约更新，以及 AI 后端离线任务准备器。没有将尚不具备执行接口的 packing、GPU 和 scale/bootstrap 联合优化标记为已实现。

## 实现和文献依据

阅读 `articles/开题报告表.doc` 的研究方案和答辩意见；253 篇 PDF 已分类索引，重点阅读 Compiler 类及与输入范围/多项式近似直接相关的文章。逐页精读范围与实现对应关系见 [ARTICLES.md](ARTICLES.md)，并非声称完整精读所有 PDF。

| 代码 | 本轮变化 |
| --- | --- |
| `compiler/Pipelines/FHE/Optimizations.cpp` | 手写 FHE 简化、CSE、DCE、深度/资源分析和显式乘法树平衡 |
| `compiler/Pipelines/FHE/ApproximateRelu.cpp` | 显式接受输入范围与局部误差预算的 SCFHE ReLU 二次近似 |
| `compiler/Pipelines/FHE/OptimizationHooks.cpp` | 默认组合简化/CSE/DCE/契约刷新，不默认重关联或近似 |
| `compiler/Conversion/FHE/` | 在正常 FHE 路径携带函数启用标记与近似元数据 |
| `tools/crossfhe-translate.cpp` | 从经过 codegen 合法性检查的 typed IR 导出资源契约；拒绝非法资源计数属性 |
| `front/crossfhe/pipeline.py`、`cli.py` | `none/scalar/depth`、ReLU 参数、前后资源 JSON、实际比较数/深度、IR 与契约 hash |
| `scripts/prepare_ai_backend.py` | 导出公开任务上下文，输出未验证状态；不调用模型或执行候选 |

详细 pass 语义、数值前提与调用方式见 [PASSES.md](PASSES.md)。FHE 变换不依赖通用 canonicalize/cse pass，不对 `heir.materialize` 做 CSE，未知操作不会被当作无副作用操作删除。

## 环境与实际验证命令

沿用 `CompilerforFHE` Conda 环境，Python 3.12.16、PyTorch 2.8.0+cpu、ONNX 1.19.0、NumPy 2.2.6、pytest 8.4.2、CMake 3.31.6、Ninja 包 1.11.1.4，Ubuntu 24.04.5、GCC 13.3.0，已有固定 revision 的 SEAL/OpenFHE。前端包未安装，通过 `PYTHONPATH=front` 读取源码。完整依赖见 [pip-freeze.txt](validation/pip-freeze.txt)；机器和原生工具链沿用 [上一轮环境记录](../validation/2026-11-16-conda/environment.json)。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate CompilerforFHE
export PYTHONPATH="$PWD/front"

cmake --build build --target crossfhe-opt crossfhe-translate --parallel 8
python -m pytest tests/compiler/test_optimizations.py -q --basetemp .crossfhe/test-new-passes
python -m pytest tests/compiler/test_ai_backend.py -q --basetemp .crossfhe/test-ai-task
python -m pytest tests/e2e/test_optimizations.py --run-fhe -q --basetemp .crossfhe/test-pass-ab --junitxml .crossfhe/pass-ab-junit.xml

# 消除两个目录同名测试模块的收集冲突后，e2e 文件改名为 test_optimization_execution.py。
# 以下完整回归使用当前文件名，涵盖修正后的 A/B 输入。
python -m pytest --run-fhe -q -ra --basetemp .crossfhe/test-optimized-all --junitxml docs/development/validation/full-junit.xml

# 完整回归后补齐 3 项 CTest 并完善非法资源属性导出拒绝，最终重建/定向验证：
cmake --build build --target crossfhe-opt crossfhe-translate --parallel 8
ctest --test-dir build --output-on-failure --output-junit "$PWD/docs/development/validation/native-junit.xml"
python -m pytest tests/compiler -q --basetemp .crossfhe/test-final-contract

python scripts/collect_reports.py .crossfhe/test-optimized-all --output docs/development/validation/measurements
python scripts/prepare_ai_backend.py .crossfhe/test-optimized-all/test_approximate_relu_runs_on_0/relu --output .crossfhe/ai-backend-verified-input
```

开发中第一次 A/B 输入使用重复 PyTorch 表达式，但导出器提前合并了它们，两个资源削减断言未成立。随后改为显式保留重复节点的 ONNX，并保留独立 PyTorch 数值参考。另修正了新增 compiler/e2e 测试文件同名导致的 pytest 收集冲突。最终完整回归结果如下，未放宽现有回归断言或原有精度容差。

| 检查 | 结果 |
| --- | --- |
| 完整 Python 回归 | **59 passed, 35 warnings in 308.10s (0:05:08)**；0 失败、0 跳过 |
| 最终 CTest | **6/6 通过**，0.02 秒 |
| 最终 compiler 定向验证 | **29/29 通过**，1.32 秒 |
| 完整回归中的真实加密执行 | **18/18 通过**：原有 11 次 + 新增 A/B/近似 7 次 |
| 资源属性异常导出检查 | 非整数计数返回 exit code 1 和诊断，没有断言崩溃 |
| AI 公开任务准备 | 对真实 artifact 成功导出，状态 `prepared_unvalidated` |

新增加的 Python 用例为 17 项 pass 变换/拒绝测试、4 项 AI 任务准备测试、4 项 FHE A/B/近似测试，共 25 项；原来的 34 项继续保留。最终 59 项覆盖 front 15、compiler 29、backend 3、e2e 12。新增 CTest 检查重复比较共享、深度平衡和有界 ReLU 转换。

35 条 warning 为旧 ONNX 导出 API 的弃用提示和有意常量输出的 tracing 提示；未产生测试失败。完整回归后最终的小改动只增加资源 JSON 非法属性拒绝和 CTest 注册，未修改已测的 FHE 图变换或 runtime；已重建并执行上表定向验证。

## 实测资源和精度

每组 A/B 使用相同的测试输入和明文参考，但重新生成密钥并加密。以下时间为 runner 内部 evaluator 的单次测量，不包含进程启动和密钥加载，也不构成统计基准。

| 用例 | 模式 | 实际深度 | 运行时密密乘法 | 运行时逻辑 sign 调用 | 最大绝对误差 | evaluator 秒 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 二次 ReLU | scalar | 2 | 5 | 0 | 6.250046e-02 | 0.030700 |
| 密文乘法 CSE | none | 1 | 4 | 0 | 5.123798e-08 | 0.011047 |
| 密文乘法 CSE | scalar | 1 | 2 | 0 | 4.993550e-08 | 0.005768 |
| 乘法树平衡 | depth | 3 | 7 | 0 | 5.071470e-06 | 0.072347 |
| 乘法树平衡 | none | 7 | 7 | 0 | 9.509025e-06 | 0.127748 |
| ReLU 比较共享 | none | 1 | 4 | 4 | 5.882800e-07 | 7.423961 |
| ReLU 比较共享 | scalar | 1 | 2 | 2 | 5.882656e-07 | 3.577280 |

- 重复乘法的真实密密乘法 **4 → 2**，两条路径都通过 `atol=rtol=0.001` 的明文比较和相互比较。
- 八项连乘深度 **7 → 3**，密密乘法仍为 7 次；两条路径均为 SEAL tc128。层数/模数链会随资源计划变化，不能将单次耗时差全部解释为调度收益。本机这两条路径最终环维度均为 16384。
- 重复 ReLU 的 CKKS→FHEW / sign / FHEW→CKKS 调用均 **4 → 2**，OpenFHE TOY，两条路径都通过原有 `0.001` 容差；资源计数与运行时计数一致。
- 二次 ReLU 输入覆盖 `[-1,-0.5,0,0.5,1]`，即解析误差极值点。局部实数近似界为 `1/16=0.0625`，密文对 ReLU 的实测误差约 `0.06250046`；额外 CKKS 误差单独留出余量，最终比较使用 `atol=0.063, rtol=0`，另以 `atol=0.0001` 对照多项式本身。它不是默认精确 ReLU 的性能替代结论，也不是小于 0.1% 的模型精度结论。
- 原有 11 次加密执行均继续通过 `atol=rtol=0.001`；其最大绝对误差为 `2.941373071e-06`。
- 18 条报告的 `validation.passed` 与 `server_has_no_secret_key` 均为 true。SEAL 使用 tc128；OpenFHE 回归使用显式 TOY，没有在本轮重测默认安全参数的 ReLU。

详尽的原语数量、前后深度、配置、容差、运行时计数和误差见 [cases.json](validation/cases.json)，通用收集器输出见 [results.json](validation/measurements/results.json) / [results.csv](validation/measurements/results.csv)。原始阶段 IR、CMake/编译/密钥生成/加密/评估/解密日志保留在 `.crossfhe/test-optimized-all/` 各用例目录。

## AI 后端结论与下一步

已阅读 `/shizy/fhe-coder` 的论文摘要、agent、RAG、compiler、executor 和 security-check 代码。可采用其结构化规格、库文档检索和有限轮修复设计；本地 checkout 的编译/执行成功判断为占位实现，不能用于接受候选。

本轮实现真实 artifact → hash 校验 → 公开任务包的离线入口。在线模型调用、隔离编译/执行、独立客户端验证及候选晋升尚未实现；任务准备成功不表示已经完成 AI 代码生成。详细状态机、API 检索、最多 10 轮修复与评估方案见 [AI_BACKEND.md](AI_BACKEND.md)。

后续优先保留 tensor/slot 结构并接入 rotate/key 接口，才能开展开题报告中的 CPU SIMD/对角线编码；GPU 和 scale/bootstrap 全局优化也需要对应原语及误差模型。范围、前置条件与验收标准已列在 [PLAN.md](PLAN.md)。

## 日志索引

- [完整 Python 日志](validation/full.log)、[完整 JUnit](validation/full-junit.xml)
- [最终构建](validation/build.log)、[6 项 CTest 日志](validation/native.log)、[原生 JUnit](validation/native-junit.xml)
- [29 项 compiler 定向验证](validation/compiler-final.log)、[非法资源属性拒绝](validation/invalid-resource.log)
- [18 次加密执行指标](validation/cases.json)、[收集日志](validation/collect.log)
- [真实 artifact 的 AI 任务准备结果](validation/ai-prepare.log)
