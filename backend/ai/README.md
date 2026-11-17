# AI 辅助后端

本目录是 Python 包 `backend.ai`，负责已确定 FHE 算法之后的任务准备、提示词、候选调度生成、构建、差分验证与反馈。统一命令由 `front/crossfhe/cli.py` 接入；任务与验证直接复用 `backend.toolchain` 和 `backend.execution` 的工具链、指纹与 runner 构建接口。

| 文件 | 职责 |
| --- | --- |
| `cli.py` | 注册 `ai prepare`、`ai baseline`、`ai run` 并分发参数 |
| `task.py`、`artifacts.py` | 准备公开任务，检查 artifact，封存文件与 hash |
| `context.py` | 读取相邻 `backend/api/` 的本地函数映射及可选辅助文档 |
| `protocol.py` | 校验任务、候选和不可变约束 |
| `prompts.py`、`provider.py` | 构造提示词，离线回放或调用显式配置的模型 |
| `driver.py`、`validation.py` | 编排候选与修复，执行真实构建、加密差分和测量 |

在仓库根使用既有 Python 环境：

```bash
python -m pip install --no-deps -e .
python -m crossfhe ai --help
python -m pytest tests/backend/ai tests/compiler/test_ai_backend.py -q
```

也可在仓库根设置 `export PYTHONPATH="$PWD:$PWD/front${PYTHONPATH:+:$PYTHONPATH}"` 直接使用源码。任务准备统一使用 `python -m crossfhe ai prepare`。

AI 仅提出受限调度 JSON，原生代码生成和密码运算分别由 `compiler/CodeGen/FHE` 及各库 runtime 执行。完整用法和边界见 [AI 后端说明](../../docs/development/AI_BACKEND.md)，库调用关系见 [函数映射](../api/README.md)。
