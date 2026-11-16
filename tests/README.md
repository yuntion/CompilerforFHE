# Tests

所有命令在仓库根目录执行。原生编译器先由 `scripts/bootstrap.py` 构建，Python 包从根 `pyproject.toml` 安装。

| 目录 | 验证内容 | 是否执行密码运算 |
| --- | --- | --- |
| `front/` | ONNX 算子、类型、静态形状、限制与拒绝诊断 | 否 |
| `compiler/` | 原生工具注册、非法 MLIR、编译产物状态、导出 tracing 拒绝 | 否 |
| `compiler/inputs/` | 可直接交给 C++ 编译器的 MLIR fixture | CTest 不执行密码运算 |
| `backend/` | SEAL 上的密文边界和复合计算 | 是，需 `--run-fhe` |
| `e2e/` | 模型到生成代码、密文推理、精度验证、资源契约与真实加密 A/B 检查 | 是，需 `--run-fhe` |

```powershell
ctest --test-dir build --output-on-failure
python -m pytest -q --basetemp .crossfhe\test-unit
python -m pytest --run-fhe -q --basetemp .crossfhe\test-all --junitxml docs\validation-junit.xml
```

`--basetemp` 是 pytest 专用的可重新生成目录，不要指向自己的数据、模型或长期保存的密钥。各组命令使用不同目录，便于保留本轮报告。Python 默认只收集这四组新测试；原仓库的 `test.py`、`tosa-mlir` 和 `gpu-module` 是旧示例，不属于 FHE 测试结果。

未配置原生编译器时，部分 `tests/compiler` 测试会跳过；完整验证须先 bootstrap，并确认没有意外跳过。OpenFHE 回归测试使用 TOY 参数，安全配置与论文性能实验单独执行，详见 [VALIDATION.md](../docs/VALIDATION.md)。

新增专项测试见 `compiler/test_optimizations.py`、`compiler/test_ai_backend.py` 和 `e2e/test_optimization_execution.py`；使用方式和实测结果见 [优化开发报告](../docs/development/RESULTS.md)。
