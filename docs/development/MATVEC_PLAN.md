# CPU SIMD / 对角线 MatVec 开发计划

这里的 SIMD 指 CKKS 密文内槽位的并行计算，由 SEAL 在 CPU 上执行。不是新增 AVX 指令后端。

| 步骤 | 交付 | 验收 |
|---|---|---|
| 1 | 阅读本地 GALA、Coyote、LinearTransform-SmallKey 的相关章节 | 写明原始算法、公式和实现边界 |
| 2 | 保留公开矩阵、秘密向量的 MatVec IR，显式周期槽位布局 | 类型、尺寸、有限系数检查 |
| 3 | `ppml-fhe-matvec`：diagonal / BSGS / auto | 省略零对角线，复用 baby rotations，生成实际旋转集合 |
| 4 | SEAL CPU runtime、代码生成、打包加解密和 Galois keys | 服务器无私钥，按实际物理槽位周期重复 |
| 5 | ONNX / Python / CLI 显式 opt-in | 支持单向量公开仿射变换，拒绝不支持的网络与后端 |
| 6 | 原生负例、矩形/非二次幂/稀疏/零矩阵、真实 CKKS A/B | 与 NumPy 和标量路径比较，核对运行时资源 |
| 7 | 研究、命令、结果和限制报告 | 记录实测误差、时间、密文和旋转数 |

最终状态和结果见 [MATVEC.md](MATVEC.md)。保持既有标量路径作为默认行为。

步骤 1—7 已完成：7 项 CTest、87 项 Python 测试全部通过，包含 16 次新增真实加密执行；研究范围、完整指令和实测表已写入上述报告。
