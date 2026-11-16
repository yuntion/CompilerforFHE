# 新增编译器 Pass

本文使用当前项目的手工 `PassWrapper` 和注册方式，命令使用 Ubuntu/Bash。示例是一个独立的深度契约检查 pass，用于演示扩展流程；它尚未加入正式源码，也不替代现有专项优化。

## 1. 选择模块和执行阶段

| Pass 职责 | 所在目录 | 输入阶段 |
| --- | --- | --- |
| 新方言到 SCFHE/HEIR 的 lowering | `compiler/Conversion/<Name>` | 对应源方言 |
| scheme/layout/precision 等分析或优化 | `compiler/Pipelines/FHE`，复杂时新建独立模块 | 明确指定的 FHE IR 层次 |
| 生成后端代码 | `compiler/CodeGen/FHE` | 最终、已验证的 FHE IR |

当前命令顺序是 `--ppml-to-scfhe`、`--ppml-to-heir`、`--ppml-fhe-optimize`。Python 驱动在 `front/crossfhe/pipeline.py` 逐阶段调用工具。注册新 pass 会增加命令行选项，默认流水线不会自动使用它。

## 2. 编写 Pass 实现

新增 `compiler/Pipelines/FHE/CheckDepthContract.cpp`：

```cpp
#include "Pipelines/FHE/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace crossfhe {
namespace {
struct CheckDepthContractPass
    : mlir::PassWrapper<CheckDepthContractPass,
                       mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckDepthContractPass)

  llvm::StringRef getArgument() const final {
    return "crossfhe-check-depth";
  }
  llvm::StringRef getDescription() const final {
    return "Check the nonnegative depth attribute on lowered FHE functions";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::func::FuncDialect>();
  }
  void runOnOperation() final {
    bool hasFunction = false;
    for (auto function : getOperation().getOps<mlir::func::FuncOp>()) {
      hasFunction = true;
      auto depth = function->getAttrOfType<mlir::IntegerAttr>(
          "ppml.multiplicative_depth");
      if (!depth || depth.getInt() < 0) {
        function.emitError("expected nonnegative ppml.multiplicative_depth");
        signalPassFailure();
        return;
      }
    }
    if (!hasFunction) {
      getOperation().emitError("expected at least one FHE function");
      signalPassFailure();
    }
  }
};
}

std::unique_ptr<mlir::Pass> createCheckDepthContractPass() {
  return std::make_unique<CheckDepthContractPass>();
}
}
```

这个 pass 只检查属性存在且非负，不证明实际乘法深度、噪声或安全参数。它应在 HEIR lowering 后运行，直接对前端 arith IR 运行会因缺少契约而报错。

`getArgument` 决定 `--crossfhe-check-depth` 选项，必须唯一。诊断使用 `emitError` 绑定到出错的操作，再调用 `signalPassFailure` 使工具退出失败。

## 3. 声明、注册和链接

在 `compiler/Pipelines/FHE/Passes.h` 的 `namespace crossfhe` 中加入：

```cpp
std::unique_ptr<mlir::Pass> createCheckDepthContractPass();
```

在 `compiler/Pipelines/FHE/OptimizationHooks.cpp` 的 `registerCrossFHEPasses` 中，保留现有的所有 pass 注册并新增一项：

```cpp
void registerCrossFHEPasses() {
  mlir::registerPass([] { return createLowerToSCFHEPass(); });
  mlir::registerPass([] { return createLowerToHEIRPass(); });
  mlir::registerPass([] { return createOptimizationHookPass(); });
  mlir::registerPass([] { return createCheckDepthContractPass(); });
}
```

在 `compiler/Pipelines/FHE/CMakeLists.txt` 的 `add_library` 后添加源文件：

```cmake
target_sources(CrossFHEPipelines PRIVATE CheckDepthContract.cpp)
```

现有 `CrossFHEPipelines` 已通过 `CrossFHEConversion` 链接所需 Func/Pass 依赖。新增 pass 若需要别的 dialect 或接口，显式补充 `target_link_libraries` 和 `getDependentDialects`。如果 pass 放入新模块，还需由 `compiler/CMakeLists.txt` 添加子目录，并把库链接到使用它的工具。

## 4. 编译和手工运行

```bash
conda activate CompilerforFHE
cmake --build build --target crossfhe-opt --parallel 4
build/bin/crossfhe-opt --help

build/bin/crossfhe-opt tests/compiler/inputs/affine_fhe.mlir \
  --crossfhe-check-depth --verify-each
build/bin/crossfhe-opt tests/compiler/inputs/relu.mlir \
  --ppml-to-scfhe --ppml-to-heir --crossfhe-check-depth --verify-each
```

compiler-only 构建使用 `build-compiler/bin/crossfhe-opt`。第二条命令先完成 lowering，再检查契约。无需更改 `tools/crossfhe-opt.cpp`，因为它已经调用 `registerCrossFHEPasses`。

也可显式指定嵌套 pipeline：

```bash
build/bin/crossfhe-opt tests/compiler/inputs/affine_fhe.mlir \
  --pass-pipeline='builtin.module(crossfhe-check-depth)' --verify-each
```

此例为 `OperationPass<ModuleOp>`，所以挂在 `builtin.module` 下；若改为 `OperationPass<func::FuncOp>`，应嵌套到 `builtin.module(func.func(...))`。

## 5. 添加正例与拒绝测试

当前已有两个可复用的 fixture：`affine_fhe.mlir` 含有效深度属性，`relu.mlir` 是未降级的 arith IR，没有该属性。

```bash
mkdir -p artifacts/pass-dev
build/bin/crossfhe-opt tests/compiler/inputs/affine_fhe.mlir \
  --crossfhe-check-depth --verify-each -o artifacts/pass-dev/valid.mlir

if build/bin/crossfhe-opt tests/compiler/inputs/relu.mlir \
    --crossfhe-check-depth --verify-each; then
  echo 'ERROR: the missing depth contract was accepted' >&2
  exit 1
fi
```

在 `tests/CMakeLists.txt` 的 FHE 分支添加：

```cmake
add_test(NAME crossfhe-depth-contract-valid COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/affine_fhe.mlir"
  --crossfhe-check-depth --verify-each)
add_test(NAME crossfhe-depth-contract-missing COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/relu.mlir"
  --crossfhe-check-depth --verify-each)
set_tests_properties(crossfhe-depth-contract-missing PROPERTIES WILL_FAIL TRUE)
```

修改 CMake 后重新构建，再执行 `ctest --test-dir build --output-on-failure`。还应覆盖负深度和无函数模块；转换 pass 要检查生成 IR、拒绝条件和真实运行语义，而不是仅验证 pass 名能出现在 help 中。

## 6. 编写转换或优化 Pass

上面的示例只读取 IR。需要修改 IR 时，使用 MLIR `OpBuilder`、`PatternRewriter`、`RewritePatternSet` 和 conversion API，避免直接改文本。参考 `compiler/Conversion/FHE/LowerToSCFHE.cpp` 和 `LowerToHEIR.cpp`。

转换通常需要类型映射、合法/非法操作集合、rewrite pattern，以及转换结束后的 verifier。新 dialect 能解析，但没有到后续合法操作的完整映射时，不应进入 codegen。

针对 FHE 的变换还要维护：

- 公开/秘密值和 ciphertext 类型，不能通过解密实现变换。
- CKKS level、scale、误差与运算资源契约。
- CKKS/FHEW 比较的有符号范围、量化和 margin 条件。
- layout、slot 映射、输出 tensor mapping 和转换编码。
- 更新的深度/比较计数等属性，使后端参数计划与生成代码一致。

仅移动操作或设置 `noise = 0` 不构成正确的 FHE 优化。新增优化应先验证合法性和数值语义，再测量精度、转换次数、延迟和应用准确率。

## 7. 接入端到端流水线

建议先把新 pass 作为独立 CLI 选项验证，再确定默认顺序。可以在 `front/crossfhe/pipeline.py` 的 `steps` 中新增原生工具调用和日志，或在编译器中定义命名 pipeline；不要把 pass 实现写到 Python 前端。

当前 `--ppml-fhe-optimize` 已实现局部简化、CSE、DCE 和资源契约刷新；Python 驱动使用原生导出的优化前后资源 JSON，不再要求两个阶段 IR 字节相同。新增 pass 应继续保留 verifier、artifact hash、资源契约和明文/密文精度验证，并添加针对实际变换的 A/B 测试。现有 pass 见 [PASSES.md](development/PASSES.md)。

最后运行对应原生检查和全链路：

```bash
ctest --test-dir build --output-on-failure
python -m pytest tests/compiler -q --basetemp .crossfhe/test-compiler
python -m pytest --run-fhe -q --basetemp .crossfhe/test-all
```

性能对比应固定模型、数据、scheme 参数、比较 scale 和输出精度。TOY 模式用于快速功能回归，默认安全参数用于独立实验。详细数值契约见 [BACKENDS.md](BACKENDS.md)。
