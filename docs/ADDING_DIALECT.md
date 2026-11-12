# 新建 MLIR 方言

本文针对本项目当前目录和仓库内 LLVM/MLIR 快照，命令使用 Ubuntu/Bash。方言归 `compiler/Dialect/`，不放进 `front/` 或 `backend/`。下面的 `demo` 是教学示例，尚未注册到正式编译器。

## 1. 先确定扩展位置

| 需求 | 通常修改的位置 |
| --- | --- |
| 接入新的 ONNX 算子 | `front/crossfhe/frontend.py`，以及所需的 compiler lowering |
| 扩展现有 SCFHE/RLWE/LWE 操作 | `compiler/Dialect/SCFHE` 或 `HEIR` |
| 引入新的中间表示，如 tensor layout 或方案区域 | 新建 `compiler/Dialect/<Name>` |
| 把新表示转成现有 FHE 操作 | `compiler/Conversion/` 中的 pass |
| 新表示的分析或变换 | `compiler/Pipelines/` 中的 pass |
| 支持新的 FHE 库 | `backend/` 和 `compiler/CodeGen/FHE` |

先为操作确定输入/输出类型、shape/layout、公开/秘密属性、数值语义和必要副作用，再写 ODS。不要直接把新操作声明为已有后端支持。

## 2. 创建目录和 ODS

以一个 f64 identity 操作为最小例子，新增这些文件：

```text
compiler/Dialect/Demo/
  CMakeLists.txt
  DemoDialect.cpp
  IR/
    DemoDialect.td
    DemoOps.td
    DemoDialect.h
```

`IR/DemoDialect.td`：

```tablegen
include "mlir/IR/OpBase.td"

def Demo_Dialect : Dialect {
  let name = "demo";
  let cppNamespace = "::crossfhe::demo";
  let summary = "Example compiler extension dialect";
}
```

`IR/DemoOps.td`：

```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "DemoDialect.td"

def Demo_IdentityOp : Op<Demo_Dialect, "identity",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Return an f64 value unchanged";
  let arguments = (ins F64:$input);
  let results = (outs F64:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

对应文本语法是 `%result = demo.identity %input : f64`。`Pure` 表示这里没有可观察副作用，不能把它理解为输入是公开值。本例只展示方言定义，没有给 FHE 密文、精度或噪声定义新语义。

## 3. 实现 C++ 声明和注册

`IR/DemoDialect.h`：

```cpp
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "DemoDialect.h.inc"

#define GET_OP_CLASSES
#include "DemoOps.h.inc"
```

`DemoDialect.cpp`：

```cpp
#include "DemoDialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "DemoDialect.cpp.inc"

#define GET_OP_CLASSES
#include "DemoOps.cpp.inc"

void crossfhe::demo::DemoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DemoOps.cpp.inc"
  >();
}
```

`.inc` 文件由 TableGen 生成，不要手工新建或修改。通常也不把生成文件提交到源码目录。

## 4. 接入 CMake

`compiler/Dialect/Demo/CMakeLists.txt`：

```cmake
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/IR"
                    "${CMAKE_CURRENT_BINARY_DIR}/IR")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/IR")

set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/IR/DemoDialect.td")
mlir_tablegen(IR/DemoDialect.h.inc -gen-dialect-decls -dialect=demo)
mlir_tablegen(IR/DemoDialect.cpp.inc -gen-dialect-defs -dialect=demo)
set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_SOURCE_DIR}/IR/DemoOps.td")
mlir_tablegen(IR/DemoOps.h.inc -gen-op-decls)
mlir_tablegen(IR/DemoOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CrossFHEDemoIncGen)

add_library(CrossFHEDemo DemoDialect.cpp)
add_dependencies(CrossFHEDemo CrossFHEDemoIncGen)
target_include_directories(CrossFHEDemo PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/IR" "${CMAKE_CURRENT_BINARY_DIR}/IR")
target_link_libraries(CrossFHEDemo PUBLIC MLIRIR
  MLIRSideEffectInterfaces MLIRInferTypeOpInterface)
if(NOT MSVC)
  target_compile_options(CrossFHEDemo PRIVATE -fno-rtti)
endif()
```

再修改 `compiler/Dialect/CMakeLists.txt` 的正常 FHE 分支。保留 HEIR 和 SCFHE，在创建 `CrossFHEDialect` 之前加入子目录，并扩展链接列表：

```cmake
add_subdirectory(HEIR)
add_subdirectory(SCFHE)
add_subdirectory(Demo)
add_library(CrossFHEDialect INTERFACE)
target_link_libraries(CrossFHEDialect INTERFACE
  CrossFHEHEIR CrossFHESCFHE CrossFHEDemo)
```

这些语句仍位于已有 `if(NOT CROSSFHE_ENABLE_LEGACY_COMPILER)` 内。如果只添加子目录而不把库链接到工具，dialect 注册会出现链接错误。

## 5. 注册到 crossfhe-opt

在 `tools/crossfhe-opt.cpp` 添加 include：

```cpp
#include "DemoDialect.h"
```

将现有 registry 列表扩展为：

```cpp
registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                heir::HEIRDialect, mlir::libra::scfhe::SCFHEDialect,
                crossfhe::demo::DemoDialect>();
```

如果新 pass 会创建该方言的操作，也在该 pass 的 `getDependentDialects` 中声明它，避免依赖工具恰好提前加载某个 dialect：

```cpp
void getDependentDialects(mlir::DialectRegistry &registry) const final {
  registry.insert<crossfhe::demo::DemoDialect>();
}
```

`crossfhe-translate` 当前只接收最终 FHE IR。仅为优化前中间表示新增方言时，把它在转换阶段完全消除即可；若它需要进入代码生成，须同时扩展 translator 的 registry、`generateCpp` 的合法性检查和相应生成规则。

## 6. 编译、解析和测试

在根目录增量构建：

```bash
source .venv/bin/activate
cmake --build build --target crossfhe-opt --parallel 4
build/bin/crossfhe-opt --show-dialects
```

使用 compiler-only 构建时，把命令中的 `build` 换成 `build-compiler`。方言列表应出现 `demo`。

新增 `tests/compiler/inputs/demo-identity.mlir`：

```mlir
module {
  func.func @identity(%x: f64) -> f64 {
    %result = demo.identity %x : f64
    return %result : f64
  }
}
```

解析并打印，再读取一次打印结果：

```bash
mkdir -p artifacts/dialect-dev
build/bin/crossfhe-opt tests/compiler/inputs/demo-identity.mlir \
  --verify-each -o artifacts/dialect-dev/printed.mlir
build/bin/crossfhe-opt artifacts/dialect-dev/printed.mlir --verify-each
```

可在 `tests/CMakeLists.txt` 的 FHE 分支加入：

```cmake
add_test(NAME crossfhe-demo-dialect COMMAND crossfhe-opt
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/inputs/demo-identity.mlir" --verify-each)
set_tests_properties(crossfhe-demo-dialect
  PROPERTIES PASS_REGULAR_EXPRESSION "demo.identity")
```

这只验证注册、解析和 verifier。还应加错误类型、非法 shape、缺失必需属性等拒绝测试；有 lowering 后，再检查其结果不残留不支持的操作，并执行真实库后端的语义比较。

## 7. 扩展成实际 FHE 中间表示

需要自定义 ciphertext/layout 类型时，参照 `compiler/Dialect/SCFHE/IR/SCFHETypes.td`，添加 `TypeDef`、类型 parser/printer 和 `-gen-typedef-decls/-gen-typedef-defs`。类型注册使用 `addTypes`；属性可用 `AttrDef` 或现有 builtin attributes 表达。为 shape、编码、scheme 和 resource contract 添加 verifier。

对方案转换、比较或非线性操作，必须明确 CKKS/FHEW 的范围、量化、误差和输出编码，保留转换键需求等契约。新的密文操作不能通过“注册名称”获得后端能力，需要真实 lowering/codegen/runtime 实现。

当前 `demo.identity` 尚无 lowering，直接交给现有 `--ppml-to-scfhe` 或 FHE codegen 会失败。可在 FHE 降级前添加一个消除 identity 的转换 pass，再接上当前流水线。新增 pass 的步骤见 [ADDING_PASS.md](ADDING_PASS.md)。
