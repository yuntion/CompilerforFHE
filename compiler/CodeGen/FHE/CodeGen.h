#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace crossfhe {
mlir::LogicalResult generateCpp(mlir::ModuleOp module, llvm::StringRef backend,
                                llvm::raw_ostream &output);
}
