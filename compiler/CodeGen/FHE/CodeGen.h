#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace crossfhe {
mlir::LogicalResult validateEvaluator(mlir::ModuleOp module, llvm::StringRef backend);
mlir::LogicalResult generateCpp(mlir::ModuleOp module, llvm::StringRef backend,
                                llvm::raw_ostream &output,
                                const std::vector<std::string> *schedule = nullptr);
mlir::LogicalResult emitDag(mlir::ModuleOp module, llvm::StringRef backend,
                            llvm::raw_ostream &output);
}
