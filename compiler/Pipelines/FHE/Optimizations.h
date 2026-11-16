#pragma once
#include "mlir/IR/BuiltinOps.h"

namespace crossfhe {
// Shared implementation for the independently registered passes and pipeline.
mlir::LogicalResult optimizeFHE(mlir::ModuleOp module, bool simplify,
                               bool cse, bool dce, bool balance = false);
void registerFHEOptimizationPasses();
}
