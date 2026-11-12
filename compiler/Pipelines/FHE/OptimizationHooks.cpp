#include "Pipelines/FHE/Passes.h"
#include "mlir/IR/BuiltinOps.h"

namespace crossfhe {
namespace {
struct OptimizationHook
    : mlir::PassWrapper<OptimizationHook, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizationHook)
  llvm::StringRef getArgument() const final { return "ppml-fhe-optimize"; }
  llvm::StringRef getDescription() const final {
    return "Reserved PPML FHE optimization hook (identity)";
  }
  void runOnOperation() final {
    // Add scheme/layout/nonlinear/maintenance passes here after the baseline.
  }
};
}
std::unique_ptr<mlir::Pass> createOptimizationHookPass() {
  return std::make_unique<OptimizationHook>();
}
void registerCrossFHEPasses() {
  mlir::registerPass([] { return createLowerToSCFHEPass(); });
  mlir::registerPass([] { return createLowerToHEIRPass(); });
  mlir::registerPass([] { return createOptimizationHookPass(); });
}
}
