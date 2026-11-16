#include "Pipelines/FHE/Passes.h"
#include "Pipelines/FHE/Optimizations.h"
#include "mlir/IR/BuiltinOps.h"

namespace crossfhe {
namespace {
struct OptimizationHook
    : mlir::PassWrapper<OptimizationHook, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizationHook)
  llvm::StringRef getArgument() const final { return "ppml-fhe-optimize"; }
  llvm::StringRef getDescription() const final {
    return "Simplify, share and prune marked scalar FHE computations, then refresh resource contracts";
  }
  void runOnOperation() final {
    if (mlir::failed(optimizeFHE(getOperation(), true, true, true)))
      signalPassFailure();
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
  mlir::registerPass([] { return createApproximateReluPass(); });
  mlir::registerPass([] { return createMatVecPass(); });
  registerFHEOptimizationPasses();
}
}
