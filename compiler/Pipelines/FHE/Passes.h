#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace crossfhe {
std::unique_ptr<mlir::Pass> createLowerToSCFHEPass();
std::unique_ptr<mlir::Pass> createLowerToHEIRPass();
std::unique_ptr<mlir::Pass> createOptimizationHookPass();
std::unique_ptr<mlir::Pass> createApproximateReluPass();
std::unique_ptr<mlir::Pass> createMatVecPass();
void registerCrossFHEPasses();
}
