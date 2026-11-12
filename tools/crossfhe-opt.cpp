#include "Pipelines/FHE/Passes.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "SCFHEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  heir::HEIRDialect, mlir::libra::scfhe::SCFHEDialect>();
  crossfhe::registerCrossFHEPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CrossFHE compiler\n", registry));
}
