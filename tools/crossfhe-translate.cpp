#include "CodeGen/FHE/CodeGen.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
    llvm::cl::desc("<FHE MLIR>"), llvm::cl::Required);
static llvm::cl::opt<std::string> outputFile("o",
    llvm::cl::desc("Generated C++ header"), llvm::cl::Required);
static llvm::cl::opt<std::string> backend("backend",
    llvm::cl::desc("seal-ckks or openfhe-cross"), llvm::cl::init("openfhe-cross"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "CrossFHE C++ code generation\n");
  mlir::DialectRegistry registry;
  registry.insert<heir::HEIRDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &context);
  if (!module) return 1;
  std::string code;
  llvm::raw_string_ostream generated(code);
  if (mlir::failed(crossfhe::generateCpp(*module, backend, generated))) return 1;
  std::error_code error;
  llvm::raw_fd_ostream output(outputFile, error, llvm::sys::fs::OF_Text);
  if (error) { llvm::errs() << error.message() << "\n"; return 1; }
  output << code;
  return 0;
}
