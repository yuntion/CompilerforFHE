#include "CodeGen/FHE/CodeGen.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
    llvm::cl::desc("<FHE MLIR>"), llvm::cl::Required);
static llvm::cl::opt<std::string> outputFile("o",
    llvm::cl::desc("Generated C++ header"), llvm::cl::Required);
static llvm::cl::opt<std::string> backend("backend",
    llvm::cl::desc("seal-ckks or openfhe-cross"), llvm::cl::init("openfhe-cross"));
static llvm::cl::opt<bool> emitContract("emit-contract",
    llvm::cl::desc("Emit the verified evaluator resource contract as JSON"));

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
  if (emitContract) {
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    llvm::json::Object contract;
    for (auto name : {"multiplicative_depth", "logic_comparisons", "openfhe_depth"})
      contract[name] = function->getAttrOfType<mlir::IntegerAttr>(std::string("ppml.")+name).getInt();
    contract["input_count"] = function.getNumArguments();
    contract["output_count"] = function.getNumResults();
    if (auto layout=function->getAttrOfType<mlir::StringAttr>("ppml.layout");
        layout && layout.getValue()=="periodic") {
      contract["layout"]=layout.getValue();
      contract["packed_period"]=mlir::cast<heir::RLWECipherType>(function.getArgument(0).getType()).getSize();
      for (auto name : {"input_elements","output_elements","matvec_baby_step","nonzero_diagonals"})
        if (auto attr=function->getAttrOfType<mlir::IntegerAttr>(std::string("ppml.")+name))
          contract[name]=attr.getInt();
      if (auto attr=function->getAttrOfType<mlir::StringAttr>("ppml.matvec_strategy"))
        contract["matvec_strategy"]=attr.getValue();
      std::set<int64_t> steps;
      function.walk([&](heir::PackedRotateOp op) { steps.insert(op->getAttrOfType<mlir::IntegerAttr>("step").getInt()); });
      llvm::json::Array keys;
      for (auto step : steps) keys.push_back(step);
      contract["rotation_steps"]=std::move(keys);
    }
    llvm::json::Object resources;
    if (auto attrs = function->getAttrOfType<mlir::DictionaryAttr>("ppml.resources"))
      for (auto attr : attrs) {
        auto value = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue());
        if (!value || value.getInt()<0) {
          function.emitError("resource counts must be nonnegative integers");
          return 1;
        }
        resources[attr.getName().strref()] = value.getInt();
      }
    contract["resources"] = std::move(resources);
    for (auto name : {"relu_input_bound", "relu_local_error_bound"})
      if (auto attr = function->getAttrOfType<mlir::FloatAttr>(std::string("ppml.")+name))
        contract[name] = attr.getValueAsDouble();
    if (auto attr = function->getAttrOfType<mlir::IntegerAttr>("ppml.relu_approximation_count"))
      contract["relu_approximation_count"] = attr.getInt();
    output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(contract)));
  } else output << code;
  return 0;
}
