#include "CodeGen/FHE/CodeGen.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace mlir;
LogicalResult crossfhe::generateCpp(ModuleOp module, llvm::StringRef backend,
                                   llvm::raw_ostream &output) {
  if (backend != "seal-ckks" && backend != "openfhe-cross") {
    module.emitError("unsupported backend"); return failure();
  }
  if (failed(verify(module))) return failure();
  auto functions = module.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(functions)) {
    llvm::errs() << "expected exactly one evaluator function\n";
    return failure();
  }
  auto function = *functions.begin();
  if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
    function.emitError("expected a one-block encrypted evaluator");
    return failure();
  }
  auto depth = function->getAttrOfType<IntegerAttr>("ppml.multiplicative_depth");
  if (!depth) {
    function.emitError("missing depth contract; run --ppml-to-heir first");
    return failure();
  }
  auto logic = function->getAttrOfType<IntegerAttr>("ppml.logic_comparisons");
  auto openfheDepth = function->getAttrOfType<IntegerAttr>("ppml.openfhe_depth");
  if (!logic || !openfheDepth || depth.getInt()<0 || openfheDepth.getInt()>128) {
    function.emitError("invalid depth/scheme contract"); return failure();
  }
  if (backend == "seal-ckks" && logic.getInt()) {
    function.emitError("SEAL has no logic FHE or scheme switching; use openfhe-cross"); return failure();
  }
  for (Type type : function.getFunctionType().getInputs())
    if (!isa<heir::RLWECipherType>(type)) { function.emitError("expected RLWE inputs"); return failure(); }
  for (Type type : function.getFunctionType().getResults())
    if (!isa<heir::RLWECipherType>(type)) { function.emitError("expected RLWE outputs"); return failure(); }
  std::ostringstream code;
  code << "#pragma once\n#include \"" << (backend == "seal-ckks" ? "seal_runtime.hpp" : "openfhe_runtime.hpp")
       << "\"\n#include <vector>\nnamespace crossfhe_generated {\n";
  code << "inline constexpr std::size_t input_count = " << function.getNumArguments() << ";\n";
  code << "inline constexpr std::size_t output_count = " << function.getNumResults() << ";\n";
  code << "inline constexpr unsigned depth = " << depth.getInt() << ";\n";
  code << "inline constexpr unsigned openfhe_depth = " << openfheDepth.getInt() << ";\n";
  code << "inline constexpr bool requires_logic = " << (logic.getInt() ? "true" : "false") << ";\n";
  code << "inline std::vector<crossfhe::Cipher> evaluate("
          "crossfhe::Runtime &runtime, "
          "const std::vector<crossfhe::Cipher> &inputs) {\n"
          "  if (inputs.size() != input_count) throw std::runtime_error(\"input count mismatch\");\n";
  llvm::DenseMap<Value, std::string> names;
  unsigned counter = 0;
  for (auto argument : function.getArguments())
    names[argument] = "inputs.at(" + std::to_string(counter++) + ")";
  for (Operation &op : function.getBody().front()) {
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      code << "  return {";
      for (unsigned i = 0; i < ret.getNumOperands(); ++i) {
        if (i) code << ", ";
        code << names.lookup(ret.getOperand(i));
      }
      code << "};\n";
      continue;
    }
    if (op.getNumResults() != 1) {
      op.emitError("only single-result encrypted primitives are supported");
      return failure();
    }
    std::string name = "v" + std::to_string(counter++);
    names[op.getResult(0)] = name;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto attr = dyn_cast<FloatAttr>(constant.getValue());
      if (!attr || !constant.getType().isF64() || !std::isfinite(attr.getValueAsDouble())) {
        op.emitError("expected an f64 public constant");
        return failure();
      }
      code << "  const double " << name << " = "
           << std::setprecision(17) << attr.getValueAsDouble() << ";\n";
      continue;
    }
    StringRef operation = op.getName().getStringRef();
    for (Value operand : op.getOperands())
      if (!names.count(operand)) { op.emitError("undefined codegen operand"); return failure(); }
    if (operation == "heir.materialize") {
      if (op.getNumOperands() != 1 || !op.getOperand(0).getType().isF64()) {
        op.emitError("only public-constant materialization is supported");
        return failure();
      }
      code << "  auto " << name << " = runtime.constant("
           << names.lookup(op.getOperand(0)) << ");\n";
      continue;
    }
    if (operation == "heir.depack" || operation == "heir.lut_lt" || operation == "heir.repack") {
      if (backend != "openfhe-cross" || op.getNumOperands()!=1) {
        op.emitError("switching operation requires openfhe-cross"); return failure();
      }
      bool inputRLWE = isa<heir::RLWECipherType>(op.getOperand(0).getType());
      bool outputRLWE = isa<heir::RLWECipherType>(op.getResult(0).getType());
      bool inputLWE = isa<heir::LWECipherVectorType>(op.getOperand(0).getType());
      bool outputLWE = isa<heir::LWECipherVectorType>(op.getResult(0).getType());
      StringRef method;
      if (operation == "heir.depack" && inputRLWE && outputLWE) method = "to_logic";
      else if (operation == "heir.lut_lt" && inputLWE && outputLWE) method = "sign";
      else if (operation == "heir.repack" && inputLWE && outputRLWE &&
               op.getOperand(0).getDefiningOp() &&
               op.getOperand(0).getDefiningOp()->getName().getStringRef()=="heir.lut_lt") method = "to_ckks_mask";
      else { op.emitError("invalid switching type or sign encoding"); return failure(); }
      code << "  auto " << name << " = runtime." << method.str() << "("
           << names.lookup(op.getOperand(0)) << ");\n";
      continue;
    }
    if (operation == "heir.select") {
      if (backend != "openfhe-cross" || op.getNumOperands()!=3 ||
          !isa<heir::RLWECipherType>(op.getOperand(0).getType())) {
        op.emitError("select requires an encrypted CKKS mask from logic FHE"); return failure();
      }
      code << "  auto " << name << " = runtime.select(" << names.lookup(op.getOperand(0))
           << ", " << names.lookup(op.getOperand(1)) << ", " << names.lookup(op.getOperand(2)) << ");\n";
      continue;
    }
    std::string method;
    if (operation == "heir.rlweadd") method = "add";
    else if (operation == "heir.rlwesub") method = "sub";
    else if (operation == "heir.rlwemul") method = "mul";
    else {
      op.emitError("operation has no implemented backend capability");
      return failure();
    }
    if (op.getNumOperands() != 2 ||
        !isa<heir::RLWECipherType>(op.getResult(0).getType())) {
      op.emitError("expected a binary RLWE operation");
      return failure();
    }
    for (Value operand : op.getOperands())
      if (!operand.getType().isF64() &&
          !isa<heir::RLWECipherType>(operand.getType())) {
        op.emitError("CKKS arithmetic cannot consume LWE or an unsupported plaintext type");
        return failure();
      }
    code << "  auto " << name << " = runtime." << method << "("
         << names.lookup(op.getOperand(0)) << ", "
         << names.lookup(op.getOperand(1)) << ");\n";
  }
  code << "}\n}\n";
  output << code.str();
  return success();
}
