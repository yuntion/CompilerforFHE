#include "CodeGen/FHE/CodeGen.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <sstream>
#include <cmath>
#include <set>
#include <map>
#include <vector>

using namespace mlir;
namespace {
// IDs and constraints are always derived from the original verified block. A
// candidate supplies only a permutation; it never supplies operations or edges.
struct EvaluatorDag {
  std::vector<Operation *> nodes;
  llvm::DenseMap<Value, std::string> ids;
  std::vector<std::set<unsigned>> dependencies;
  std::vector<bool> barriers;
};

EvaluatorDag buildDag(func::FuncOp function, llvm::StringRef backend) {
  EvaluatorDag dag;
  unsigned argument = 0;
  for (Value value : function.getArguments())
    dag.ids[value] = "arg" + std::to_string(argument++);
  llvm::DenseMap<Operation *, unsigned> positions;
  for (Operation &op : function.getBody().front()) {
    if (isa<func::ReturnOp>(op)) continue;
    unsigned index = dag.nodes.size();
    positions[&op] = index;
    dag.nodes.push_back(&op);
    dag.ids[op.getResult(0)] = "n" + std::to_string(index);
  }
  int lastBarrier = -1;
  for (unsigned index = 0; index < dag.nodes.size(); ++index) {
    Operation *op = dag.nodes[index];
    auto name = op->getName().getStringRef();
    bool barrier = backend == "openfhe-cross" || name == "heir.materialize" ||
                   name == "heir.packed_zero" || name == "heir.depack" ||
                   name == "heir.lut_lt" || name == "heir.repack";
    // The CPU/GPU adapters materialize a fresh public-key zero for these
    // scalar corner cases, even though they appear as arithmetic in the IR.
    if (name == "heir.rlwesub" && op->getNumOperands() == 2 &&
        op->getOperand(0) == op->getOperand(1))
      barrier = true;
    if (name == "heir.rlwemul")
      for (Value operand : op->getOperands())
        if (auto constant = operand.getDefiningOp<arith::ConstantOp>())
          if (auto value = dyn_cast<FloatAttr>(constant.getValue());
              value && value.getValueAsDouble() == 0.0)
            barrier = true;
    std::set<unsigned> dependencies;
    for (Value operand : op->getOperands())
      if (auto *definition = operand.getDefiningOp())
        dependencies.insert(positions.lookup(definition));
    // The previous barrier already covers its entire prefix. Only add edges
    // for the current segment, keeping conservative ordering linear in graph
    // size even when every OpenFHE operation is a barrier.
    if (barrier)
      for (unsigned previous = lastBarrier < 0 ? 0 : static_cast<unsigned>(lastBarrier);
           previous < index; ++previous)
        dependencies.insert(previous);
    else if (lastBarrier >= 0)
      dependencies.insert(static_cast<unsigned>(lastBarrier));
    dag.dependencies.push_back(std::move(dependencies));
    dag.barriers.push_back(barrier);
    if (barrier) lastBarrier = index;
  }
  return dag;
}

LogicalResult orderNodes(func::FuncOp function, const EvaluatorDag &dag,
                         const std::vector<std::string> *schedule,
                         std::vector<Operation *> &ordered) {
  if (!schedule) {
    ordered = dag.nodes;
  } else {
    if (schedule->size() != dag.nodes.size()) {
      function.emitError("schedule must contain every node exactly once");
      return failure();
    }
    std::map<std::string, unsigned> positions;
    for (unsigned index = 0; index < dag.nodes.size(); ++index)
      positions.emplace("n" + std::to_string(index), index);
    std::set<unsigned> visited;
    for (const std::string &id : *schedule) {
      auto found = positions.find(id);
      if (found == positions.end()) {
        function.emitError("schedule contains unknown node: ") << id;
        return failure();
      }
      unsigned index = found->second;
      if (!visited.insert(index).second) {
        function.emitError("schedule contains duplicate node: ") << id;
        return failure();
      }
      for (unsigned dependency : dag.dependencies[index])
        if (!visited.count(dependency)) {
          function.emitError("schedule violates data/effect dependency: ")
              << id << " requires n" << dependency;
          return failure();
        }
      ordered.push_back(dag.nodes[index]);
    }
  }
  ordered.push_back(function.getBody().front().getTerminator());
  return success();
}

std::string typeString(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}
struct EvaluatorInfo {
  func::FuncOp function;
  IntegerAttr depth, logic, openfheDepth;
  bool packed = false;
  unsigned period = 0, inputElements = 0, outputElements = 0;
  std::set<int64_t> rotations;
};

StringRef runtimeMethod(StringRef operation) {
  return llvm::StringSwitch<StringRef>(operation)
      .Case("heir.packed_rotate", "rotate")
      .Case("heir.packed_zero", "zero_like")
      .Case("heir.packed_add", "add")
      .Case("heir.packed_mul_plain", "mul_periodic")
      .Case("heir.packed_add_plain", "add_periodic")
      .Case("heir.materialize", "constant")
      .Case("heir.depack", "to_logic")
      .Case("heir.lut_lt", "sign")
      .Case("heir.repack", "to_ckks_mask")
      .Case("heir.select", "select")
      .Case("heir.rlweadd", "add")
      .Case("heir.rlwesub", "sub")
      .Case("heir.rlwemul", "mul")
      .Default("");
}

LogicalResult validatePrimitives(func::FuncOp function, StringRef backend, bool packed) {
  // MLIR verification owns SSA dominance and the registered operations' arity.
  // This boundary checks the narrower capabilities of the runtime adapters.
  for (Operation &op : function.getBody().front()) {
    if (isa<func::ReturnOp>(op)) continue;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto attr = dyn_cast<FloatAttr>(constant.getValue());
      if (!attr || !constant.getType().isF64() || !std::isfinite(attr.getValueAsDouble())) {
        op.emitError("expected an f64 public constant");
        return failure();
      }
      continue;
    }
    StringRef operation = op.getName().getStringRef();
    if (runtimeMethod(operation).empty()) {
      op.emitError(packed ? "unsupported packed primitive"
                          : "operation has no implemented backend capability");
      return failure();
    }
    if (packed) continue;
    if (operation == "heir.materialize") {
      if (!op.getOperand(0).getType().isF64()) {
        op.emitError("only public-constant materialization is supported");
        return failure();
      }
      continue;
    }
    if (operation == "heir.depack" || operation == "heir.lut_lt" || operation == "heir.repack") {
      if (backend != "openfhe-cross") {
        op.emitError("switching operation requires openfhe-cross"); return failure();
      }
      bool inputRLWE = isa<heir::RLWECipherType>(op.getOperand(0).getType());
      bool outputRLWE = isa<heir::RLWECipherType>(op.getResult(0).getType());
      bool inputLWE = isa<heir::LWECipherVectorType>(op.getOperand(0).getType());
      bool outputLWE = isa<heir::LWECipherVectorType>(op.getResult(0).getType());
      bool valid = (operation == "heir.depack" && inputRLWE && outputLWE) ||
                   (operation == "heir.lut_lt" && inputLWE && outputLWE) ||
                   (operation == "heir.repack" && inputLWE && outputRLWE &&
                    op.getOperand(0).getDefiningOp() &&
                    op.getOperand(0).getDefiningOp()->getName().getStringRef()=="heir.lut_lt");
      if (!valid) { op.emitError("invalid switching type or sign encoding"); return failure(); }
      continue;
    }
    if (operation == "heir.select") {
      if (backend != "openfhe-cross" ||
          !isa<heir::RLWECipherType>(op.getOperand(0).getType())) {
        op.emitError("select requires an encrypted CKKS mask from logic FHE"); return failure();
      }
      continue;
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
  }
  return success();
}

LogicalResult validateEvaluatorImpl(ModuleOp module, StringRef backend,
                                    EvaluatorInfo &info) {
  if (backend != "seal-ckks" && backend != "openfhe-cross" && backend != "phantom-ckks") {
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
  if (!logic || !openfheDepth || depth.getInt()<0 || logic.getInt()<0 ||
      openfheDepth.getInt()<depth.getInt() || openfheDepth.getInt()>128) {
    function.emitError("invalid depth/scheme contract"); return failure();
  }
  if (backend != "openfhe-cross" && logic.getInt()) {
    function.emitError(backend == "seal-ckks"
        ? "SEAL has no logic FHE or scheme switching; use openfhe-cross"
        : "Phantom has no public logic FHE or scheme switching; use openfhe-cross"); return failure();
  }
  for (Type type : function.getFunctionType().getInputs())
    if (!isa<heir::RLWECipherType>(type)) { function.emitError("expected RLWE inputs"); return failure(); }
  for (Type type : function.getFunctionType().getResults())
    if (!isa<heir::RLWECipherType>(type)) { function.emitError("expected RLWE outputs"); return failure(); }
  auto layout = function->getAttrOfType<StringAttr>("ppml.layout");
  bool packed = layout && layout.getValue()=="periodic";
  unsigned period=0, inputElements=function.getNumArguments(), outputElements=function.getNumResults();
  std::set<int64_t> rotations;
  if (layout && !packed && layout.getValue()!="one-scalar-per-ciphertext") {
    function.emitError("unsupported CKKS layout"); return failure();
  }
  if (packed) {
    auto in=function->getAttrOfType<IntegerAttr>("ppml.input_elements");
    auto out=function->getAttrOfType<IntegerAttr>("ppml.output_elements");
    if (backend=="openfhe-cross" || function.getNumArguments()!=1 || function.getNumResults()!=1 || logic.getInt()) {
      function.emitError("periodic packed evaluators require seal-ckks or phantom-ckks, one input/output ciphertext and no logic"); return failure();
    }
    auto type=cast<heir::RLWECipherType>(function.getArgument(0).getType());
    int size=type.getSize();
    // Identity evaluators have no packed operation to verify their cipher type.
    if (!type.getPlaintextType().isF64() || size<1 || size>16384 || (size&(size-1)) ||
        function.getResultTypes()[0]!=type || !in || !out ||
        in.getInt()<1 || in.getInt()>size || out.getInt()<1 || out.getInt()>size) {
      function.emitError("invalid periodic packed shape contract"); return failure();
    }
    period=size; inputElements=in.getInt(); outputElements=out.getInt();
  } else {
    for (Type type : llvm::concat<const Type>(function.getArgumentTypes(),function.getResultTypes())) {
      auto cipher=cast<heir::RLWECipherType>(type);
      if (cipher.getSize()!=1 || !cipher.getPlaintextType().isF64()) {
        function.emitError("non-scalar cipher requires an explicit periodic layout contract"); return failure();
      }
    }
  }
  unsigned observedDepth=0;
  llvm::DenseMap<Value,unsigned> packedDepths;
  for (Operation &op : function.getBody().front()) {
    if (isa<func::ReturnOp>(op)) continue;
    StringRef name=op.getName().getStringRef();
    if (name.startswith("heir.packed_")!=packed) {
      op.emitError("cannot mix packed and scalar operations; lower MatVec first"); return failure();
    }
    if (!packed) continue;
    auto type=function.getArgument(0).getType();
    // PackedOps verifies matching operands/results; the function-level period
    // contract is the only remaining cross-operation constraint here.
    if (op.getResult(0).getType()!=type) {
      op.emitError("packed primitive does not match evaluator period"); return failure();
    }
    unsigned d=0;
    for (Value value : op.getOperands()) d=std::max(d,packedDepths.lookup(value));
    if (name=="heir.packed_mul_plain") ++d;
    if (name=="heir.packed_rotate") rotations.insert(op.getAttrOfType<IntegerAttr>("step").getInt());
    packedDepths[op.getResult(0)]=d; observedDepth=std::max(observedDepth,d);
  }
  if (packed && observedDepth!=depth.getInt()) {
    function.emitError("packed depth contract is stale; refresh it before codegen"); return failure();
  }
  if (failed(validatePrimitives(function, backend, packed))) return failure();
  info = {function, depth, logic, openfheDepth, packed, period,
          inputElements, outputElements, std::move(rotations)};
  return success();
}
} // namespace

LogicalResult crossfhe::validateEvaluator(ModuleOp module, llvm::StringRef backend) {
  EvaluatorInfo info;
  return validateEvaluatorImpl(module, backend, info);
}

LogicalResult crossfhe::generateCpp(ModuleOp module, llvm::StringRef backend,
                                   llvm::raw_ostream &output,
                                   const std::vector<std::string> *schedule) {
  EvaluatorInfo info;
  if (failed(validateEvaluatorImpl(module, backend, info))) return failure();
  auto function = info.function;
  std::vector<Operation *> ordered;
  EvaluatorDag dag;
  if (schedule) {
    dag = buildDag(function, backend);
    if (failed(orderNodes(function, dag, schedule, ordered))) return failure();
  } else {
    for (Operation &op : function.getBody().front()) ordered.push_back(&op);
  }
  std::vector<std::vector<Value>> releaseBefore(ordered.size());
  if (schedule) {
    llvm::DenseMap<Value, unsigned> lastUse;
    for (unsigned index = 0; index < ordered.size(); ++index) {
      for (Value result : ordered[index]->getResults()) lastUse[result] = index;
      for (Value operand : ordered[index]->getOperands()) lastUse[operand] = index;
    }
    for (Operation *op : dag.nodes) {
      Value value = op->getResult(0);
      if ((isa<heir::RLWECipherType>(value.getType()) ||
           isa<heir::LWECipherVectorType>(value.getType())) &&
          lastUse.lookup(value) + 1 < ordered.size())
        releaseBefore[lastUse.lookup(value) + 1].push_back(value);
    }
  }
  std::ostringstream code;
  code << "#pragma once\n#include \"" << (backend == "seal-ckks" ? "seal_runtime.hpp" :
                                         backend == "phantom-ckks" ? "phantom_runtime.hpp" : "openfhe_runtime.hpp")
       << "\"\n#include <vector>\nnamespace crossfhe_generated {\n";
  code << "inline constexpr std::size_t input_count = " << function.getNumArguments() << ";\n";
  code << "inline constexpr std::size_t output_count = " << function.getNumResults() << ";\n";
  if (backend!="openfhe-cross") {
    code << "inline constexpr std::size_t packed_period = " << info.period << ";\n";
    code << "inline constexpr std::size_t input_value_count = " << info.inputElements << ";\n";
    code << "inline constexpr std::size_t output_value_count = " << info.outputElements << ";\n";
    code << "inline const std::vector<int> rotation_steps = {";
    for (auto step : info.rotations) code << step << ",";
    code << "};\n";
  }
  code << "inline constexpr unsigned depth = " << info.depth.getInt() << ";\n";
  code << "inline constexpr unsigned openfhe_depth = " << info.openfheDepth.getInt() << ";\n";
  code << "inline constexpr bool requires_logic = " << (info.logic.getInt() ? "true" : "false") << ";\n";
  code << "inline std::vector<crossfhe::Cipher> evaluate("
          "crossfhe::Runtime &runtime, "
          "const std::vector<crossfhe::Cipher> &inputs) {\n"
          "  if (inputs.size() != input_count) throw std::runtime_error(\"input count mismatch\");\n";
  llvm::DenseMap<Value, std::string> names;
  unsigned counter = 0;
  for (auto argument : function.getArguments())
    names[argument] = "inputs.at(" + std::to_string(counter++) + ")";
  for (unsigned index = 0; index < ordered.size(); ++index) {
    for (Value value : releaseBefore[index])
      code << "  " << names.lookup(value) << " = {}; // release "
           << dag.ids.lookup(value) << " after last use\n";
    Operation &op = *ordered[index];
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      code << "  return {";
      for (unsigned i = 0; i < ret.getNumOperands(); ++i) {
        if (i) code << ", ";
        code << names.lookup(ret.getOperand(i));
      }
      code << "};\n";
      continue;
    }
    std::string name = "v" + std::to_string(counter++);
    names[op.getResult(0)] = name;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto attr = cast<FloatAttr>(constant.getValue());
      code << "  const double " << name << " = "
           << std::setprecision(17) << attr.getValueAsDouble() << ";\n";
      continue;
    }
    StringRef operation = op.getName().getStringRef();
    if (info.packed) {
      code << "  auto " << name << " = runtime.";
      if (operation=="heir.packed_rotate")
        code << "rotate(" << names.lookup(op.getOperand(0)) << ", "
             << op.getAttrOfType<IntegerAttr>("step").getInt() << ");\n";
      else if (operation=="heir.packed_zero")
        code << "zero_like(" << names.lookup(op.getOperand(0)) << ");\n";
      else if (operation=="heir.packed_add")
        code << "add(" << names.lookup(op.getOperand(0)) << ", " << names.lookup(op.getOperand(1)) << ");\n";
      else if (operation=="heir.packed_mul_plain" || operation=="heir.packed_add_plain") {
        code << (operation=="heir.packed_mul_plain" ? "mul_periodic(" : "add_periodic(")
             << names.lookup(op.getOperand(0)) << ", std::vector<double>{";
        for (double value : op.getAttrOfType<DenseFPElementsAttr>("values").getValues<double>())
          code << std::setprecision(17) << value << ",";
        code << "});\n";
      }
      continue;
    }
    if (operation == "heir.materialize") {
      code << "  auto " << name << " = runtime.constant("
           << names.lookup(op.getOperand(0)) << ");\n";
      continue;
    }
    if (operation == "heir.depack" || operation == "heir.lut_lt" || operation == "heir.repack") {
      code << "  auto " << name << " = runtime." << runtimeMethod(operation).str() << "("
           << names.lookup(op.getOperand(0)) << ");\n";
      continue;
    }
    if (operation == "heir.select") {
      code << "  auto " << name << " = runtime.select(" << names.lookup(op.getOperand(0))
           << ", " << names.lookup(op.getOperand(1)) << ", " << names.lookup(op.getOperand(2)) << ");\n";
      continue;
    }
    code << "  auto " << name << " = runtime." << runtimeMethod(operation).str() << "("
         << names.lookup(op.getOperand(0)) << ", "
         << names.lookup(op.getOperand(1)) << ");\n";
  }
  code << "}\n}\n";
  output << code.str();
  return success();
}

LogicalResult crossfhe::emitDag(ModuleOp module, llvm::StringRef backend,
                               llvm::raw_ostream &output) {
  EvaluatorInfo info;
  if (failed(validateEvaluatorImpl(module, backend, info))) return failure();
  auto function = info.function;
  auto dag = buildDag(function, backend);
  llvm::json::Array inputs, nodes, outputs, referenceSchedule;
  for (Value argument : function.getArguments())
    inputs.push_back(llvm::json::Object{{"id", dag.ids.lookup(argument)},
                                      {"type", typeString(argument.getType())}});
  for (unsigned index = 0; index < dag.nodes.size(); ++index) {
    Operation *op = dag.nodes[index];
    llvm::json::Array operands, dependencies;
    for (Value operand : op->getOperands()) operands.push_back(dag.ids.lookup(operand));
    for (unsigned dependency : dag.dependencies[index])
      dependencies.push_back("n" + std::to_string(dependency));
    std::string attributes;
    llvm::raw_string_ostream stream(attributes);
    op->getAttrDictionary().print(stream);
    auto id = "n" + std::to_string(index);
    nodes.push_back(llvm::json::Object{
        {"id", id}, {"op", op->getName().getStringRef()},
        {"operands", std::move(operands)}, {"dependencies", std::move(dependencies)},
        {"result_type", typeString(op->getResult(0).getType())},
        {"attributes", stream.str()}, {"effect", dag.barriers[index] ? "barrier" : "pure"}});
    referenceSchedule.push_back(id);
  }
  for (Value value : function.getBody().front().getTerminator()->getOperands())
    outputs.push_back(dag.ids.lookup(value));
  llvm::json::Object result{{"schema_version", 1}, {"backend", backend},
                           {"inputs", std::move(inputs)}, {"nodes", std::move(nodes)},
                           {"outputs", std::move(outputs)},
                           {"reference_schedule", std::move(referenceSchedule)}};
  output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(result)));
  return success();
}
