#include "Pipelines/FHE/Optimizations.h"
#include "Pipelines/FHE/Passes.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <cmath>
#include <queue>

using namespace mlir;
namespace crossfhe {
namespace {
bool enabled(func::FuncOp function) {
  auto flag = function->getAttrOfType<BoolAttr>("ppml.fhe_enabled");
  return flag && flag.getValue();
}
bool cipher(Type type) { return isa<heir::RLWECipherType>(type); }
bool primitive(Operation &op) {
  return llvm::StringSwitch<bool>(op.getName().getStringRef())
      .Cases("heir.rlweadd", "heir.rlwesub", "heir.rlwemul", true)
      .Cases("heir.depack", "heir.lut_lt", "heir.repack", true)
      .Cases("heir.select", "heir.materialize", true)
      .Cases("heir.packed_rotate", "heir.packed_mul_plain", "heir.packed_add_plain", true)
      .Cases("heir.packed_add", "heir.packed_zero", true).Default(false);
}

// This models the existing scalar runtimes' level contract, not CKKS noise.
// Switch overhead remains the conservative 17-level budget used by lowering.
LogicalResult refresh(func::FuncOp function) {
  if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody()))
    return function.emitError("FHE optimization requires a one-block evaluator");
  for (Type type : function.getFunctionType().getInputs())
    if (!cipher(type)) return function.emitError("FHE optimization expects RLWE inputs");
  llvm::DenseMap<Value, unsigned> depths;
  unsigned maximum = 0, logic = 0;
  unsigned ctMul = 0, ptMul = 0, adds = 0, subs = 0, selects = 0;
  unsigned depacks = 0, repacks = 0, materializations = 0, operations = 0;
  unsigned rotations = 0, plaintextAdds = 0;
  for (Operation &op : function.getBody().front()) {
    if (isa<func::ReturnOp>(op)) continue;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto value=dyn_cast<FloatAttr>(constant.getValue());
      if (!constant.getType().isF64() || !value || !std::isfinite(value.getValueAsDouble()))
        return op.emitError("FHE optimization supports f64 public constants only");
      continue;
    }
    if (!primitive(op) || op.getNumRegions() || op.getNumResults() != 1)
      return op.emitError("operation is outside the scalar FHE optimization contract");
    unsigned depth = 0;
    for (Value value : op.getOperands()) depth = std::max(depth, depths.lookup(value));
    auto name = op.getName().getStringRef();
    if (name == "heir.packed_mul_plain") { ++depth; ++ptMul; }
    else if (name == "heir.packed_rotate") ++rotations;
    else if (name == "heir.packed_add") ++adds;
    else if (name == "heir.packed_add_plain") ++plaintextAdds;
    else if (name == "heir.packed_zero") ++materializations;
    else if (name == "heir.rlwemul") {
      if (op.getNumOperands()!=2 || !cipher(op.getResult(0).getType()) ||
          (!cipher(op.getOperand(0).getType()) && !cipher(op.getOperand(1).getType())))
        return op.emitError("expected ciphertext/ciphertext or ciphertext/plaintext multiplication");
      ++depth;
      if (cipher(op.getOperand(0).getType()) && cipher(op.getOperand(1).getType())) ++ctMul;
      else ++ptMul;
    } else if (name == "heir.select") { ++depth; ++selects; }
    else if (name == "heir.lut_lt") ++logic;
    else if (name == "heir.depack") ++depacks;
    else if (name == "heir.repack") ++repacks;
    else if (name == "heir.rlweadd") ++adds;
    else if (name == "heir.rlwesub") ++subs;
    else if (name == "heir.materialize") { depth = 0; ++materializations; }
    depths[op.getResult(0)] = depth;
    maximum = std::max(maximum, depth);
    ++operations;
  }
  OpBuilder builder(function);
  function->setAttr("ppml.multiplicative_depth", builder.getI64IntegerAttr(maximum));
  function->setAttr("ppml.logic_comparisons", builder.getI64IntegerAttr(logic));
  function->setAttr("ppml.openfhe_depth", builder.getI64IntegerAttr(maximum+(logic ? 17 : 0)));
  NamedAttrList resources;
  auto put = [&](StringRef name, unsigned value) { resources.set(name,builder.getI64IntegerAttr(value)); };
  put("operations",operations); put("ciphertext_multiplications",ctMul);
  put("plaintext_multiplications",ptMul); put("additions",adds); put("subtractions",subs);
  put("selects",selects); put("ckks_to_logic",depacks); put("logic_bootstraps",logic);
  put("logic_to_ckks",repacks); put("materializations",materializations);
  put("rotations",rotations); put("plaintext_additions",plaintextAdds);
  function->setAttr("ppml.resources",resources.getDictionary(function.getContext()));
  // Never replace unanalysed noise attributes with invented noise bounds.
  return success();
}

bool number(Value value, double expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto attr = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  return attr && value.getType().isF64() && attr.getValueAsDouble()==expected;
}
void simplify(func::FuncOp function) {
  for (Operation &op : llvm::make_early_inc_range(function.getBody().front())) {
    if (!primitive(op)) continue;
    StringRef name = op.getName().getStringRef();
    Value replacement;
    if (op.getNumOperands()==2) {
      Value a = op.getOperand(0), b = op.getOperand(1);
      if ((name=="heir.rlweadd" && number(b,0)) ||
          (name=="heir.rlwesub" && number(b,0)) ||
          (name=="heir.rlwemul" && number(b,1))) replacement=a;
      else if ((name=="heir.rlweadd" && number(a,0)) ||
               (name=="heir.rlwemul" && number(a,1))) replacement=b;
      else if (name=="heir.rlwemul" && (number(a,0) || number(b,0))) {
        OpBuilder builder(&op);
        OperationState state(op.getLoc(),"heir.materialize");
        state.addOperands(number(a,0) ? a : b); state.addTypes(op.getResult(0).getType());
        replacement=builder.create(state)->getResult(0);
      }
    } else if (name=="heir.select" && op.getNumOperands()==3 &&
               op.getOperand(1)==op.getOperand(2)) {
      replacement=op.getOperand(1);
      if (replacement.getType().isF64()) {
        OpBuilder builder(&op);
        OperationState state(op.getLoc(),"heir.materialize");
        state.addOperands(replacement); state.addTypes(op.getResult(0).getType());
        replacement=builder.create(state)->getResult(0);
      }
    }
    if (replacement && replacement.getType()==op.getResult(0).getType()) {
      op.getResult(0).replaceAllUsesWith(replacement);
      op.erase();
    }
  }
}

NamedAttrList semanticAttributes(Operation &op) {
  NamedAttrList attrs(op.getAttrs());
  // Source attribution is diagnostic, not a different ciphertext computation.
  attrs.erase("ppml.source");
  return attrs;
}
void cse(func::FuncOp function) {
  llvm::DenseMap<size_t, SmallVector<Operation *>> available;
  for (Operation &op : llvm::make_early_inc_range(function.getBody().front())) {
    if ((!primitive(op) && !isa<arith::ConstantOp>(op)) ||
        op.getName().getStringRef()=="heir.materialize" ||
        op.getName().getStringRef()=="heir.packed_zero") continue;
    auto attrs=semanticAttributes(op).getDictionary(function.getContext());
    auto hash=llvm::hash_combine(op.getName().getStringRef(),attrs);
    for (Value operand : op.getOperands()) hash=llvm::hash_combine(hash,operand);
    for (Type type : op.getResultTypes()) hash=llvm::hash_combine(hash,type);
    auto &bucket=available[static_cast<size_t>(hash)];
    Operation *match=nullptr;
    for (Operation *candidate : bucket) {
      if (candidate->getName()==op.getName() &&
          candidate->getOperands()==op.getOperands() &&
          candidate->getResultTypes()==op.getResultTypes() &&
          semanticAttributes(*candidate).getDictionary(function.getContext())==attrs) {
        match=candidate; break;
      }
    }
    if (match) { op.getResult(0).replaceAllUsesWith(match->getResult(0)); op.erase(); }
    else bucket.push_back(&op);
  }
}
void dce(func::FuncOp function) {
  // Reverse topological order removes an entire unused switching/arithmetic
  // chain in one walk; unknown operations are deliberately never considered pure.
  Operation *op=&function.getBody().front().back();
  while (op) {
    Operation *previous=op->getPrevNode();
    if ((primitive(*op) || isa<arith::ConstantOp>(op)) && op->use_empty()) op->erase();
    op=previous;
  }
}

void balance(func::FuncOp function) {
  llvm::DenseMap<Value,unsigned> depths;
  SmallVector<Operation *> roots;
  for (Operation &op : function.getBody().front()) {
    unsigned depth=0;
    for (Value value : op.getOperands()) depth=std::max(depth,depths.lookup(value));
    if (op.getName().getStringRef()=="heir.rlwemul" || op.getName().getStringRef()=="heir.select") ++depth;
    for (Value value : op.getResults()) depths[value]=depth;
    if (op.getName().getStringRef()=="heir.rlwemul") roots.push_back(&op);
  }
  // Work from the last operation so internal nodes are not rewritten first.
  for (Operation *root : llvm::reverse(roots)) {
    if (root->use_empty() || (root->getResult(0).hasOneUse() &&
        root->getResult(0).user_begin()->getName().getStringRef()=="heir.rlwemul")) continue;
    if (root->getNumOperands()!=2 || !cipher(root->getOperand(0).getType()) ||
        !cipher(root->getOperand(1).getType())) continue;
    SmallVector<Value> pending{root->getOperand(1),root->getOperand(0)}, leaves;
    while (!pending.empty()) {
      Value value=pending.pop_back_val();
      Operation *def=value.getDefiningOp();
      if (def && def->getName().getStringRef()=="heir.rlwemul" && value.hasOneUse() &&
          def->getNumOperands()==2 && cipher(def->getOperand(0).getType()) &&
          cipher(def->getOperand(1).getType())) {
        pending.push_back(def->getOperand(1)); pending.push_back(def->getOperand(0));
      } else leaves.push_back(value);
    }
    if (leaves.size()<3) continue;
    std::priority_queue<unsigned,std::vector<unsigned>,std::greater<unsigned>> trial;
    for (Value value : leaves) trial.push(depths.lookup(value));
    while (trial.size()>1) { unsigned a=trial.top(); trial.pop(); unsigned b=trial.top(); trial.pop(); trial.push(std::max(a,b)+1); }
    if (trial.top()>=depths.lookup(root->getResult(0))) continue;
    struct Node { unsigned depth; size_t order; Value value; };
    auto later=[](const Node &a,const Node &b) { return a.depth!=b.depth ? a.depth>b.depth : a.order>b.order; };
    std::priority_queue<Node,std::vector<Node>,decltype(later)> heap(later);
    size_t order=0;
    for (Value value : leaves) heap.push({depths.lookup(value),order++,value});
    OpBuilder builder(root);
    while (heap.size()>1) {
      Node a=heap.top();heap.pop();Node b=heap.top();heap.pop();
      OperationState state(root->getLoc(),"heir.rlwemul");
      state.addOperands({a.value,b.value});state.addTypes(root->getResult(0).getType());
      state.addAttributes(root->getAttrs());
      Value value=builder.create(state)->getResult(0);
      depths[value]=std::max(a.depth,b.depth)+1;
      heap.push({depths[value],order++,value});
    }
    root->getResult(0).replaceAllUsesWith(heap.top().value);
    // Keep root alive until this traversal ends: roots contains its internal
    // nodes. DCE below erases the obsolete tree after all roots are visited.
    function->setAttr("ppml.reassociated",builder.getBoolAttr(true));
  }
}

template<int Kind> struct ScalarOptimization
    : PassWrapper<ScalarOptimization<Kind>,OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScalarOptimization)
  StringRef getArgument() const final {
    if constexpr (Kind==0) return "ppml-fhe-simplify";
    if constexpr (Kind==1) return "ppml-fhe-cse";
    if constexpr (Kind==2) return "ppml-fhe-dce";
    if constexpr (Kind==3) return "ppml-fhe-refresh-contract";
    return "ppml-fhe-balance-mul";
  }
  StringRef getDescription() const final {
    if constexpr (Kind==0) return "Simplify scalar FHE identities and redundant selects";
    if constexpr (Kind==1) return "Share identical scalar FHE computations and switching chains";
    if constexpr (Kind==2) return "Eliminate unused scalar ciphertext computations";
    if constexpr (Kind==3) return "Recompute scalar FHE depth and operation resource contracts";
    return "Opt in to reassociation of scalar ciphertext multiplication trees";
  }
  void runOnOperation() final {
    if (failed(optimizeFHE(this->getOperation(),Kind==0,Kind==1,Kind==2,Kind==4))) this->signalPassFailure();
  }
};
}
LogicalResult optimizeFHE(ModuleOp module,bool doSimplify,bool doCSE,bool doDCE,bool doBalance) {
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!enabled(function)) continue;
    if (failed(refresh(function))) return failure();
    if (doSimplify) simplify(function);
    if (doCSE) cse(function);
    if (doDCE) dce(function);
    if (doBalance) { balance(function); cse(function); dce(function); }
    if (failed(refresh(function))) return failure();
  }
  return success();
}
void registerFHEOptimizationPasses() {
  registerPass([] { return std::make_unique<ScalarOptimization<0>>(); });
  registerPass([] { return std::make_unique<ScalarOptimization<1>>(); });
  registerPass([] { return std::make_unique<ScalarOptimization<2>>(); });
  registerPass([] { return std::make_unique<ScalarOptimization<3>>(); });
  registerPass([] { return std::make_unique<ScalarOptimization<4>>(); });
}
}
