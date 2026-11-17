#include "Pipelines/FHE/Passes.h"
#include "Pipelines/FHE/Optimizations.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <map>
#include <set>
#include <vector>

using namespace mlir;
namespace crossfhe {
namespace {
struct MatVecPass : PassWrapper<MatVecPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatVecPass)
  MatVecPass() = default;
  MatVecPass(const MatVecPass &other) : PassWrapper(other) {}
  Option<std::string> strategy{*this,"strategy",llvm::cl::desc("diagonal, bsgs or auto"),llvm::cl::init("auto")};
  StringRef getArgument() const final { return "ppml-fhe-matvec"; }
  StringRef getDescription() const final {
    return "Lower a public affine MatVec to periodic CPU CKKS SIMD diagonal/BSGS primitives";
  }
  void runOnOperation() final {
    if (strategy!="auto" && strategy!="diagonal" && strategy!="bsgs") {
      getOperation().emitError("MatVec strategy must be diagonal, bsgs or auto");
      return signalPassFailure();
    }
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      auto enabled = function->getAttrOfType<BoolAttr>("ppml.fhe_enabled");
      if (!enabled || !enabled.getValue()) continue;
      if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        function.emitError("MatVec requires a one-block evaluator"); return signalPassFailure();
      }
      SmallVector<heir::MatVecOp> ops(function.getOps<heir::MatVecOp>());
      if (ops.empty()) continue;
      if (ops.size()!=1 || function.getNumArguments()!=1 || function.getNumResults()!=1 ||
          function.getBody().front().getOperations().size()!=2 ||
          ops[0].getInput()!=function.getArgument(0)) {
        function.emitError("packed path supports exactly one affine MatVec and its return");
        return signalPassFailure();
      }
      auto ret = dyn_cast<func::ReturnOp>(function.getBody().front().back());
      if (!ret || ret.getOperand(0)!=ops[0].getOutput()) {
        function.emitError("MatVec result must be returned directly"); return signalPassFailure();
      }
      lower(function,ops[0]);
    }
    if (failed(optimizeFHE(getOperation(),false,false,false))) signalPassFailure();
  }
  void lower(func::FuncOp function, heir::MatVecOp op) {
    auto type = cast<heir::RLWECipherType>(op.getOutput().getType());
    int period = type.getSize();
    auto matrix = op->getAttrOfType<DenseFPElementsAttr>("matrix");
    int rows = matrix.getType().getShape()[0], cols = matrix.getType().getShape()[1];
    std::vector<double> weights(matrix.getValues<double>().begin(),matrix.getValues<double>().end());
    std::map<int,std::vector<double>> diagonals;
    // D_k[r] = A[r,(r+k) mod L], padded to L x L.
    for (int row=0; row<rows; ++row) for (int col=0; col<cols; ++col) {
      double value = weights[row*cols+col];
      if (value==0) continue;
      int k = (col-row+period)%period;
      if (!diagonals.count(k)) diagonals[k].resize(period,0);
      diagonals[k][row] = value;
    }
    auto cost = [&](int baby) {
      std::set<int> babies, giants;
      for (auto &entry : diagonals) {
        if (entry.first % baby) babies.insert(entry.first % baby);
        if (entry.first / baby) giants.insert(entry.first / baby);
      }
      return babies.size()+giants.size();
    };
    int baby = period;
    if (strategy!="diagonal") {
      auto best = cost(baby);
      for (int candidate=1; candidate<period; candidate*=2)
        if (cost(candidate)<best) { best=cost(candidate); baby=candidate; }
    }
    OpBuilder builder(op);
    auto make = [&](StringRef name, ValueRange operands, NamedAttrList attrs = {}) {
      OperationState state(op.getLoc(),name);
      state.addOperands(operands); state.addTypes(type); state.addAttributes(attrs);
      state.addAttribute("ppml.noise_status",builder.getStringAttr("unanalysed"));
      return builder.create(state)->getResult(0);
    };
    auto rotate = [&](Value input,int step) -> Value {
      if (!step) return input;
      NamedAttrList attrs; attrs.set("step",builder.getI64IntegerAttr(step));
      return make("heir.packed_rotate",input,attrs);
    };
    auto vectorAttr = [&](const std::vector<double> &values) {
      return DenseFPElementsAttr::get(RankedTensorType::get({period},builder.getF64Type()),ArrayRef<double>(values));
    };
    std::map<int,Value> babies, groups;
    babies[0] = op.getInput();
    for (auto &entry : diagonals) {
      int j=entry.first%baby, giant=entry.first-j;
      if (!babies.count(j)) babies[j] = rotate(op.getInput(),j);
      // Rot(sum_j Rot(D_{g+j},-g) * Rot(x,j),g).
      std::vector<double> shifted(period);
      for (int r=0; r<period; ++r) shifted[r]=entry.second[(r-giant+period)%period];
      NamedAttrList attrs; attrs.set("values",vectorAttr(shifted));
      Value term = make("heir.packed_mul_plain",babies[j],attrs);
      if (groups.count(giant)) groups[giant]=make("heir.packed_add",{groups[giant],term});
      else groups[giant]=term;
    }
    Value result;
    for (auto &group : groups) {
      Value term=rotate(group.second,group.first);
      result = result ? make("heir.packed_add",{result,term}) : term;
    }
    if (!result) result=make("heir.packed_zero",op.getInput());
    auto bias=op->getAttrOfType<DenseFPElementsAttr>("bias");
    std::vector<double> padded(period,0);
    bool hasBias=false; int index=0;
    for (double value : bias.getValues<double>()) { padded[index++]=value; hasBias |= value!=0; }
    if (hasBias) {
      NamedAttrList attrs; attrs.set("values",vectorAttr(padded));
      result=make("heir.packed_add_plain",result,attrs);
    }
    op.getOutput().replaceAllUsesWith(result); op.erase();
    function->setAttr("ppml.layout",builder.getStringAttr("periodic"));
    function->setAttr("ppml.input_elements",builder.getI64IntegerAttr(cols));
    function->setAttr("ppml.output_elements",builder.getI64IntegerAttr(rows));
    function->setAttr("ppml.matvec_strategy",builder.getStringAttr(baby==period ? "diagonal" : "bsgs"));
    function->setAttr("ppml.matvec_baby_step",builder.getI64IntegerAttr(baby));
    function->setAttr("ppml.nonzero_diagonals",builder.getI64IntegerAttr(diagonals.size()));
  }
};
}
std::unique_ptr<Pass> createMatVecPass() { return std::make_unique<MatVecPass>(); }
}
