#include "Pipelines/FHE/Passes.h"
#include "SCFHEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cmath>

using namespace mlir;
namespace crossfhe {
namespace {
struct ApproximateRelu : PassWrapper<ApproximateRelu,OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApproximateRelu)
  ApproximateRelu()=default;
  ApproximateRelu(const ApproximateRelu &other):PassWrapper(other) {}
  Option<double> bound{*this,"bound",llvm::cl::desc("Declared absolute bound on every matched ReLU input"),llvm::cl::init(0.0)};
  Option<double> maxError{*this,"max-error",llvm::cl::desc("Accepted local real-arithmetic approximation error"),llvm::cl::init(0.0)};
  StringRef getArgument() const final { return "ppml-scfhe-approx-relu"; }
  StringRef getDescription() const final { return "Opt in to bounded quadratic ReLU approximation on SCFHE"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect,mlir::libra::scfhe::SCFHEDialect>();
  }
  void runOnOperation() final {
    double b=bound, tolerance=maxError;
    if (!std::isfinite(b) || b<=0 || !std::isfinite(tolerance) || tolerance<b/16 ||
        !std::isfinite(0.5/b) || b/16==0) {
      getOperation().emitError("ReLU approximation requires finite bound > 0 and max-error >= bound/16");
      return signalPassFailure();
    }
    for (auto function : getOperation().getOps<func::FuncOp>()) {
      auto enabled=function->getAttrOfType<BoolAttr>("ppml.fhe_enabled");
      if (!enabled || !enabled.getValue()) continue;
      if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        function.emitError("ReLU approximation requires a one-block SCFHE function");return signalPassFailure();
      }
      unsigned count=0;
      for (Operation &op : llvm::make_early_inc_range(function.getBody().front())) {
        if (op.getName().getStringRef()!="scfhe.select" || op.getNumOperands()!=3) continue;
        auto *compare=op.getOperand(0).getDefiningOp();
        if (!compare || compare->getName().getStringRef()!="scfhe.cmp" || compare->getNumOperands()!=2) continue;
        auto predicate=compare->getAttrOfType<IntegerAttr>("predicate");
        auto zero=[](Value value) {
          auto c=value.getDefiningOp<arith::ConstantOp>();
          auto f=c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
          return f && f.getValueAsDouble()==0;
        };
        Value x=compare->getOperand(0);
        if (!predicate || !zero(compare->getOperand(1))) continue;
        bool olt=predicate.getInt()==static_cast<int64_t>(arith::CmpFPredicate::OLT);
        bool ogt=predicate.getInt()==static_cast<int64_t>(arith::CmpFPredicate::OGT);
        if (!((olt && zero(op.getOperand(1)) && op.getOperand(2)==x) ||
              (ogt && op.getOperand(1)==x && zero(op.getOperand(2))))) continue;
        OpBuilder builder(&op);
        auto make=[&](StringRef name,Value a,Value c) {
          OperationState state(op.getLoc(),name);state.addOperands({a,c});state.addTypes(op.getResult(0).getType());
          if (auto source=op.getAttr("ppml.source")) state.addAttribute("ppml.source",source);
          return builder.create(state)->getResult(0);
        };
        auto constant=[&](double value)->Value {
          return builder.create<arith::ConstantOp>(op.getLoc(),builder.getF64FloatAttr(value));
        };
        // ReLU(x)=(x+|x|)/2. On [-B,B], x*x/B+B/8 approximates
        // |x| with error <= B/8 (extrema at 0, +/-B/2, +/-B).
        Value square=make("scfhe.mult",x,x);
        Value quadratic=make("scfhe.mult",square,constant(0.5/b));
        Value linear=make("scfhe.mult",x,constant(0.5));
        Value result=make("scfhe.add",make("scfhe.add",quadratic,linear),constant(b/16));
        op.getResult(0).replaceAllUsesWith(result);op.erase();++count;
      }
      // Remove only comparisons made dead by the replacement. Shared masks
      // used by another (non-ReLU) select must remain intact.
      for (Operation &op : llvm::make_early_inc_range(function.getBody().front()))
        if (op.getName().getStringRef()=="scfhe.cmp" && op.use_empty()) op.erase();
      if (count) {
        OpBuilder builder(function);
        function->setAttr("ppml.relu_approximation_count",builder.getI64IntegerAttr(count));
        function->setAttr("ppml.relu_input_bound",builder.getF64FloatAttr(b));
        function->setAttr("ppml.relu_local_error_bound",builder.getF64FloatAttr(b/16));
      }
    }
  }
};
}
std::unique_ptr<Pass> createApproximateReluPass() { return std::make_unique<ApproximateRelu>(); }
}
