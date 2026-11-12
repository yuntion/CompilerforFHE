#include "Pipelines/FHE/Passes.h"
#include "SCFHEDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using mlir::libra::scfhe::SCFHECipherType;
namespace crossfhe {
namespace {
struct LowerToSCFHE : PassWrapper<LowerToSCFHE, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToSCFHE)
  StringRef getArgument() const final { return "ppml-to-scfhe"; }
  StringRef getDescription() const final { return "Lower static PPML to Libra's SCFHE dialect"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::libra::scfhe::SCFHEDialect, arith::ArithDialect, func::FuncDialect>();
  }
  void runOnOperation() final {
    SmallVector<func::FuncOp> functions(getOperation().getOps<func::FuncOp>());
    if (functions.size() != 1) {
      getOperation().emitError("expected one model function"); return signalPassFailure();
    }
    auto function = functions.front();
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
      function.emitError("only one-block static functions are supported"); return signalPassFailure();
    }
    for (Type type : function.getFunctionType().getInputs())
      if (!type.isF64()) { function.emitError("inputs must be f64"); return signalPassFailure(); }
    for (Type type : function.getFunctionType().getResults())
      if (!type.isF64()) { function.emitError("outputs must be f64"); return signalPassFailure(); }
    auto *ctx = &getContext();
    auto numeric = SCFHECipherType::get(ctx, 1, Float64Type::get(ctx));
    auto boolean = SCFHECipherType::get(ctx, 1, IntegerType::get(ctx, 1));
    OpBuilder builder(function);
    auto replacement = builder.create<func::FuncOp>(function.getLoc(), "model_scfhe",
        builder.getFunctionType(SmallVector<Type>(function.getNumArguments(),numeric),
                                SmallVector<Type>(function.getNumResults(),numeric)));
    builder.setInsertionPointToEnd(replacement.addEntryBlock());
    IRMapping mapping;
    for (auto pair : llvm::zip(function.getArguments(),replacement.getArguments()))
      mapping.map(std::get<0>(pair),std::get<1>(pair));
    for (Operation &op : function.getBody().front()) {
      if (isa<arith::ConstantOp>(op)) { builder.clone(op,mapping); continue; }
      if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        SmallVector<Value> values;
        for (Value value : ret.getOperands()) {
          value = mapping.lookup(value);
          if (value.getType().isF64()) {
            OperationState state(op.getLoc(),"scfhe.encrypt");
            state.addOperands(value); state.addTypes(numeric);
            value = builder.create(state)->getResult(0);
          }
          values.push_back(value);
        }
        builder.create<func::ReturnOp>(op.getLoc(),values); continue;
      }
      SmallVector<Value> operands;
      bool secret = false;
      for (Value value : op.getOperands()) {
        auto mapped = mapping.lookup(value); operands.push_back(mapped);
        secret |= isa<SCFHECipherType>(mapped.getType());
      }
      StringRef source = op.getName().getStringRef(), target;
      if (source == "arith.addf") target = "scfhe.add";
      else if (source == "arith.subf") target = "scfhe.sub";
      else if (source == "arith.mulf") target = "scfhe.mult";
      else if (source == "arith.cmpf") target = "scfhe.cmp";
      else if (source == "arith.select") target = "scfhe.select";
      else { op.emitError("source operation has no SCFHE lowering"); return signalPassFailure(); }
      if (!secret) { builder.clone(op,mapping); continue; }
      OperationState state(op.getLoc(),target);
      state.addOperands(operands); state.addTypes(source == "arith.cmpf" ? boolean : numeric);
      if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
        auto predicate = cmp.getPredicate();
        if (predicate != arith::CmpFPredicate::OLT && predicate != arith::CmpFPredicate::OGT) {
          op.emitError("only ordered < and > are supported by the sign backend"); return signalPassFailure();
        }
        state.addAttribute("predicate",builder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
      }
      if (auto attr = op.getAttr("ppml.source")) state.addAttribute("ppml.source",attr);
      auto *lowered = builder.create(state);
      mapping.map(op.getResult(0),lowered->getResult(0));
    }
    replacement->setAttr("ppml.model_parameters",builder.getStringAttr("public"));
    replacement->setAttr("ppml.layout",builder.getStringAttr("one-scalar-per-ciphertext"));
    function.erase();
  }
};
}
std::unique_ptr<Pass> createLowerToSCFHEPass() { return std::make_unique<LowerToSCFHE>(); }
}
