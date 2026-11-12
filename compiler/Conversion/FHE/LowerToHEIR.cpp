#include "Pipelines/FHE/Passes.h"
#include "SCFHEDialect.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
namespace crossfhe {
namespace {
struct LowerToHEIR : PassWrapper<LowerToHEIR, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToHEIR)
  StringRef getArgument() const final { return "ppml-to-heir"; }
  StringRef getDescription() const final {
    return "Lower Libra SCFHE to CKKS arithmetic and HEIR/CHLOE LWE sign operations";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<heir::HEIRDialect,mlir::libra::scfhe::SCFHEDialect,
                    arith::ArithDialect,func::FuncDialect>();
  }
  void runOnOperation() final {
    SmallVector<func::FuncOp> functions(getOperation().getOps<func::FuncOp>());
    if (functions.size()!=1) { getOperation().emitError("expected one model"); return signalPassFailure(); }
    auto function = functions.front();
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
      function.emitError("only one-block functions are supported"); return signalPassFailure();
    }
    using mlir::libra::scfhe::SCFHECipherType;
    auto isNumeric = [](Type type) {
      auto cipher = dyn_cast<SCFHECipherType>(type);
      return cipher && cipher.getPlaintextCount()==1 && cipher.getElementType().isF64();
    };
    for (Type type : function.getFunctionType().getInputs())
      if (!isNumeric(type)) { function.emitError("run --ppml-to-scfhe first"); return signalPassFailure(); }
    for (Type type : function.getFunctionType().getResults())
      if (!isNumeric(type)) { function.emitError("only encrypted numeric outputs supported"); return signalPassFailure(); }
    auto *ctx = &getContext();
    Type cipher = heir::RLWECipherType::get(ctx,Float64Type::get(ctx),1,heir::NoiseType());
    Type lwe = heir::LWECipherVectorType::get(ctx,Float64Type::get(ctx),1,heir::NoiseType());
    OpBuilder builder(function);
    auto replacement = builder.create<func::FuncOp>(function.getLoc(),"model_fhe",
        builder.getFunctionType(SmallVector<Type>(function.getNumArguments(),cipher),
                                SmallVector<Type>(function.getNumResults(),cipher)));
    builder.setInsertionPointToEnd(replacement.addEntryBlock());
    IRMapping mapping;
    llvm::DenseMap<Value,unsigned> depths;
    unsigned maxDepth=0, logicCount=0;
    for (auto pair : llvm::zip(function.getArguments(),replacement.getArguments()))
      mapping.map(std::get<0>(pair),std::get<1>(pair));
    for (Operation &op : function.getBody().front()) {
      if (isa<arith::ConstantOp>(op)) { builder.clone(op,mapping); continue; }
      if (auto ret = dyn_cast<func::ReturnOp>(op)) {
        SmallVector<Value> values;
        for (Value operand : ret.getOperands()) values.push_back(mapping.lookup(operand));
        builder.create<func::ReturnOp>(op.getLoc(),values); continue;
      }
      SmallVector<Value> operands;
      unsigned depth=0;
      for (Value value : op.getOperands()) {
        auto mapped = mapping.lookup(value); operands.push_back(mapped);
        depth = std::max(depth,depths.lookup(mapped));
      }
      auto make = [&](StringRef name, ValueRange inputs, Type result, bool noise=true) {
        OperationState state(op.getLoc(),name);
        state.addOperands(inputs); state.addTypes(result);
        if (noise) {
          state.addAttribute("noise",builder.getF64FloatAttr(0.0));
          state.addAttribute("ppml.noise_status",builder.getStringAttr("unanalysed"));
        }
        if (auto source = op.getAttr("ppml.source")) state.addAttribute("ppml.source",source);
        return builder.create(state)->getResult(0);
      };
      StringRef name=op.getName().getStringRef();
      Value result;
      if (name=="scfhe.add") result=make("heir.rlweadd",operands,cipher);
      else if (name=="scfhe.sub") result=make("heir.rlwesub",operands,cipher);
      else if (name=="scfhe.mult") { result=make("heir.rlwemul",operands,cipher); ++depth; }
      else if (name=="scfhe.encrypt") result=make("heir.materialize",operands,cipher,false);
      else if (name=="scfhe.cmp") {
        auto attr=op.getAttrOfType<IntegerAttr>("predicate");
        if (!attr || (attr.getInt()!=static_cast<int64_t>(arith::CmpFPredicate::OLT) &&
                      attr.getInt()!=static_cast<int64_t>(arith::CmpFPredicate::OGT))) {
          op.emitError("only < and > comparison supported"); return signalPassFailure();
        }
        if (attr.getInt()==static_cast<int64_t>(arith::CmpFPredicate::OGT))
          std::swap(operands[0],operands[1]);
        Value diff=make("heir.rlwesub",operands,cipher);
        Value unpacked=make("heir.depack",ValueRange{diff},lwe,false);
        unpacked.getDefiningOp()->setAttr("ppml.scheme",builder.getStringAttr("ckks-to-fhew"));
        Value sign=make("heir.lut_lt",ValueRange{unpacked},lwe);
        sign.getDefiningOp()->setAttr("ppml.encoding",builder.getStringAttr("negative-sign-bit"));
        result=make("heir.repack",ValueRange{sign},cipher);
        result.getDefiningOp()->setAttr("ppml.scheme",builder.getStringAttr("fhew-sign-to-ckks-mask"));
        ++logicCount;
      } else if (name=="scfhe.select") {
        result=make("heir.select",operands,cipher,false); ++depth;
      } else {
        op.emitError("unsupported SCFHE operation; backend has no capability"); return signalPassFailure();
      }
      depths[result]=depth; maxDepth=std::max(maxDepth,depth);
      mapping.map(op.getResult(0),result);
    }
    // Reserve switching levels in addition to source arithmetic. This is a
    // conservative executable contract, not an optimization or noise model.
    replacement->setAttr("ppml.multiplicative_depth",builder.getI64IntegerAttr(maxDepth));
    replacement->setAttr("ppml.logic_comparisons",builder.getI64IntegerAttr(logicCount));
    replacement->setAttr("ppml.openfhe_depth",builder.getI64IntegerAttr(maxDepth+(logicCount ? 17 : 0)));
    replacement->setAttr("ppml.layout",builder.getStringAttr("one-scalar-per-ciphertext"));
    replacement->setAttr("ppml.arithmetic_scheme",builder.getStringAttr("ckks"));
    replacement->setAttr("ppml.logic_scheme",builder.getStringAttr("fhew-sign"));
    function.erase();
  }
};
}
std::unique_ptr<Pass> createLowerToHEIRPass() { return std::make_unique<LowerToHEIR>(); }
}
