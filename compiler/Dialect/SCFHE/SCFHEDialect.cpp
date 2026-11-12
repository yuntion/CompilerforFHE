// Libra's ODS definitions, with LLVM 17 registration glue and include compatibility.
// Shape inference accepts scalar public operands on either side. Optimization
// hooks remain empty until the user supplies a pass pipeline.
#include "SCFHEDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::libra::scfhe;

static ParseResult parseCipherDim(AsmParser &parser, int64_t &dim) {
  if (succeeded(parser.parseOptionalQuestion())) {
    dim = ShapedType::kDynamic;
    return success();
  }
  return parser.parseInteger(dim);
}
static void printCipherDim(AsmPrinter &printer, int64_t dim) {
  if (dim == ShapedType::kDynamic) printer << "?";
  else printer << dim;
}
static LogicalResult inferBinary(ValueRange values, SmallVectorImpl<Type> &types) {
  if (values.size() != 2) return failure();
  auto left = dyn_cast<SCFHECipherType>(values[0].getType());
  auto right = dyn_cast<SCFHECipherType>(values[1].getType());
  if (!left && !right) return failure();
  if (left && right && left != right) return failure();
  Type other = left ? values[1].getType() : values[0].getType();
  if (!(left && right) && !other.isF64()) return failure();
  types.push_back(left ? left : right);
  return success();
}

#define INFER_BINARY(CLASS) \
LogicalResult CLASS::inferReturnTypes(MLIRContext *, std::optional<Location>, \
    ValueRange operands, DictionaryAttr, RegionRange, \
    SmallVectorImpl<Type> &types) { return inferBinary(operands, types); }
INFER_BINARY(SCFHEAddOp)
INFER_BINARY(SCFHESubOp)
INFER_BINARY(SCFHEMultOp)
INFER_BINARY(SCFHEDivOp)
#undef INFER_BINARY
LogicalResult SCFHEDecryptOp::inferReturnTypes(MLIRContext *,
    std::optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type> &types) {
  auto input = dyn_cast<SCFHECipherType>(operands[0].getType());
  if (!input) return failure();
  types.push_back(MemRefType::get({input.getPlaintextCount()}, input.getElementType()));
  return success();
}
#define EMPTY_CANONICALIZER(CLASS) \
void CLASS::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) {}
EMPTY_CANONICALIZER(SCFHEEncryptOp)
EMPTY_CANONICALIZER(SCFHEAddOp)
EMPTY_CANONICALIZER(SCFHESubOp)
EMPTY_CANONICALIZER(SCFHEMultOp)
EMPTY_CANONICALIZER(SCFHEDivOp)
EMPTY_CANONICALIZER(SCFHEDecryptOp)
EMPTY_CANONICALIZER(SCFHEReduceAddOp)
#undef EMPTY_CANONICALIZER

#define GET_TYPEDEF_CLASSES
#include "SCFHETypes.cpp.inc"
#define GET_OP_CLASSES
#include "SCFHEOps.cpp.inc"
#include "SCFHEDialect.cpp.inc"
void SCFHEDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "SCFHETypes.cpp.inc"
  >();
  addOperations<
#define GET_OP_LIST
#include "SCFHEOps.cpp.inc"
  >();
}
