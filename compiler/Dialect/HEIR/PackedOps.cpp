#include "heir/IR/FHE/HEIRDialect.h"
#include <cmath>

using namespace mlir;
namespace {
LogicalResult checkTypes(Operation *op) {
  auto type = dyn_cast<heir::RLWECipherType>(op->getResult(0).getType());
  if (!type || !type.getPlaintextType().isF64() || type.getSize() < 1 ||
      type.getSize() > 16384 || (type.getSize() & (type.getSize()-1)))
    return op->emitOpError("requires an f64 cipher period that is a power of two in [1,16384]");
  for (Value value : op->getOperands())
    if (value.getType() != type)
      return op->emitOpError("requires matching packed input/output types");
  return success();
}
LogicalResult checkValues(Operation *op, StringRef name, int rank) {
  auto attr = op->getAttrOfType<DenseFPElementsAttr>(name);
  if (!attr || !attr.getType().getElementType().isF64() || attr.getType().getRank()!=rank)
    return op->emitOpError("requires dense f64 ") << name << " of rank " << rank;
  for (double value : attr.getValues<double>())
    if (!std::isfinite(value)) return op->emitOpError("requires finite public coefficients");
  return success();
}
LogicalResult checkVector(Operation *op) {
  if (failed(checkTypes(op)) || failed(checkValues(op,"values",1))) return failure();
  if (op->getAttrOfType<DenseFPElementsAttr>("values").getNumElements()!=
      cast<heir::RLWECipherType>(op->getResult(0).getType()).getSize())
    return op->emitOpError("plaintext vector must fill exactly one packed period");
  return success();
}
}
LogicalResult heir::MatVecOp::verify() {
  if (failed(checkTypes(*this)) || failed(checkValues(*this,"matrix",2)) ||
      failed(checkValues(*this,"bias",1))) return failure();
  auto matrix = (*this)->getAttrOfType<DenseFPElementsAttr>("matrix");
  auto bias = (*this)->getAttrOfType<DenseFPElementsAttr>("bias");
  auto shape = matrix.getType().getShape();
  auto period = cast<RLWECipherType>(getOutput().getType()).getSize();
  if (shape[0]<=0 || shape[1]<=0 || shape[0]>period || shape[1]>period ||
      bias.getNumElements()!=shape[0])
    return emitOpError("matrix dimensions must fit the packed period and bias must match rows");
  return success();
}
LogicalResult heir::PackedRotateOp::verify() {
  if (failed(checkTypes(*this))) return failure();
  auto period = cast<RLWECipherType>(getOutput().getType()).getSize();
  auto step = (*this)->getAttrOfType<IntegerAttr>("step").getInt();
  if (step<=0 || step>=period) return emitOpError("rotation step must be in (0, period)");
  return success();
}
LogicalResult heir::PackedMulPlainOp::verify() {
  if (failed(checkVector(*this))) return failure();
  auto values = (*this)->getAttrOfType<DenseFPElementsAttr>("values");
  bool nonzero = false;
  for (double value : values.getValues<double>()) nonzero |= value!=0;
  if (!nonzero) return emitOpError("zero plaintext multiplication must use packed_zero");
  return success();
}
LogicalResult heir::PackedAddPlainOp::verify() { return checkVector(*this); }
LogicalResult heir::PackedAddOp::verify() { return checkTypes(*this); }
LogicalResult heir::PackedZeroOp::verify() { return checkTypes(*this); }
