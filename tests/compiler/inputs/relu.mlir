module {
  func.func @model(%x: f64) -> f64 {
    %zero = arith.constant 0.0 : f64
    %negative = arith.cmpf olt, %x, %zero : f64
    %result = arith.select %negative, %zero, %x : f64
    return %result : f64
  }
}
