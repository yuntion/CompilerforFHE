module {
  func.func @model(%x: f64) -> (f64, f64) {
    %zero = arith.constant 0.0 : f64
    %a = arith.cmpf olt, %x, %zero {ppml.source = "first"} : f64
    %b = arith.cmpf olt, %x, %zero {ppml.source = "second"} : f64
    %u = arith.select %a, %zero, %x : f64
    %v = arith.select %b, %zero, %x : f64
    return %u, %v : f64, f64
  }
}
