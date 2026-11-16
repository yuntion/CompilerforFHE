module {
  func.func @model(%a: f64, %b: f64, %c: f64, %d: f64,
                   %e: f64, %f: f64, %g: f64, %h: f64) -> f64 {
    %v1 = arith.mulf %a, %b : f64
    %v2 = arith.mulf %v1, %c : f64
    %v3 = arith.mulf %v2, %d : f64
    %v4 = arith.mulf %v3, %e : f64
    %v5 = arith.mulf %v4, %f : f64
    %v6 = arith.mulf %v5, %g : f64
    %v7 = arith.mulf %v6, %h : f64
    return %v7 : f64
  }
}
