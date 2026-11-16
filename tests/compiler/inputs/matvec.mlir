module {
  func.func @model(%x: !heir.rlwecipher<4 x f64>) -> !heir.rlwecipher<4 x f64>
      attributes {ppml.fhe_enabled = true} {
    %y = "heir.matvec"(%x) {matrix = dense<1.0> : tensor<4x4xf64>, bias = dense<0.0> : tensor<4xf64>}
      : (!heir.rlwecipher<4 x f64>) -> !heir.rlwecipher<4 x f64>
    return %y : !heir.rlwecipher<4 x f64>
  }
}
