module {
  func.func @model_fhe(%x: !heir.rlwecipher<1 x f64>) -> !heir.rlwecipher<1 x f64>
      attributes {ppml.multiplicative_depth = 0 : i64, ppml.logic_comparisons = 0 : i64,
                  ppml.openfhe_depth = 0 : i64} {
    %constant = arith.constant 0.25 : f64
    %result = heir.rlweadd(%x, %constant) {noise = 0.0 : f64, ppml.noise_status = "unanalysed"}
        : (!heir.rlwecipher<1 x f64>, f64) -> !heir.rlwecipher<1 x f64>
    return %result : !heir.rlwecipher<1 x f64>
  }
}
