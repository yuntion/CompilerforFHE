#pragma once
#include <seal/seal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace crossfhe {
using Cipher = seal::Ciphertext;
struct Counts {
  std::uint64_t add = 0, sub = 0, multiply_plain = 0, multiply_cipher = 0;
  std::uint64_t rescale = 0, mod_switch = 0, relinearize = 0;
};

class SealRuntime {
 public:
  SealRuntime(const seal::SEALContext &context, const seal::PublicKey &key,
              const seal::RelinKeys &relin, double scale)
      : context_(context), encoder_(context), evaluator_(context),
        encryptor_(context, key), relin_(relin), scale_(scale) {}
  Counts counts;

  seal::Ciphertext constant(double value) {
    seal::Plaintext plaintext;
    encoder_.encode(value, scale_, plaintext);
    seal::Ciphertext output;
    encryptor_.encrypt(plaintext, output);
    return output;
  }
  seal::Ciphertext add(const seal::Ciphertext &a, const seal::Ciphertext &b) {
    auto pair = align(a, b);
    seal::Ciphertext output;
    evaluator_.add(pair.first, pair.second, output);
    ++counts.add;
    return output;
  }
  seal::Ciphertext add(const seal::Ciphertext &a, double b) {
    seal::Plaintext plaintext;
    encoder_.encode(b, a.parms_id(), a.scale(), plaintext);
    seal::Ciphertext output;
    evaluator_.add_plain(a, plaintext, output);
    ++counts.add;
    return output;
  }
  seal::Ciphertext add(double a, const seal::Ciphertext &b) { return add(b, a); }
  seal::Ciphertext sub(const seal::Ciphertext &a, const seal::Ciphertext &b) {
    auto pair = align(a, b);
    seal::Ciphertext output;
    // SEAL rejects transparent ciphertexts; fresh public-key zero preserves secrecy.
    if (a.data() == b.data()) {
      encryptor_.encrypt_zero(pair.first.parms_id(), output);
      output.scale() = scale_;
    } else evaluator_.sub(pair.first, pair.second, output);
    ++counts.sub;
    return output;
  }
  seal::Ciphertext sub(const seal::Ciphertext &a, double b) { return add(a, -b); }
  seal::Ciphertext sub(double a, const seal::Ciphertext &b) {
    seal::Ciphertext negated;
    evaluator_.negate(b, negated);
    return add(negated, a);
  }
  seal::Ciphertext mul(const seal::Ciphertext &a, const seal::Ciphertext &b) {
    auto pair = align(a, b);
    seal::Ciphertext output;
    evaluator_.multiply(pair.first, pair.second, output);
    evaluator_.relinearize_inplace(output, relin_);
    ++counts.multiply_cipher;
    ++counts.relinearize;
    refreshScale(output);
    return output;
  }
  seal::Ciphertext mul(const seal::Ciphertext &a, double b) {
    if (!std::isfinite(b)) throw std::runtime_error("nonfinite public coefficient");
    seal::Ciphertext output;
    if (b == 0.0) {
      encryptor_.encrypt_zero(a.parms_id(), output);
      output.scale() = a.scale() * scale_;
    } else {
      seal::Plaintext plaintext;
      encoder_.encode(b, a.parms_id(), scale_, plaintext);
      evaluator_.multiply_plain(a, plaintext, output);
    }
    ++counts.multiply_plain;
    refreshScale(output);
    return output;
  }
  seal::Ciphertext mul(double a, const seal::Ciphertext &b) { return mul(b, a); }

 private:
  const seal::SEALContext &context_;
  seal::CKKSEncoder encoder_;
  seal::Evaluator evaluator_;
  seal::Encryptor encryptor_;
  const seal::RelinKeys &relin_;
  double scale_;
  std::pair<seal::Ciphertext, seal::Ciphertext> align(
      const seal::Ciphertext &a, const seal::Ciphertext &b) {
    auto left = context_.get_context_data(a.parms_id());
    auto right = context_.get_context_data(b.parms_id());
    if (!left || !right) throw std::runtime_error("ciphertext parameter context mismatch");
    auto x = a, y = b;
    if (left->chain_index() > right->chain_index()) {
      evaluator_.mod_switch_to_inplace(x, b.parms_id()); ++counts.mod_switch;
    } else if (right->chain_index() > left->chain_index()) {
      evaluator_.mod_switch_to_inplace(y, a.parms_id()); ++counts.mod_switch;
    }
    if (std::abs(std::log2(x.scale()) - std::log2(y.scale())) > 0.01)
      throw std::runtime_error("scale mismatch");
    x.scale() = y.scale() = scale_;
    return {std::move(x), std::move(y)};
  }
  void refreshScale(seal::Ciphertext &value) {
    auto data = context_.get_context_data(value.parms_id());
    if (!data || !data->next_context_data())
      throw std::runtime_error("modulus chain exhausted");
    evaluator_.rescale_to_next_inplace(value);
    value.scale() = scale_;
    ++counts.rescale;
  }
};
using Runtime = SealRuntime;
}
