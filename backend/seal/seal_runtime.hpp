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
  std::uint64_t rotations = 0, add_plain = 0, materializations = 0;
};

class SealRuntime {
 public:
  SealRuntime(const seal::SEALContext &context, const seal::PublicKey &key,
              const seal::RelinKeys &relin, double scale,
              const seal::GaloisKeys *galois = nullptr)
      : context_(context), encoder_(context), evaluator_(context),
        encryptor_(context, key), relin_(relin), scale_(scale), galois_(galois) {}
  Counts counts;

  seal::Ciphertext rotate(const seal::Ciphertext &a, int step) {
    if (!galois_) throw std::runtime_error("missing packed rotation keys");
    seal::Ciphertext output;
    evaluator_.rotate_vector(a, step, *galois_, output);
    ++counts.rotations;
    return output;
  }
  seal::Ciphertext zero_like(const seal::Ciphertext &a) {
    seal::Ciphertext output;
    encryptor_.encrypt_zero(a.parms_id(), output);
    output.scale() = a.scale();
    ++counts.materializations;
    return output;
  }
  seal::Ciphertext mul_periodic(const seal::Ciphertext &a, const std::vector<double> &values) {
    auto expanded = expand(values);
    if (std::all_of(values.begin(),values.end(),[](double v) { return v==0; }))
      throw std::runtime_error("zero packed product must be materialized explicitly");
    seal::Plaintext plain; seal::Ciphertext output;
    encoder_.encode(expanded,a.parms_id(),scale_,plain);
    evaluator_.multiply_plain(a,plain,output);
    ++counts.multiply_plain;
    refreshScale(output);
    return output;
  }
  seal::Ciphertext add_periodic(const seal::Ciphertext &a, const std::vector<double> &values) {
    seal::Plaintext plain; seal::Ciphertext output;
    encoder_.encode(expand(values),a.parms_id(),a.scale(),plain);
    evaluator_.add_plain(a,plain,output);
    ++counts.add_plain;
    return output;
  }

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
  const seal::GaloisKeys *galois_;
  std::vector<double> expand(const std::vector<double> &values) {
    if (values.empty() || encoder_.slot_count()%values.size())
      throw std::runtime_error("packed period must divide physical slot count");
    for (double value : values)
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite packed coefficient");
    std::vector<double> output(encoder_.slot_count());
    for (std::size_t i=0; i<output.size(); ++i) output[i]=values[i%values.size()];
    return output;
  }
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
