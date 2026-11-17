#pragma once
#include "phantom.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace crossfhe {
using Cipher = PhantomCiphertext;
inline void cudaCheck(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Counts {
  std::uint64_t add=0, sub=0, multiply_plain=0, multiply_cipher=0;
  std::uint64_t rescale=0, mod_switch=0, relinearize=0;
  std::uint64_t rotations=0, add_plain=0, materializations=0;
};

// Evaluation needs only public/evaluation keys. The secret key belongs to runner's client phases.
class PhantomRuntime {
 public:
  static constexpr double scale = 1099511627776.0;
  PhantomRuntime(const PhantomContext &context, PhantomPublicKey &key,
                 const PhantomRelinKey &relin, const PhantomGaloisKey *galois=nullptr)
      : context_(context), encoder_(context), key_(key), relin_(relin), galois_(galois) {}
  Counts counts;

  Cipher constant(double value) {
    auto plain = encode({value}, scale, context_.get_first_index());
    auto result = key_.encrypt_asymmetric(context_, plain);
    ++counts.materializations;
    return result;
  }
  Cipher zero_like(const Cipher &value) {
    validate(value);
    // Phantom's direct asymmetric encryption at lower levels failed the feasibility probe.
    auto result = key_.encrypt_zero_asymmetric(context_);
    result.set_scale(value.scale());
    switchTo(result, value.chain_index());
    ++counts.materializations;
    return result;
  }
  Cipher rotate(const Cipher &value, int step) {
    validate(value);
    if (!galois_) throw std::runtime_error("missing packed rotation keys");
    auto result = phantom::rotate(context_, value, step, *galois_);
    ++counts.rotations;
    return result;
  }
  Cipher add(const Cipher &a, const Cipher &b) {
    auto pair = align(a,b);
    ++counts.add;
    return phantom::add(context_, pair.first, pair.second);
  }
  Cipher add(const Cipher &a, double b) {
    validate(a);
    auto plain = encode({b}, a.scale(), a.chain_index());
    ++counts.add;
    return phantom::add_plain(context_, a, plain);
  }
  Cipher add(double a, const Cipher &b) { return add(b,a); }
  Cipher sub(const Cipher &a, const Cipher &b) {
    auto pair = align(a,b);
    ++counts.sub;
    if (a.data()==b.data()) return zero_like(a);
    return phantom::sub(context_, pair.first, pair.second);
  }
  Cipher sub(const Cipher &a, double b) { return add(a,-b); }
  Cipher sub(double a, const Cipher &b) {
    validate(b);
    return add(phantom::negate(context_, b),a);
  }
  Cipher mul(const Cipher &a, const Cipher &b) {
    auto pair = align(a,b);
    requireNext(pair.first);
    auto result = phantom::multiply_and_relin(context_, pair.first, pair.second, relin_);
    ++counts.multiply_cipher;
    ++counts.relinearize;
    refreshScale(result);
    return result;
  }
  Cipher mul(const Cipher &a, double b) { return mul_periodic(a,{b}); }
  Cipher mul(double a, const Cipher &b) { return mul(b,a); }
  Cipher mul_periodic(const Cipher &a, const std::vector<double> &weights) {
    validate(a);
    requireNext(a);
    auto plain = encode(weights, scale, a.chain_index());
    const bool zero=std::all_of(weights.begin(), weights.end(), [](double v) { return v==0; });
    auto result = zero ? zero_like(a) : phantom::multiply_plain(context_, a, plain);
    if (zero)
      result.set_scale(a.scale()*scale);
    ++counts.multiply_plain;
    refreshScale(result);
    return result;
  }
  Cipher add_periodic(const Cipher &a, const std::vector<double> &bias) {
    validate(a);
    auto plain = encode(bias, a.scale(), a.chain_index());
    ++counts.add_plain;
    return phantom::add_plain(context_, a, plain);
  }
 private:
  const PhantomContext &context_;
  PhantomCKKSEncoder encoder_;
  PhantomPublicKey &key_;
  const PhantomRelinKey &relin_;
  const PhantomGaloisKey *galois_;

  PhantomPlaintext encode(const std::vector<double> &values, double valueScale, std::size_t level) {
    if (values.empty() || encoder_.slot_count()%values.size())
      throw std::runtime_error("packed period must divide physical slot count");
    for (double v : values)
      if (!std::isfinite(v)) throw std::runtime_error("nonfinite public coefficient");
    std::vector<double> expanded(encoder_.slot_count());
    for (std::size_t i=0; i<expanded.size(); ++i) expanded[i]=values[i%values.size()];
    return encoder_.encode(context_, expanded, valueScale, level);
  }
  void validate(const Cipher &value) const {
    if (value.chain_index()<context_.get_first_index() ||
        value.chain_index()>=context_.total_parm_size())
      throw std::runtime_error("invalid ciphertext level");
    const auto &params=context_.get_context_data(value.chain_index()).parms();
    if (value.size()!=2 || !value.is_ntt_form() ||
        value.poly_modulus_degree()!=params.poly_modulus_degree() ||
        value.coeff_modulus_size()!=params.coeff_modulus().size() ||
        !std::isfinite(value.scale()) || value.scale()<=0 ||
        std::abs(std::log2(value.scale())-40.0)>0.01)
      throw std::runtime_error("ciphertext parameter context mismatch");
  }
  void switchTo(Cipher &value, std::size_t level) {
    if (value.chain_index()!=level) {
      phantom::mod_switch_to_inplace(context_, value, level);
      ++counts.mod_switch;
    }
  }
  std::pair<Cipher,Cipher> align(const Cipher &a, const Cipher &b) {
    validate(a); validate(b);
    if (std::abs(std::log2(a.scale())-std::log2(b.scale()))>0.01)
      throw std::runtime_error("scale mismatch");
    auto x=a, y=b;
    // Phantom indices increase as the modulus chain is consumed.
    auto level=std::max(a.chain_index(),b.chain_index());
    switchTo(x,level); switchTo(y,level);
    x.set_scale(scale); y.set_scale(scale);
    return {std::move(x),std::move(y)};
  }
  void requireNext(const Cipher &value) const {
    if (value.chain_index()+1>=context_.total_parm_size())
      throw std::runtime_error("modulus chain exhausted");
  }
  void refreshScale(Cipher &value) {
    phantom::rescale_to_next_inplace(context_, value);
    value.set_scale(scale);
    ++counts.rescale;
  }
};
using Runtime = PhantomRuntime;
}
