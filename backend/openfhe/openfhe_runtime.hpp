#pragma once
#include "openfhe.h"
#include "binfhecontext.h"
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace crossfhe {
using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using LogicCipher = std::vector<lbcrypto::LWECiphertext>;
struct Counts {
  std::uint64_t add=0, sub=0, multiply_plain=0, multiply_cipher=0;
  std::uint64_t ckks_to_logic=0, logic_bootstrap=0, logic_to_ckks=0, select=0;
  double ckks_to_logic_seconds=0, logic_seconds=0, logic_to_ckks_seconds=0;
};
class Runtime {
 public:
  Runtime(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context,
          lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey)
      : context_(std::move(context)), publicKey_(std::move(publicKey)) {}
  Counts counts;
  Cipher constant(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite public value");
    return context_->Encrypt(publicKey_,context_->MakeCKKSPackedPlaintext(std::vector<double>{value},1,0,nullptr,1));
  }
  Cipher add(const Cipher &a,const Cipher &b) { ++counts.add; return context_->EvalAdd(a,b); }
  Cipher add(const Cipher &a,double b) { ++counts.add; return context_->EvalAdd(a,b); }
  Cipher add(double a,const Cipher &b) { return add(b,a); }
  double add(double a,double b) { return a+b; }
  Cipher sub(const Cipher &a,const Cipher &b) { ++counts.sub; return context_->EvalSub(a,b); }
  Cipher sub(const Cipher &a,double b) { ++counts.sub; return context_->EvalSub(a,b); }
  Cipher sub(double a,const Cipher &b) { ++counts.sub; return context_->EvalAdd(context_->EvalNegate(b),a); }
  double sub(double a,double b) { return a-b; }
  Cipher mul(const Cipher &a,const Cipher &b) { ++counts.multiply_cipher; return context_->EvalMult(a,b); }
  Cipher mul(const Cipher &a,double b) { ++counts.multiply_plain; return context_->EvalMult(a,b); }
  Cipher mul(double a,const Cipher &b) { return mul(b,a); }
  double mul(double a,double b) { return a*b; }
  LogicCipher to_logic(const Cipher &value) {
    auto start=std::chrono::steady_clock::now();
    auto output=context_->EvalCKKStoFHEW(value,1);
    counts.ckks_to_logic_seconds+=elapsed(start); ++counts.ckks_to_logic;
    if (output.size()!=1) throw std::runtime_error("invalid scalar CKKS to LWE conversion");
    return output;
  }
  LogicCipher sign(const LogicCipher &value) {
    if (value.size()!=1) throw std::runtime_error("expected one logic ciphertext");
    auto start=std::chrono::steady_clock::now();
    // The switch-compatible sign encoding matches OpenFHE's comparison path.
    auto output=context_->GetBinCCForSchemeSwitch()->EvalSign(value[0],true);
    counts.logic_seconds+=elapsed(start); ++counts.logic_bootstrap;
    return {output};
  }
  Cipher to_ckks_mask(LogicCipher value) {
    if (value.size()!=1) throw std::runtime_error("expected one sign ciphertext");
    auto start=std::chrono::steady_clock::now();
    auto output=context_->EvalFHEWtoCKKS(value,1,1,4,-1.0,1.0,0);
    counts.logic_to_ckks_seconds+=elapsed(start); ++counts.logic_to_ckks;
    return output;
  }
  template<class A,class B> Cipher select(const Cipher &mask,const A &a,const B &b) {
    ++counts.select;
    return add(mul(mask,sub(a,b)),b);
  }
 private:
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context_;
  lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey_;
  static double elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  }
};
}
