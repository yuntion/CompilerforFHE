#pragma once
#include "phantom_runtime.hpp"
#include "host/rns.h"
#include "host/uintarith.h"
#include "host/uintcore.h"

namespace crossfhe {
// Client-only workaround for the pinned library's intermittent GPU CRT decode
// failures. Keep inverse NTT and FFT on CUDA, but reconstruct signed integers
// exactly on the host before converting them to floating point.
inline std::vector<double> decodeClient(const PhantomContext &context,
                                        PhantomCKKSEncoder &encoder,
                                        const PhantomPlaintext &plain) {
  const auto &data = context.get_context_data(plain.chain_index());
  const auto &parms = data.parms();
  const auto degree = parms.poly_modulus_degree();
  const auto limbs = parms.coeff_modulus().size();
  const auto slots = degree / 2;
  if (!std::isfinite(plain.scale()) || plain.scale() <= 0 ||
      std::log2(plain.scale()) >= data.total_coeff_modulus_bit_count())
    throw std::runtime_error("decode scale out of bounds");
  const auto stream = cudaStreamPerThread;
  auto copy = phantom::util::make_cuda_auto_ptr<std::uint64_t>(degree * limbs, stream);
  cudaCheck(cudaMemcpyAsync(copy.get(), plain.data(), degree * limbs * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToDevice, stream));
  nwt_2d_radix8_backward_inplace(copy.get(), context.gpu_rns_tables(), limbs, 0, stream);
  std::vector<std::uint64_t> coefficients(degree * limbs);
  cudaCheck(cudaMemcpyAsync(coefficients.data(), copy.get(), coefficients.size() * sizeof(std::uint64_t),
                            cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaStreamSynchronize(stream));

  phantom::arith::RNSBase base(parms.coeff_modulus());
  base.compose_array(coefficients.data(), degree);
  const auto &threshold = data.upper_half_threshold();
  std::vector<std::uint64_t> magnitude(limbs);
  std::vector<cuDoubleComplex> message(slots);
  for (std::size_t i = 0; i < degree; ++i) {
    const auto *value = coefficients.data() + i * limbs;
    const bool negative = phantom::arith::is_greater_than_or_equal_uint(value, threshold.data(), limbs);
    if (negative) {
      phantom::arith::sub_uint(base.big_modulus(), value, limbs, magnitude.data());
      value = magnitude.data();
    }
    double scaled = 0;
    for (std::size_t j = limbs; j-- > 0;)
      scaled = std::ldexp(scaled, 64) + static_cast<double>(value[j]);
    scaled = (negative ? -scaled : scaled) / plain.scale();
    if (i < slots) message[i].x = scaled;
    else message[i - slots].y = scaled;
  }
  auto &gpu = encoder.gpu_ckks_msg_vec();
  cudaCheck(cudaMemcpyAsync(gpu.in(), message.data(), slots * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice, stream));
  const auto logSlots = phantom::arith::get_power_of_two(slots);
  special_fft_forward(gpu, logSlots, stream);
  cudaCheck(cudaMemcpyAsync(message.data(), gpu.in(), slots * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaStreamSynchronize(stream));
  std::vector<double> result(slots);
  for (std::size_t i = 0; i < slots; ++i) {
    std::size_t reversed = 0, index = i;
    for (int bit = 0; bit < logSlots; ++bit, index >>= 1)
      reversed = (reversed << 1) | (index & 1);
    result[reversed] = message[i].x;
  }
  return result;
}
} // namespace crossfhe
