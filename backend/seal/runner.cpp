#include "seal_runtime.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#ifdef CROSSFHE_MODEL_HEADER
#include CROSSFHE_MODEL_HEADER
#else
#error "CROSSFHE_MODEL_HEADER must name a generated model; there is no default evaluator"
#endif

namespace fs = std::filesystem;
using namespace seal;
static constexpr std::uint64_t maxValues = 1000000;
template<class T> void save(const T &value, const fs::path &path) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot create " + path.string());
  value.save(stream, compr_mode_type::none);
}
template<class T> void load(T &value, const SEALContext &context, const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read " + path.string());
  value.load(context, stream);
  if (!stream) throw std::runtime_error("truncated artifact");
}
static std::uint64_t readCount(std::istream &stream) {
  std::uint64_t count = 0;
  stream.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (!stream || count > maxValues) throw std::runtime_error("invalid vector count");
  return count;
}
static void writeCount(std::ostream &stream, std::uint64_t count) {
  stream.write(reinterpret_cast<const char *>(&count), sizeof(count));
}
static std::vector<double> readValues(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open input values");
  auto count = readCount(stream);
  std::vector<double> values(count);
  stream.read(reinterpret_cast<char *>(values.data()), count * sizeof(double));
  if (!stream) throw std::runtime_error("truncated value file");
  for (double value : values)
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite input");
  return values;
}
static void writeValues(const std::vector<double> &values, const fs::path &path) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot create value file");
  writeCount(stream, values.size());
  stream.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(double));
}
static EncryptionParameters readParameters(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  EncryptionParameters parameters;
  parameters.load(stream);
  if (!stream) throw std::runtime_error("invalid parameter file");
  return parameters;
}
static void keygen(const fs::path &client, const fs::path &server) {
  fs::create_directories(client);
  fs::create_directories(server);
  unsigned levels = std::max(1u, crossfhe_generated::depth);
  std::vector<int> bits(levels + 2, 40);
  bits.front() = bits.back() = 60;
  std::size_t degree = 4096;
  unsigned total = 120 + 40 * levels;
  while (CoeffModulus::MaxBitCount(degree, sec_level_type::tc128) < total ||
         degree/2 < crossfhe_generated::packed_period) {
    degree *= 2;
    if (degree > 32768) throw std::runtime_error("model depth exceeds supported 128-bit CKKS policy");
  }
  EncryptionParameters parameters(scheme_type::ckks);
  parameters.set_poly_modulus_degree(degree);
  parameters.set_coeff_modulus(CoeffModulus::Create(degree, bits));
  SEALContext context(parameters, true, sec_level_type::tc128);
  if (!context.parameters_set()) throw std::runtime_error(context.parameter_error_message());
  KeyGenerator generator(context);
  PublicKey publicKey; RelinKeys relin;
  generator.create_public_key(publicKey);
  generator.create_relin_keys(relin);
  save(parameters, client / "parameters.seal");
  save(parameters, server / "parameters.seal");
  save(generator.secret_key(), client / "secret.seal");
  save(publicKey, server / "public.seal");
  save(relin, server / "relin.seal");
  if (!crossfhe_generated::rotation_steps.empty()) {
    GaloisKeys galois;
    generator.create_galois_keys(crossfhe_generated::rotation_steps,galois);
    save(galois, server / "galois.seal");
  }
  std::cout << "{\"phase\":\"keygen\",\"ring_dimension\":" << degree
            << ",\"depth\":" << levels << ",\"security_bits\":128"
            << ",\"slots\":" << degree/2
            << ",\"rotation_keys\":" << crossfhe_generated::rotation_steps.size() << "}\n";
}
static void encrypt(const fs::path &client, const fs::path &server,
                    const fs::path &plain, const fs::path &cipher) {
  auto parameters = readParameters(client / "parameters.seal");
  SEALContext context(parameters, true, sec_level_type::tc128);
  PublicKey key; load(key, context, server / "public.seal");
  Encryptor encryptor(context, key);
  CKKSEncoder encoder(context);
  auto values = readValues(plain);
  if (values.size() != crossfhe_generated::input_value_count)
    throw std::runtime_error("input count does not match compiled model");
  std::ofstream stream(cipher, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot create ciphertext file");
  writeCount(stream, crossfhe_generated::input_count);
  if (crossfhe_generated::packed_period) {
    const auto period=crossfhe_generated::packed_period;
    if (period>encoder.slot_count() || encoder.slot_count()%period)
      throw std::runtime_error("packed period does not fit physical slots");
    std::vector<double> packed(encoder.slot_count(),0);
    for (std::size_t i=0; i<packed.size(); ++i)
      if (i%period<values.size()) packed[i]=values[i%period];
    Plaintext plaintext; Ciphertext encrypted;
    encoder.encode(packed,std::pow(2.0,40),plaintext);
    encryptor.encrypt(plaintext,encrypted);
    encrypted.save(stream,compr_mode_type::none);
    return;
  }
  for (double value : values) {
    Plaintext plaintext; Ciphertext encrypted;
    encoder.encode(value, std::pow(2.0, 40), plaintext);
    encryptor.encrypt(plaintext, encrypted);
    encrypted.save(stream, compr_mode_type::none);
  }
}
static void evaluate(const fs::path &server, const fs::path &input, const fs::path &output) {
  auto parameters = readParameters(server / "parameters.seal");
  SEALContext context(parameters, true, sec_level_type::tc128);
  PublicKey key; RelinKeys relin;
  load(key, context, server / "public.seal");
  load(relin, context, server / "relin.seal");
  std::ifstream stream(input, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read encrypted inputs");
  auto count = readCount(stream);
  if (count != crossfhe_generated::input_count) throw std::runtime_error("ciphertext count mismatch");
  std::vector<Ciphertext> values(count);
  for (auto &value : values) {
    value.load(context, stream);
    if (!stream) throw std::runtime_error("truncated ciphertext file");
  }
  GaloisKeys galois;
  if (!crossfhe_generated::rotation_steps.empty()) load(galois,context,server / "galois.seal");
  crossfhe::SealRuntime runtime(context, key, relin, std::pow(2.0, 40), &galois);
  auto start = std::chrono::steady_clock::now();
  auto results = crossfhe_generated::evaluate(runtime, values);
  auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::ofstream destination(output, std::ios::binary);
  if (!destination) throw std::runtime_error("cannot create encrypted outputs");
  writeCount(destination, results.size());
  for (const auto &value : results) value.save(destination, compr_mode_type::none);
  std::cout << "{\"phase\":\"evaluate\",\"seconds\":" << seconds
            << ",\"add\":" << runtime.counts.add
            << ",\"add_plain\":" << runtime.counts.add_plain
            << ",\"rotations\":" << runtime.counts.rotations
            << ",\"materializations\":" << runtime.counts.materializations
            << ",\"input_ciphertexts\":" << values.size()
            << ",\"output_ciphertexts\":" << results.size()
            << ",\"multiply_plain\":" << runtime.counts.multiply_plain
            << ",\"multiply_cipher\":" << runtime.counts.multiply_cipher
            << ",\"rescale\":" << runtime.counts.rescale
            << ",\"mod_switch\":" << runtime.counts.mod_switch
            << ",\"relinearize\":" << runtime.counts.relinearize << "}\n";
}
static void decrypt(const fs::path &client, const fs::path &input, const fs::path &output) {
  auto parameters = readParameters(client / "parameters.seal");
  SEALContext context(parameters, true, sec_level_type::tc128);
  SecretKey key; load(key, context, client / "secret.seal");
  Decryptor decryptor(context, key); CKKSEncoder encoder(context);
  std::ifstream stream(input, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read encrypted output");
  auto count = readCount(stream);
  if (count != crossfhe_generated::output_count) throw std::runtime_error("output count mismatch");
  std::vector<double> results;
  for (std::uint64_t i = 0; i < count; ++i) {
    Ciphertext cipher; Plaintext plain; std::vector<double> decoded;
    cipher.load(context, stream);
    decryptor.decrypt(cipher, plain);
    encoder.decode(plain, decoded);
    if (crossfhe_generated::packed_period) {
      if (crossfhe_generated::output_value_count>decoded.size())
        throw std::runtime_error("output elements exceed physical slots");
      results.insert(results.end(),decoded.begin(),decoded.begin()+crossfhe_generated::output_value_count);
    } else results.push_back(decoded.at(0));
  }
  writeValues(results, output);
}
int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: runner keygen|encrypt|evaluate|decrypt ...");
    std::string phase = argv[1];
    if (phase == "keygen" && argc == 4) keygen(argv[2], argv[3]);
    else if (phase == "encrypt" && argc == 6) encrypt(argv[2], argv[3], argv[4], argv[5]);
    else if (phase == "evaluate" && argc == 5) evaluate(argv[2], argv[3], argv[4]);
    else if (phase == "decrypt" && argc == 5) decrypt(argv[2], argv[3], argv[4]);
    else throw std::runtime_error("invalid phase or argument count");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CrossFHE error: " << error.what() << "\n";
    return 1;
  }
}
