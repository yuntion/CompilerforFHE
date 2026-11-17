#include "phantom_runtime.hpp"
#include "phantom_decode.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include CROSSFHE_MODEL_HEADER

namespace fs = std::filesystem;
using namespace phantom;
using namespace phantom::arith;
namespace model = crossfhe_generated;
static constexpr std::uint64_t maxValues=1000000;
// These are trusted local research artifacts, using the pinned library's native payloads.
static std::string header(const std::string &kind) {
  return "CROSSFHE_PHANTOM_V1 1f4a198443b3af77118e51f53d5b8f332154b875 "
         + std::string(CROSSFHE_MODEL_ID) + " " + kind;
}
static std::ofstream outputFile(const fs::path &path, const std::string &kind) {
  std::ofstream stream(path,std::ios::binary);
  if (!stream) throw std::runtime_error("cannot create " + path.string());
  stream << header(kind) << '\n';
  return stream;
}
static std::ifstream inputFile(const fs::path &path, const std::string &kind) {
  if (!fs::is_regular_file(path) || fs::file_size(path)>std::uint64_t(2)*1024*1024*1024)
    throw std::runtime_error("missing or oversized artifact: " + path.string());
  std::ifstream stream(path,std::ios::binary);
  const auto expected=header(kind)+"\n";
  std::string actual(expected.size(),'\0');
  stream.read(actual.data(),actual.size());
  if (!stream || actual!=expected)
    throw std::runtime_error("artifact backend/version/model/kind mismatch: " + path.string());
  return stream;
}
static void finished(std::istream &stream) {
  if (!stream || stream.peek()!=std::char_traits<char>::eof())
    throw std::runtime_error("truncated artifact or trailing data");
}
template<class T> static void save(const T &value, const fs::path &path, const std::string &kind) {
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  auto stream=outputFile(path,kind);
  value.save(stream);
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  if (!stream) throw std::runtime_error("artifact write failed");
}
template<class T> static void load(T &value, const fs::path &path, const std::string &kind) {
  auto stream=inputFile(path,kind);
  value.load(stream);
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  finished(stream);
}
static std::uint64_t readCount(std::istream &stream) {
  std::uint64_t count=0;
  stream.read(reinterpret_cast<char *>(&count),sizeof(count));
  if (!stream || count>maxValues) throw std::runtime_error("invalid vector count");
  return count;
}
static void writeCount(std::ostream &stream,std::uint64_t count) {
  stream.write(reinterpret_cast<const char *>(&count),sizeof(count));
}
static std::vector<double> readValues(const fs::path &path) {
  std::ifstream stream(path,std::ios::binary);
  auto count=readCount(stream);
  std::vector<double> values(count);
  stream.read(reinterpret_cast<char *>(values.data()),count*sizeof(double));
  finished(stream);
  for (double v:values) if (!std::isfinite(v)) throw std::runtime_error("nonfinite input");
  return values;
}
static void writeValues(const std::vector<double> &values,const fs::path &path) {
  std::ofstream stream(path,std::ios::binary);
  writeCount(stream,values.size());
  stream.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(double));
  if (!stream) throw std::runtime_error("cannot write plaintext output");
}
static std::size_t degreeForModel() {
  const unsigned levels=std::max(1u,model::depth);
  const unsigned total=120+40*levels;
  std::size_t degree=4096;
  while (CoeffModulus::MaxBitCount(degree,sec_level_type::tc128)<total ||
         degree/2<std::max(std::size_t(1),model::packed_period)) {
    degree*=2;
    if (degree>32768) throw std::runtime_error("model depth exceeds supported 128-bit CKKS policy");
  }
  return degree;
}
static EncryptionParameters parametersForModel() {
  const auto degree=degreeForModel();
  std::vector<int> bits(std::max(1u,model::depth)+2,40);
  bits.front()=bits.back()=60;
  EncryptionParameters parameters(scheme_type::ckks);
  parameters.set_poly_modulus_degree(degree);
  parameters.set_special_modulus_size(1);
  parameters.set_coeff_modulus(CoeffModulus::Create(degree,bits));
  if (!model::rotation_steps.empty())
    parameters.set_galois_elts(phantom::util::get_elts_from_steps(model::rotation_steps,degree));
  return parameters;
}
static void saveParameters(const EncryptionParameters &params,const fs::path &path) {
  auto stream=outputFile(path,"parameters");
  stream << params.poly_modulus_degree() << ' ' << params.coeff_modulus().size() << '\n';
  for (auto &q:params.coeff_modulus()) stream << q.value() << '\n';
  if (!stream) throw std::runtime_error("cannot write parameters");
}
static EncryptionParameters readParameters(const fs::path &path) {
  auto stream=inputFile(path,"parameters");
  auto parameters=parametersForModel();
  std::size_t degree=0,count=0;
  stream >> degree >> count;
  if (degree!=parameters.poly_modulus_degree() || count!=parameters.coeff_modulus().size())
    throw std::runtime_error("parameter contract mismatch");
  for (auto &q:parameters.coeff_modulus()) {
    std::uint64_t value=0; stream >> value;
    if (!stream || value!=q.value()) throw std::runtime_error("coefficient modulus mismatch");
  }
  stream >> std::ws;
  if (stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing parameter data");
  return parameters;
}
static void keygen(const fs::path &client,const fs::path &server) {
  fs::create_directories(client); fs::create_directories(server);
  auto params=parametersForModel();
  PhantomContext context(params);
  PhantomSecretKey secret(context);
  auto publicKey=secret.gen_publickey(context);
  auto relin=secret.gen_relinkey(context);
  saveParameters(params,client/"parameters.phantom");
  saveParameters(params,server/"parameters.phantom");
  // Save after evaluation key generation, which can extend the secret key's powers.
  save(secret,client/"secret.phantom","secret");
  save(publicKey,server/"public.phantom","public");
  save(relin,server/"relin.phantom","relin");
  if (!model::rotation_steps.empty()) {
    auto galois=secret.create_galois_keys(context);
    save(galois,server/"galois.phantom","galois");
  }
  std::cout << "{\"phase\":\"keygen\",\"ring_dimension\":" << params.poly_modulus_degree()
            << ",\"depth\":" << std::max(1u,model::depth) << ",\"security_bits\":128"
            << ",\"slots\":" << params.poly_modulus_degree()/2
            << ",\"rotation_keys\":" << model::rotation_steps.size() << "}\n";
}
static void encrypt(const fs::path &client,const fs::path &server,const fs::path &plain,const fs::path &cipher) {
  PhantomContext context(readParameters(client/"parameters.phantom"));
  PhantomPublicKey key; load(key,server/"public.phantom","public");
  PhantomCKKSEncoder encoder(context);
  auto values=readValues(plain);
  if (values.size()!=model::input_value_count) throw std::runtime_error("input count mismatch");
  auto stream=outputFile(cipher,"input");
  writeCount(stream,model::input_count);
  auto encryptOne=[&](const std::vector<double> &slots) {
    auto plaintext=encoder.encode(context,slots,crossfhe::Runtime::scale);
    auto encrypted=key.encrypt_asymmetric(context,plaintext);
    crossfhe::cudaCheck(cudaDeviceSynchronize());
    encrypted.save(stream);
  };
  if constexpr (model::packed_period!=0) {
    const auto period=std::max(std::size_t(1),model::packed_period);
    if (encoder.slot_count()%period) throw std::runtime_error("invalid packed period");
    std::vector<double> slots(encoder.slot_count(),0);
    for (std::size_t i=0;i<slots.size();++i)
      if (i%period<values.size()) slots[i]=values[i%period];
    encryptOne(slots);
  } else for (double value:values) encryptOne(std::vector<double>(encoder.slot_count(),value));
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  if (!stream) throw std::runtime_error("ciphertext write failed");
}
static std::vector<PhantomCiphertext> readCiphers(const fs::path &path,const std::string &kind,
                                                std::size_t expected,const PhantomContext &context) {
  auto stream=inputFile(path,kind);
  auto count=readCount(stream);
  if (count!=expected) throw std::runtime_error("ciphertext count mismatch");
  std::vector<PhantomCiphertext> values(count);
  for (auto &value:values) {
    // Validate native ciphertext dimensions and remaining length before upstream allocates GPU memory.
    const auto position=stream.tellg();
    std::size_t level=0,size=0,degree=0,moduli=0;
    stream.read(reinterpret_cast<char *>(&level),sizeof(level));
    stream.read(reinterpret_cast<char *>(&size),sizeof(size));
    stream.read(reinterpret_cast<char *>(&degree),sizeof(degree));
    stream.read(reinterpret_cast<char *>(&moduli),sizeof(moduli));
    if (!stream || level<context.get_first_index() || level>=context.total_parm_size() || size!=2)
      throw std::runtime_error("invalid ciphertext header");
    const auto &params=context.get_context_data(level).parms();
    const auto bytes=5*sizeof(std::size_t)+sizeof(double)+sizeof(std::uint64_t)+2*sizeof(bool)
                     +size*degree*moduli*sizeof(std::uint64_t);
    if (degree!=params.poly_modulus_degree() || moduli!=params.coeff_modulus().size() ||
        fs::file_size(path)-static_cast<std::uint64_t>(position)<bytes)
      throw std::runtime_error("invalid or truncated ciphertext dimensions");
    stream.seekg(position);
    value.load(stream);
    if (!stream || !value.is_ntt_form() || !std::isfinite(value.scale()) || value.scale()<=0)
      throw std::runtime_error("invalid ciphertext payload");
  }
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  finished(stream);
  return values;
}
static void evaluate(const fs::path &server,const fs::path &input,const fs::path &output) {
  PhantomContext context(readParameters(server/"parameters.phantom"));
  PhantomPublicKey key; PhantomRelinKey relin; PhantomGaloisKey galois;
  load(key,server/"public.phantom","public");
  load(relin,server/"relin.phantom","relin");
  if (!model::rotation_steps.empty()) load(galois,server/"galois.phantom","galois");
  auto values=readCiphers(input,"input",model::input_count,context);
  crossfhe::Runtime runtime(context,key,relin,model::rotation_steps.empty()?nullptr:&galois);
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  auto start=std::chrono::steady_clock::now();
  auto results=model::evaluate(runtime,values);
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  auto destination=outputFile(output,"output");
  writeCount(destination,results.size());
  for (auto &value:results) value.save(destination);
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  if (!destination) throw std::runtime_error("ciphertext write failed");
  std::cout << "{\"phase\":\"evaluate\",\"seconds\":" << seconds
            << ",\"add\":" << runtime.counts.add << ",\"sub\":" << runtime.counts.sub
            << ",\"add_plain\":" << runtime.counts.add_plain
            << ",\"rotations\":" << runtime.counts.rotations
            << ",\"materializations\":" << runtime.counts.materializations
            << ",\"input_ciphertexts\":" << values.size() << ",\"output_ciphertexts\":" << results.size()
            << ",\"multiply_plain\":" << runtime.counts.multiply_plain
            << ",\"multiply_cipher\":" << runtime.counts.multiply_cipher
            << ",\"rescale\":" << runtime.counts.rescale << ",\"mod_switch\":" << runtime.counts.mod_switch
            << ",\"relinearize\":" << runtime.counts.relinearize << "}\n";
}
static void decrypt(const fs::path &client,const fs::path &input,const fs::path &output) {
  PhantomContext context(readParameters(client/"parameters.phantom"));
  PhantomSecretKey secret; load(secret,client/"secret.phantom","secret");
  PhantomCKKSEncoder encoder(context);
  auto values=readCiphers(input,"output",model::output_count,context);
  std::vector<double> results;
  for (auto &value:values) {
    auto plain=secret.decrypt(context,value);
    auto decoded=crossfhe::decodeClient(context,encoder,plain);
    auto count=model::packed_period?model::output_value_count:1;
    if (count>decoded.size()) throw std::runtime_error("output count exceeds physical slots");
    results.insert(results.end(),decoded.begin(),decoded.begin()+count);
  }
  crossfhe::cudaCheck(cudaDeviceSynchronize());
  writeValues(results,output);
}
int main(int argc,char **argv) {
  try {
    int devices=0; crossfhe::cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("phantom-ckks requires an NVIDIA CUDA GPU");
    if (argc<2) throw std::runtime_error("usage: runner keygen|encrypt|evaluate|decrypt ...");
    const std::string phase=argv[1];
    if (phase=="keygen" && argc==4) keygen(argv[2],argv[3]);
    else if (phase=="encrypt" && argc==6) encrypt(argv[2],argv[3],argv[4],argv[5]);
    else if (phase=="evaluate" && argc==5) evaluate(argv[2],argv[3],argv[4]);
    else if (phase=="decrypt" && argc==5) decrypt(argv[2],argv[3],argv[4]);
    else throw std::runtime_error("invalid phase or argument count");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CrossFHE Phantom error: " << error.what() << '\n';
    return 1;
  }
}
