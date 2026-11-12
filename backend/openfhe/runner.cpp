#include "openfhe_runtime.hpp"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "binfhecontext-ser.h"
#include CROSSFHE_MODEL_HEADER
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

namespace fs=std::filesystem;
using namespace lbcrypto;
using crossfhe::Cipher;
constexpr unsigned logLWE=25;
constexpr std::uint64_t maxValues=50000;
template<class T> void save(const T &value,const fs::path &path) {
  std::ofstream stream(path,std::ios::binary);
  if (!stream) throw std::runtime_error("cannot create artifact");
  Serial::Serialize(value,stream,SerType::BINARY);
  if (!stream) throw std::runtime_error("artifact write failed");
}
template<class T> T load(const fs::path &path) {
  std::ifstream stream(path,std::ios::binary);
  if (!stream) throw std::runtime_error("missing artifact: "+path.string());
  T value; Serial::Deserialize(value,stream,SerType::BINARY);
  if (!stream) throw std::runtime_error("truncated artifact");
  return value;
}
static std::uint64_t readCount(std::istream &stream) {
  std::uint64_t n=0;
  stream.read(reinterpret_cast<char*>(&n),sizeof(n));
  if (!stream || n>maxValues) throw std::runtime_error("invalid vector count");
  return n;
}
static std::vector<double> readValues(const fs::path &path) {
  std::ifstream stream(path,std::ios::binary);
  auto n=readCount(stream);
  std::vector<double> out(n);
  stream.read(reinterpret_cast<char*>(out.data()),n*sizeof(double));
  if (!stream || stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("invalid plaintext vector");
  for (double v:out) if (!std::isfinite(v)) throw std::runtime_error("nonfinite input");
  return out;
}
static void writeValues(const std::vector<double> &values,const fs::path &path) {
  std::ofstream stream(path,std::ios::binary);
  std::uint64_t n=values.size();
  stream.write(reinterpret_cast<const char*>(&n),sizeof(n));
  stream.write(reinterpret_cast<const char*>(values.data()),n*sizeof(double));
  if (!stream) throw std::runtime_error("plaintext write failed");
}
static void checkDirectories(const fs::path &client,const fs::path &server) {
  auto a=fs::absolute(client).lexically_normal(), b=fs::absolute(server).lexically_normal();
  auto contained=[](const fs::path &child,const fs::path &parent) {
    auto relative=child.lexically_relative(parent);
    return !relative.empty() && *relative.begin()!="..";
  };
  if (a==b || contained(a,b) || contained(b,a))
    throw std::runtime_error("client and server directories must be separate siblings");
  if ((fs::exists(a) && !fs::is_empty(a)) || (fs::exists(b) && !fs::is_empty(b)))
    throw std::runtime_error("keygen requires empty directories to avoid stale key material");
  fs::create_directories(a); fs::create_directories(b);
}
static void saveEvaluationKeys(const CryptoContext<DCRTPoly> &cc,const fs::path &server) {
  std::ofstream stream(server/"mult.bin",std::ios::binary);
  if (!cc->SerializeEvalMultKey(stream,SerType::BINARY) || !stream)
    throw std::runtime_error("cannot serialize multiplication keys");
  if (!crossfhe_generated::requires_logic) return;
  std::ofstream rotations(server/"rotations.bin",std::ios::binary);
  if (!cc->SerializeEvalAutomorphismKey(rotations,SerType::BINARY) || !rotations)
    throw std::runtime_error("cannot serialize switching rotation keys");
  save(cc->GetSwkFC(),server/"logic_to_ckks.bin");
  auto bin=cc->GetBinCCForSchemeSwitch();
  save(bin,server/"logic_context.bin");
  save(bin->GetRefreshKey(),server/"refresh.bin");
  save(bin->GetSwitchKey(),server/"logic_switch.bin");
  std::vector<std::uint32_t> bases;
  auto bootstrapKeys=bin->GetBTKeyMap();
  for (const auto &pair:*bootstrapKeys) {
    bases.push_back(pair.first);
    save(pair.second.BSkey,server/("refresh_"+std::to_string(pair.first)+".bin"));
    save(pair.second.KSkey,server/("switch_"+std::to_string(pair.first)+".bin"));
  }
  save(bases,server/"bases.bin");
}
static void loadEvaluationKeys(const CryptoContext<DCRTPoly> &cc,const fs::path &server) {
  std::ifstream stream(server/"mult.bin",std::ios::binary);
  if (!cc->DeserializeEvalMultKey(stream,SerType::BINARY) || !stream)
    throw std::runtime_error("invalid multiplication keys");
  if (!crossfhe_generated::requires_logic) return;
  std::ifstream rotations(server/"rotations.bin",std::ios::binary);
  if (!cc->DeserializeEvalAutomorphismKey(rotations,SerType::BINARY) || !rotations)
    throw std::runtime_error("invalid switching rotation keys");
  cc->SetSwkFC(load<Cipher>(server/"logic_to_ckks.bin"));
  auto bin=load<std::shared_ptr<BinFHEContext>>(server/"logic_context.bin");
  RingGSWBTKey key;
  key.BSkey=load<RingGSWACCKey>(server/"refresh.bin");
  key.KSkey=load<LWESwitchingKey>(server/"logic_switch.bin");
  bin->BTKeyLoad(key);
  for (auto base:load<std::vector<std::uint32_t>>(server/"bases.bin")) {
    RingGSWBTKey entry;
    entry.BSkey=load<RingGSWACCKey>(server/("refresh_"+std::to_string(base)+".bin"));
    entry.KSkey=load<LWESwitchingKey>(server/("switch_"+std::to_string(base)+".bin"));
    bin->BTKeyMapLoadSingleElement(base,entry);
  }
  cc->SetBinCCForSchemeSwitch(bin);
  double scale=load<double>(server/"comparison_scale.bin");
  if (!std::isfinite(scale) || scale<=0) throw std::runtime_error("invalid comparison scale");
  const auto p=(std::uint32_t{1}<<logLWE)/(2*bin->GetBeta().ConvertToInt());
  cc->EvalCompareSwitchPrecompute(p,scale,false);
}
static void keygen(const fs::path &client,const fs::path &server,bool insecure,double comparisonScale) {
  checkDirectories(client,server);
  if (!std::isfinite(comparisonScale) || comparisonScale<=0)
    throw std::runtime_error("comparison scale must be finite and positive");
  CCParams<CryptoContextCKKSRNS> parameters;
  auto security=insecure ? HEStd_NotSet : HEStd_128_classic;
  auto depth=std::max(2u,crossfhe_generated::openfhe_depth);
  parameters.SetSecurityLevel(security);
  parameters.SetMultiplicativeDepth(depth);
  parameters.SetScalingModSize(50); parameters.SetFirstModSize(60);
  parameters.SetScalingTechnique(FLEXIBLEAUTO);
  parameters.SetBatchSize(1);
  parameters.SetSecretKeyDist(UNIFORM_TERNARY);
  parameters.SetKeySwitchTechnique(HYBRID);
  parameters.SetNumLargeDigits(3);
  if (insecure) parameters.SetRingDim(8192);
  auto cc=GenCryptoContext(parameters);
  cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE);
  cc->Enable(SCHEMESWITCH);
  auto keys=cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  if (crossfhe_generated::requires_logic) {
    SchSwchParams switching;
    switching.SetSecurityLevelCKKS(security);
    switching.SetSecurityLevelFHEW(insecure ? TOY : STD128);
    switching.SetCtxtModSizeFHEWLargePrec(logLWE);
    switching.SetNumSlotsCKKS(1); switching.SetNumValues(1);
    auto lweSecret=cc->EvalSchemeSwitchingSetup(switching);
    cc->EvalSchemeSwitchingKeyGen(keys,lweSecret);
  }
  save(cc,client/"context.bin"); save(cc,server/"context.bin");
  save(keys.secretKey,client/"secret.bin"); save(keys.publicKey,server/"public.bin");
  saveEvaluationKeys(cc,server);
  save(comparisonScale,server/"comparison_scale.bin");
  std::cout << "{\"phase\":\"keygen\",\"backend\":\"openfhe-cross\",\"ring_dimension\":"
            << cc->GetRingDimension() << ",\"depth\":" << depth
            << ",\"security_profile\":\"" << (insecure ? "INSECURE_TOY" : "HEStd_128_classic+STD128")
            << "\",\"comparison_scale\":" << comparisonScale << "}\n";
}
static void encrypt(const fs::path &client,const fs::path &server,const fs::path &plain,const fs::path &output) {
  auto cc=load<CryptoContext<DCRTPoly>>(client/"context.bin");
  auto pk=load<PublicKey<DCRTPoly>>(server/"public.bin");
  if (pk->GetCryptoContext()!=cc) throw std::runtime_error("public key context mismatch");
  auto values=readValues(plain);
  if (values.size()!=crossfhe_generated::input_count) throw std::runtime_error("model input count mismatch");
  std::vector<Cipher> encrypted;
  for (double value:values) {
    auto pt=cc->MakeCKKSPackedPlaintext(std::vector<double>{value},1,0,nullptr,1);
    encrypted.push_back(cc->Encrypt(pk,pt));
  }
  save(encrypted,output);
}
static void evaluate(const fs::path &server,const fs::path &input,const fs::path &output) {
  if (fs::exists(server/"secret.bin")) throw std::runtime_error("server directory contains a secret key");
  auto cc=load<CryptoContext<DCRTPoly>>(server/"context.bin");
  auto pk=load<PublicKey<DCRTPoly>>(server/"public.bin");
  loadEvaluationKeys(cc,server);
  auto values=load<std::vector<Cipher>>(input);
  if (values.size()!=crossfhe_generated::input_count) throw std::runtime_error("model ciphertext count mismatch");
  for (const auto &ct:values)
    if (!ct || ct->GetCryptoContext()!=cc || ct->GetKeyTag()!=pk->GetKeyTag())
      throw std::runtime_error("ciphertext context/key mismatch");
  crossfhe::Runtime runtime(cc,pk);
  auto start=std::chrono::steady_clock::now();
  auto results=crossfhe_generated::evaluate(runtime,values);
  auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  save(results,output);
  const auto &c=runtime.counts;
  std::cout << std::setprecision(12) << "{\"phase\":\"evaluate\",\"seconds\":" << seconds
            << ",\"add\":" << c.add << ",\"sub\":" << c.sub
            << ",\"multiply_plain\":" << c.multiply_plain << ",\"multiply_cipher\":" << c.multiply_cipher
            << ",\"ckks_to_logic\":" << c.ckks_to_logic << ",\"logic_bootstrap\":" << c.logic_bootstrap
            << ",\"logic_to_ckks\":" << c.logic_to_ckks << ",\"select\":" << c.select
            << ",\"ckks_to_logic_seconds\":" << c.ckks_to_logic_seconds
            << ",\"logic_seconds\":" << c.logic_seconds
            << ",\"logic_to_ckks_seconds\":" << c.logic_to_ckks_seconds << "}\n";
}
static void decrypt(const fs::path &client,const fs::path &input,const fs::path &output) {
  auto cc=load<CryptoContext<DCRTPoly>>(client/"context.bin");
  auto sk=load<PrivateKey<DCRTPoly>>(client/"secret.bin");
  auto encrypted=load<std::vector<Cipher>>(input);
  if (encrypted.size()!=crossfhe_generated::output_count) throw std::runtime_error("output count mismatch");
  std::vector<double> values;
  for (const auto &ct:encrypted) {
    if (!ct || ct->GetCryptoContext()!=cc || ct->GetKeyTag()!=sk->GetKeyTag())
      throw std::runtime_error("output context/key mismatch");
    Plaintext pt; cc->Decrypt(sk,ct,&pt); pt->SetLength(1);
    double value=pt->GetRealPackedValue().at(0);
    if (!std::isfinite(value)) throw std::runtime_error("decrypted nonfinite result");
    values.push_back(value);
  }
  writeValues(values,output);
}
int main(int argc,char **argv) {
  try {
    if (argc<2) throw std::runtime_error("usage: runner keygen|encrypt|evaluate|decrypt ...");
    std::string phase=argv[1];
    if (phase=="keygen" && argc>=4) {
      bool insecure=false; double scale=1024;
      for (int i=4;i<argc;++i) {
        std::string arg=argv[i];
        if (arg=="--insecure-demo") insecure=true;
        else if (arg=="--comparison-scale" && i+1<argc) scale=std::stod(argv[++i]);
        else throw std::runtime_error("unknown keygen option");
      }
      keygen(argv[2],argv[3],insecure,scale);
    } else if (phase=="encrypt" && argc==6) encrypt(argv[2],argv[3],argv[4],argv[5]);
    else if (phase=="evaluate" && argc==5) evaluate(argv[2],argv[3],argv[4]);
    else if (phase=="decrypt" && argc==5) decrypt(argv[2],argv[3],argv[4]);
    else throw std::runtime_error("invalid phase or arguments");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "CrossFHE error: " << e.what() << "\n"; return 1;
  }
}
