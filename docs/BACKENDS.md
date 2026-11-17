# Backend and numerical contracts

## Microsoft SEAL

`seal-ckks` accepts scalar RLWE add/subtract/multiply and public constants. It rejects LWE and switching. It uses one scalar per ciphertext, scale `2^40`, coefficient bits `[60] + [40] * depth + [60]`, and chooses a ring dimension accepted by SEAL's `tc128` policy up to 32768. Relinearization, level alignment and rescaling are required execution maintenance and are counted.

The scale reset to `2^40` follows the selected near-power-of-two primes; it introduces approximation error. No compiler noise bound is implemented. The decrypted result must be validated at the application's precision target.

## Phantom CUDA CKKS (experimental)

`phantom-ckks` implements scalar CKKS arithmetic and periodic packed MatVec on NVIDIA GPUs using the pinned PhantomFHE 2.1 library. It uses scale `2^40`, one special modulus, and the same coefficient-bit planning and tc128 table bound as SEAL, with dimensions capped at 32768. It rejects logic/switching, and does not implement bootstrapping. The four runner phases use separate processes and keep the secret key in the client directory. The runtime synchronizes CUDA around evaluator timing. Artifact formats are specific to Phantom and assume trusted local files. See [build instructions and detailed contracts](PHANTOM_BACKEND.md).

## OpenFHE CKKS/FHEW

`openfhe-cross` executes CKKS arithmetic and a cross-scheme sign circuit. It uses FLEXIBLEAUTO CKKS, 50-bit scaling moduli, a 60-bit first modulus, HYBRID key switching and one logical slot per ciphertext. The compiler reserves 17 switching levels in addition to the source arithmetic depth. This conservative budget is not an optimized bootstrap plan; deeper networks may need new resource planning and CKKS refresh support.

The default profile requests `HEStd_128_classic` and FHEW `STD128`; OpenFHE selects the CKKS ring size. `--insecure-demo` selects `HEStd_NotSet`, ring dimension 8192, and FHEW `TOY`. That profile has no production security claim. OpenFHE's selected parameter policy does not prove the security of the whole application or address access control, side channels, authenticated computation, malicious servers, or related-key/circular-security assumptions of evaluation keys.

The conversion follows OpenFHE's public comparison implementation:

```cpp
cc->EvalCompareSwitchPrecompute(pLWE, comparisonScale, false);
auto lwe = cc->EvalCKKStoFHEW(difference, 1);
auto sign = cc->GetBinCCForSchemeSwitch()->EvalSign(lwe[0], true);
auto mask = cc->EvalFHEWtoCKKS(signs, 1, 1, 4, -1.0, 1.0, 0);
```

`logQ_LWE = 25`, `pLWE = 2^25 / (2 * beta)`, and the default comparison scale is 1024. The LWE phase represents approximately `round(difference * comparisonScale) mod pLWE`. Thus a signed comparison requires approximately:

```text
abs(difference * comparisonScale) < pLWE / 2
abs(difference) > quantization + CKKS/conversion uncertainty
```

Both conditions apply to each internal comparison, not only the initial inputs. A larger scale improves small-margin resolution but reduces the unwrapped signed range. Scale must be calibrated against public bounds or a representative client dataset. Current code records the scale and has no static interval proof, so it cannot certify ordering near zero or outside the signed range. Equality, <= and >= are deliberately unsupported.

The reverse conversion supports the sign/Boolean encoding used for a CKKS selection mask. It is not a generic arbitrary-real-valued LUT return path. Sigmoid, GELU, Softmax and LayerNorm need a separately designed numerical/quantization contract and are rejected today.

## Artifacts and threat model

The client generates CKKS/FHEW secrets and public conversion keys. Only the CKKS secret is needed to decrypt final results; the LWE secret remains local during key generation and is discarded. The server loads public/evaluation keys and encrypted inputs, executes the generated static circuit and serializes encrypted outputs.

Artifacts use the libraries' binary research formats and assume trusted local files. They are not hardened network parsers. Tensor dimensions, weights, graph structure, circuit size and timings are public. Input data and returned activations are encrypted. This prototype does not hide model weights or access patterns, prove evaluation correctness, or implement transport authentication.

Breakdown reports distinguish evaluator time from process wall time (including key loading), CKKS→LWE time, sign evaluation time and LWE→CKKS time. `logic_bootstrap` counts calls to the library's sign evaluation, which can contain several internal bootstraps; it is not the exact number of low-level blind rotations.
