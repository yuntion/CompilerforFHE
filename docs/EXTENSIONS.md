# Extension points

## Reserved optimization pipeline

`compiler/Pipelines/FHE/OptimizationHooks.cpp` registers `--ppml-fhe-optimize` as an identity module pass. The regression suite checks that it preserves the FHE IR. It is the place to add the user-selected optimization sequence later. Libra ODS canonicalizers also have empty implementations; none of the published optimization pipelines has been silently enabled.

Natural PPML targets for later evaluation include:

1. Tensor layout and CKKS packing for Gemm, convolution and attention projections.
2. Switch-region formation, batches of activation comparisons and repeated mask reuse.
3. Public range analysis and per-layer sign quantization/scale selection.
4. Level, rescale and relinearization placement with a numerical error budget.
5. Multi-level vectorization and control-path reduction following CHLOE.
6. Backend pattern/scheduling choice following Libra, with measured GPU costs.

These are research hooks, not implemented passes or contribution claims. First establish a baseline with identical graph semantics, dataset, precision and security profile, then compare each optimization against it.

## Replacing the library backend

`compiler/CodeGen/FHE/CodeGen.cpp` verifies FHE MLIR, checks the backend capability set, and emits a model-specific C++ evaluator through the reusable `generateCpp` API. `tools/crossfhe-translate.cpp` handles command-line parsing and files. Runtime adapters under `backend/` implement `constant`, `add`, `sub`, `mul`, `to_logic`, `sign`, `to_ckks_mask` and `select`.

SEAL and OpenFHE use different ciphertext types and parameter contexts. A TFHEpp/HALO/HE³DB backend must generate its own conversion keys and implement the actual arithmetic↔logic mapping. Linking two libraries and reinterpreting ciphertext buffers is not scheme switching. Add a new explicit backend name and refuse unsupported encodings during translation.

## Future AI-assisted GPU code generation

The generated GPU implementation should start from verified, typed FHE MLIR and explicit library/kernel contracts. A proposed kernel needs a reference-library differential test, parameter/precision checks, and device memory/timing validation before being enabled.

An agent may propose source code, but it must not supply key material, decrypt intermediate values, invent primitive costs, or bypass legality/resource checks. Keep emitted kernel sources, compiler options, device metadata and test reports with each artifact. The current CLI rejects GPU backend names; there is no placeholder GPU executor returning plaintext or identity results.
