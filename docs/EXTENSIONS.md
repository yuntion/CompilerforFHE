# Extension points

## Scalar FHE optimization pipeline

`compiler/Pipelines/FHE/OptimizationHooks.cpp` now composes handwritten scalar FHE simplification, structural CSE, DCE and resource-contract analysis. Multiplication-tree balancing and bounded quadratic ReLU approximation require explicit opt-in. See [implemented passes](development/PASSES.md), [literature review](development/ARTICLES.md) and [validation](development/RESULTS.md). The driver records native resource contracts before/after optimization and checks encrypted results; it no longer requires byte-identical IR. Libra ODS canonicalizers remain separate from these passes.

The explicit SEAL CPU single-affine MatVec path now supports periodic CKKS packing,
diagonal/BSGS lowering, sparse-diagonal elimination and planned rotation keys.
See [implementation and encrypted measurements](development/MATVEC.md).

Natural PPML targets for later evaluation include:

1. General network layout assignment, multiple-vector batching, convolution and attention packing beyond the implemented single affine MatVec.
2. Switch-region formation, batches of activation comparisons and repeated mask reuse.
3. Public range analysis and per-layer sign quantization/scale selection.
4. Level, rescale and relinearization placement with a numerical error budget.
5. Multi-level vectorization and control-path reduction following CHLOE.
6. Backend pattern/scheduling choice following Libra, with measured GPU costs.

These larger tensor/layout/maintenance projects remain research work; repeated scalar mask reuse is already implemented by the current CSE pass. First establish a baseline with identical graph semantics, dataset, precision and security profile, then compare each optimization against it.

## Replacing the library backend

`compiler/CodeGen/FHE/CodeGen.cpp` verifies FHE MLIR, checks the backend capability set, and emits a model-specific C++ evaluator through the reusable `generateCpp` API. `tools/crossfhe-translate.cpp` handles command-line parsing and files. Runtime adapters under `backend/` implement `constant`, `add`, `sub`, `mul`, `to_logic`, `sign`, `to_ckks_mask` and `select`.

SEAL and OpenFHE use different ciphertext types and parameter contexts. A TFHEpp/HALO/HE³DB backend must generate its own conversion keys and implement the actual arithmetic↔logic mapping. Linking two libraries and reinterpreting ciphertext buffers is not scheme switching. Add a new explicit backend name and refuse unsupported encodings during translation.

## Future AI-assisted GPU code generation

The generated GPU implementation should start from verified, typed FHE MLIR and explicit library/kernel contracts. A proposed kernel needs a reference-library differential test, parameter/precision checks, and device memory/timing validation before being enabled.

An agent may propose source code, but it must not supply key material, decrypt intermediate values, invent primitive costs, or bypass legality/resource checks. Keep emitted kernel sources, compiler options, device metadata and test reports with each artifact. The current CLI rejects GPU backend names; there is no placeholder GPU executor returning plaintext or identity results.

The AI backend design and working offline task preparer are described in [AI_BACKEND.md](development/AI_BACKEND.md). Online model calls, candidate execution and automatic promotion are not implemented.
