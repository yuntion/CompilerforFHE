# Upstream provenance

| Project | Pinned revision | Reuse |
| --- | --- | --- |
| [Compile_via_MLIR](https://github.com/Loki-chen/Compile_via_MLIR) | `3b7dd66313a709a08ba8bf09ba94cc197b15a994` | Host repository and bundled LLVM 17.0.0git snapshot |
| [HEIR](https://github.com/heir-compiler/HEIR) | `d2e673e4374ed07b3c8791cbba71c6e08dd5ff08` | NDSS compiler's RLWE/LWE representation and conversion semantics |
| [CHLOE](https://github.com/heir-compiler/CHLOE) | `a7a4b9595699cc4386abaa1355fa059c6386b124` | `HEIR.td`, `HEIRDialect.h`, `HEIRDialect.cpp` and license migrated into `compiler/Dialect/HEIR` |
| [Libra](https://github.com/sunnchioo/Libra) | `7edb01a668db8532157b06d95a546ffaf5d66aa7` | SCFHE ODS and declaration headers migrated into `compiler/Dialect/SCFHE` |
| [SEAL](https://github.com/microsoft/SEAL) | `119dc32e135cb89c1062076a69310d4413ebc824` | CKKS execution, version 4.1.2 |
| [OpenFHE](https://github.com/openfheorg/openfhe-development) | `aa391988d354d4360f390f223a90e0d1b98839d7` | CKKS/FHEW, conversion, sign bootstrap, version 1.4.2 |
| [OpenFHE cereal](https://github.com/openfheorg/cereal) | `984e3f194862b17916536b5fade40cba6e47a6fe` | Serialization dependency |
| [HE³DB](https://github.com/zhouzhangwalker/HE3DB) | `1d8d7827baf32b4094b7547aeeaa9faf59253b26` | Arithmetic-and-logic FHE design and TFHEpp→SEAL repacking reference; not an executable backend in this version |

The `heir` dialect used here is the NDSS HEIR/CHLOE project linked by the user, not Google's separately maintained HEIR project. HEIR and CHLOE define the same dialect name, so CHLOE's expanded definitions register once rather than registering incompatible copies.

`compiler/Dialect/HEIR` retains the original CHLOE definitions, registration source and license. CHLOE's TableGen-generated types need an explicit empty `NoiseType()` constructor parameter on the host LLVM snapshot. The compiler marks the required operation noise field as unanalysed.

Libra's operation names, attributes and type representation are retained in `compiler/Dialect/SCFHE/IR`. One include in `SCFHETypes.td` changes from LLVM 22's `mlir/IR/Traits.td` to the snapshot's `mlir/IR/OpBase.td`; `SCFHEDialect.cpp` implements the older inference interface. Canonicalization entry points are empty. The original FlyHE/CUDA translator, pattern selection and scheduling passes are not enabled.

The local `ppml` attributes describe scalar layout, source ONNX nodes, switching encodings and runtime resource contracts. These attributes are local additions, not capabilities claimed for the upstream papers.

Windows portability uses `SEAL_USE_ALIGNED_ALLOC=OFF`, links `bcrypt`, and includes `cstdlib` when building OpenFHE with llvm-mingw. Ciphertext and key serialization follows OpenFHE's scheme-switching serialization example, including refresh-key maps and separately saved FHEW→CKKS evaluation ciphertexts. The local runtime contains no OpenFHE private key in the evaluator.

HEIR/CHLOE/Libra/SEAL are MIT-licensed in these snapshots; OpenFHE uses BSD-2-Clause. Vendored licenses are kept with their files, and fetched library sources retain their licenses. Original implementation code in this module has no new standalone licensing claim; apply the host repository's licensing policy before publishing.
