# Compiler-centered layout migration

The FHE pipeline now builds from the repository root by default. `front/` contains only Python model import/export and orchestration; the compiler owns dialects, lowering and code generation; `backend/` owns cryptographic execution.

| Previous location | Current location |
| --- | --- |
| `ppml_fhe/python/crossfhe` | `front/crossfhe` |
| `ppml_fhe/native/LowerTo*.cpp` | `compiler/Conversion/FHE` |
| `ppml_fhe/native/Passes.h`, `OptimizationHooks.cpp` | `compiler/Pipelines/FHE` |
| `ppml_fhe/native/translate.cpp` | `compiler/CodeGen/FHE/CodeGen.cpp` plus `tools/crossfhe-translate.cpp` |
| `ppml_fhe/native/opt.cpp` | `tools/crossfhe-opt.cpp` |
| `ppml_fhe/third_party/chloe` | `compiler/Dialect/HEIR` |
| `ppml_fhe/third_party/libra`, `native/LibraDialect.cpp` | `compiler/Dialect/SCFHE` |
| `ppml_fhe/runtime` | `backend/seal`, `backend/openfhe` |
| `ppml_fhe/tests` | `tests/front`, `compiler`, `backend`, `e2e` and root `tests/conftest.py` |
| `ppml_fhe/scripts`, `docs`, `examples` | Root `scripts`, `docs`, `examples` |
| `ppml_fhe/.crossfhe`, `artifacts` | Root `.crossfhe`, `artifacts` |
| Original root CMake configuration | Optional `build-tools/cmake/LegacyCompiler.cmake` |

The old package distribution name is now `crossfhe-front`; Python imports remain `crossfhe`. Reinstall from the repository root with `python -m pip install --no-deps -e .`. Remove an old `crossfhe-ppml` editable installation before reinstalling if it still points to the previous directory.

Run the root `scripts/bootstrap.py` to regenerate `.crossfhe/toolchain.json` for the new root build. The previous native build remains outside the repository, and historical artifact binaries are preserved. Recompile historical models into a fresh artifact directory: their CMake cache refers to the old runtime source directory. Old reports/logs retain their original absolute paths and are historical data.

The residual empty old directories and pytest caches were preserved under ignored `.crossfhe/archive/ppml_fhe-layout-cache`. No user models, keys or ciphertexts were removed. Source packaging excludes all archives, runtime artifacts, dependencies and build outputs.

The migrated ODS definitions keep upstream operation/type names and licenses. No optimization pass or GPU implementation was added in this restructuring.
