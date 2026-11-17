# Third-party dependencies

The four dependencies below are Git submodules. Initialize the exact commits recorded by the superproject with `python scripts/bootstrap.py --init-only`, or use `git submodule update --init --depth 1` followed by initialization of OpenFHE's `third-party/cereal` submodule. Run Python commands in the `CompilerforFHE` Conda environment.

| Directory | Upstream | Requested version |
| --- | --- | --- |
| `llvm-project` | https://github.com/llvm/llvm-project | `release/17.x` branch; exact commit stored in the gitlink |
| `SEAL` | https://github.com/microsoft/SEAL | `v4.1.2` tag, commit `119dc32e135cb89c1062076a69310d4413ebc824` |
| `OpenFHE` | https://github.com/openfheorg/openfhe-development | `aa391988d354d4360f390f223a90e0d1b98839d7` |
| `phantom-fhe` | https://github.com/encryptorion-lab/phantom-fhe | `1f4a198443b3af77118e51f53d5b8f332154b875` |

All four revisions are pinned by the superproject gitlinks. LLVM comes from `release/17.x`, and SEAL's pinned commit corresponds to the `v4.1.2` tag. The current `.gitmodules` also contains version or commit values in the other `branch` fields; these do not pin revisions and are not remote branch names. Use ordinary `git submodule update --init`, without `--remote`, or bootstrap to reproduce the pinned versions. Commit an intentional submodule update separately when upgrading dependencies.

`scripts/bootstrap.py` builds the compiler, SEAL and OpenFHE by default. OpenFHE's nested cereal dependency is pinned to `984e3f194862b17916536b5fade40cba6e47a6fe`; unused benchmark and test submodules are not fetched. Use `--with-phantom` to additionally build Phantom with CUDA and register the optional experimental `phantom-ckks` backend; see [backend usage](../docs/PHANTOM_BACKEND.md). CUDA is not required for the default CPU build.

Successful bootstrap runs, except `--init-only`, replace `.crossfhe/toolchain.json`. Keep `--with-phantom` and the corresponding source, build and CUDA options when re-registering a GPU environment; an ordinary CPU bootstrap removes the existing Phantom registration. Use the test runner or incremental CMake builds for routine updates.

The migrated HEIR/CHLOE and Libra dialect sources are in `compiler/Dialect/HEIR` and `compiler/Dialect/SCFHE`, with their licenses. They are compiler modules rather than external compiler executables. Provenance and modifications are recorded in `docs/UPSTREAM.md`.

The source handoff archive excludes submodule source trees and Git metadata. Prepare the dependencies separately and supply existing paths with `--llvm-source`, `--seal-source`, `--openfhe-source` and `--phantom-source`. Bootstrap does not download toolchains or clone repositories outside Git's submodule initialization. CMake, Ninja and C/C++ compilers must already be on PATH, or use `--toolchain` for a Clang bin directory. The resulting toolchain file records actual Git revisions where available. Verify custom source revisions against [UPSTREAM.md](../docs/UPSTREAM.md): bootstrap does not validate every custom dependency revision, although it explicitly checks the Phantom Git revision when enabled.
