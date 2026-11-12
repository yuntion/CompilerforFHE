# Validation record

Validation ran on Windows x86-64 using Python 3.12.14, PyTorch 2.8.0, ONNX 1.19.0, NumPy 2.2.6, CMake 3.31.6, Ninja 1.11.1.4 and llvm-mingw Clang 23.1.2. The compiler was built from the host repository's bundled LLVM 17.0.0git source snapshot. SEAL and OpenFHE were compiled from the revisions in UPSTREAM.md.

The compiler-centered revision passed all 34 Python tests in 243.99 seconds on 2026-09-29, with no failed or skipped tests, after a fresh native build from the repository root. It includes real encrypted runs for affine, square MLP, small CNN, polynomial attention, cross-scheme ReLU/ReLU-MLP/MaxPool, grouped/dilated convolution, batch normalization, batched secret MatMul, scalar/public outputs and zero multiplication. Frontend and native tests also check unsupported operations, encrypted division, invalid shapes/dtypes, custom domains, operation limits, malformed MLIR, the empty optimization hook, scalar reductions, incomplete-artifact rejection and unsafe PyTorch data-dependent tracing. Two additional tests directly lower MLIR and reject incompatible SEAL/GPU code-generation choices without creating an output header. The machine-readable result is recorded in `validation-junit.xml`. The 29 reported warnings are PyTorch ONNX-export deprecations and one expected warning about a deliberately constant tensor output.

The suite was run from the repository root with `python -m pytest --run-fhe -q --basetemp .crossfhe/refactor-test-all --junitxml docs/validation-junit.xml`, collecting `tests/front`, `tests/compiler`, `tests/backend` and `tests/e2e`. Separately, all 3 native CTest checks passed in 0.13 seconds: dialect registration, direct MLIR lowering and standalone C++ generation. Their result is recorded in `native-validation-junit.xml`. These checks do not require PyTorch.

`docs/measurements/results.json` and `results.csv` contain 11 completed encrypted runs from the new suite plus the separately measured default-profile ReLU below; earlier development measurements are excluded. The migration preserves the runtime implementation: both backend headers and both runner sources are byte-identical to the previous verified source package, and the migrated CHLOE ODS SHA-256 equals the pinned upstream file.

The OpenFHE regression runs explicitly use `INSECURE_TOY` for speed. Separately, the ReLU demo was run with default `HEStd_128_classic+STD128` parameters before the directory migration. This is a historical measurement of the unchanged cryptographic runtime, not a new secure-profile performance experiment for the refactored build:

| Measurement | Observed value |
| --- | --- |
| CKKS ring dimension | 65536 |
| Source arithmetic depth / reserved total depth | 2 / 19 |
| Input/output scalars | 2 / 2 |
| CKKS→LWE / sign / LWE→CKKS calls | 2 / 2 / 2 |
| Comparison scale | 1024 |
| Evaluator seconds | 43.8670076 |
| CKKS→LWE seconds | 0.6965343 |
| Logic sign seconds | 5.7792222 |
| LWE→CKKS seconds | 36.6241928 |
| Evaluation process wall seconds, including key load | 98.7493365 |
| Key-generation process wall seconds | 332.7165867 |
| Serialized evaluation bundle bytes | 11882709356 |
| Maximum absolute error vs PyTorch | 9.787168870220057e-6 |

These are single-run functional measurements of an unoptimized scalar baseline, taken while other tests were also running. They are not a controlled performance benchmark, a statistical result or a claim that every supported model has been tested with default security parameters. The bundle retains duplicate bootstrap-key artifacts to mirror the reference serialization; optimizing its representation is future work.

The default-profile client input was `[-0.5, 0.75]`, and the model was `relu(1.5*x + 0.125)`, yielding expected `[0, 1.25]`. The decrypted result was approximately `[-9.78717e-6, 1.24999902]`. The evaluator loaded only server artifacts; the client decrypted the final ciphertexts.

The initial SEAL demos used `tc128`: affine max error `1.788e-7`, tiny CNN `4.115e-7`, polynomial attention `3.632e-7`. The polynomial attention example deliberately uses square-based weights, with no Softmax. Fresh encryption changes numerical errors slightly on each run.

Raw per-run reports remain in local artifacts. `scripts/collect_reports.py` exports only metrics and model hashes to JSON/CSV; keys and plaintexts are excluded. The test XML may contain local paths, so trim those before publishing it outside the project.

The root bootstrap command also built the compiler/SEAL targets and registered OpenFHE successfully. OpenFHE's upstream CMake can print a nonfatal `demoData` directory-copy warning on Windows; the compiled libraries and model runner were usable in the encrypted tests above.

For paper experiments, freeze data, library revisions, machine configuration, scheme parameters, range/precision contracts and compiler flags; isolate runs; report medians and tails; and include application accuracy and ablations for each newly added pass. Compare secure configurations at matched output precision and semantics.
