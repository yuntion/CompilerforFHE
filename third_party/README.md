# Third-party dependencies

`third_party/llvm-project` is the LLVM/MLIR source snapshot already bundled in the host repository. The default compiler uses this snapshot. The existing `.gitmodules` file is retained from the original repository; the current snapshot is committed as a source tree, not fetched by the FHE frontend.

`scripts/bootstrap.py` obtains pinned Microsoft SEAL and OpenFHE source revisions in `third_party/SEAL` and `third_party/OpenFHE`, unless explicit existing source paths are supplied. These fetched working directories are ignored by Git. OpenFHE's cereal dependency is pinned separately.

The migrated HEIR/CHLOE and Libra dialect sources are in `compiler/Dialect/HEIR` and `compiler/Dialect/SCFHE`, with their licenses. They are compiler modules rather than external compiler executables. Provenance and modifications are recorded in `docs/UPSTREAM.md`.

The source handoff archive excludes the large LLVM source snapshot and fetched cryptographic libraries. Use the host repository's bundled LLVM or supply its snapshot with `--llvm-source`; bootstrap fetches the cryptographic libraries.
