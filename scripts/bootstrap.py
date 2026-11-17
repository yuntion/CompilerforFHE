"""Initialize Git submodules, build CrossFHE dependencies, and write toolchain.json."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"llvm": "third_party/llvm-project", "seal": "third_party/SEAL",
           "openfhe": "third_party/OpenFHE", "phantom": "third_party/phantom-fhe"}


def run(*command):
    command = list(map(str, command))
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def tool(name):
    found = shutil.which(str(name))
    if not found:
        raise RuntimeError(f"required tool missing: {name}")
    return found


def initialize(args, sources):
    for path in [*sources.values(), sources["openfhe"] / "third-party/cereal"]:
        if (path / ".git").exists() and git(path, "status", "--porcelain"):
            raise RuntimeError(f"dependency has local changes: {path}")
    modules = [path for name, path in SOURCES.items() if getattr(args, name + "_source") is None]
    if modules:
        run("git", "submodule", "sync", "--", *modules)
        run("git", "submodule", "update", "--init", "--depth", "1", "--", *modules)
    if (sources["openfhe"] / ".git").exists():
        run("git", "-C", sources["openfhe"], "submodule", "update", "--init", "--depth", "1",
            "--", "third-party/cereal")
    for name, marker in {"llvm": "mlir/include/mlir/IR/OpBase.td", "seal": "CMakeLists.txt",
                         "openfhe": "third-party/cereal/include/cereal/cereal.hpp",
                         "phantom": "include/phantom.h"}.items():
        if not (sources[name] / marker).is_file():
            raise RuntimeError(f"dependency source missing: {sources[name] / marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument(f"--{name}-source", type=Path, help="use an existing source tree")
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".crossfhe")
    for name in ("native-build", "openfhe-build", "openfhe-prefix", "phantom-build", "toolchain"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--enable-cuda", action="store_true", help="also build the optional CUDA library")
    parser.add_argument("--cuda-compiler", type=Path, help="nvcc path; otherwise use PATH")
    parser.add_argument("--cuda-architectures", default="native", help="CMake CUDA architectures, e.g. 89")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--init-only", action="store_true", help="initialize sources without building")
    modes.add_argument("--skip-build", action="store_true", help="register existing artifacts without downloading")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    sources = {name: (getattr(args, name + "_source") or ROOT / path).resolve()
               for name, path in SOURCES.items()}
    if not args.skip_build:
        initialize(args, sources)
    if args.init_only:
        return

    work = args.work_dir.resolve()
    native = (args.native_build or ROOT / "build").resolve()
    openfhe_build = (args.openfhe_build or work / "openfhe-build").resolve()
    prefix = (args.openfhe_prefix or work / "openfhe-install").resolve()
    phantom_build = (args.phantom_build or work / "phantom-build").resolve()
    suffix = ".exe" if os.name == "nt" else ""
    cc = tool(args.toolchain / ("clang" + suffix) if args.toolchain else
              os.environ.get("CC", "clang" if shutil.which("clang") else "gcc"))
    cxx = tool(args.toolchain / ("clang++" + suffix) if args.toolchain else
               os.environ.get("CXX", "clang++" if shutil.which("clang++") else "g++"))
    cmake, ninja = tool("cmake"), tool("ninja")
    common = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={ninja}",
              f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"]
    if not args.skip_build:
        run(cmake, "-S", ROOT, "-B", native, *common,
            f"-DPython3_EXECUTABLE={sys.executable}", f"-DCROSSFHE_LLVM_SOURCE={sources['llvm']}",
            f"-DCROSSFHE_SEAL_SOURCE={sources['seal']}", "-DLLVM_BUILD_TOOLS=OFF",
            "-DLLVM_INCLUDE_UTILS=OFF", "-DLLVM_BUILD_UTILS=OFF", "-DMLIR_INCLUDE_TESTS=OFF",
            "-DMLIR_INCLUDE_INTEGRATION_TESTS=OFF", "-DLLVM_PARALLEL_LINK_JOBS=1")
        run(cmake, "--build", native, "--target", "crossfhe-opt", "crossfhe-translate", "seal",
            "--parallel", args.jobs)
        windows = (["-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -Wno-error -include cstdlib",
                    "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -Wno-error"] if os.name == "nt" else [])
        run(cmake, "-S", sources["openfhe"], "-B", openfhe_build, *common, *windows,
            "-DBUILD_STATIC=ON", "-DBUILD_SHARED=OFF", "-DBUILD_UNITTESTS=OFF", "-DBUILD_EXAMPLES=OFF",
            "-DBUILD_BENCHMARKS=OFF", "-DWITH_OPENMP=OFF", "-DGIT_SUBMOD_AUTO=OFF",
            f"-DCMAKE_INSTALL_PREFIX={prefix}")
        run(cmake, "--build", openfhe_build, "--target", "OPENFHEpke_static", "OPENFHEbinfhe_static",
            "OPENFHEcore_static", "--parallel", args.jobs)
        run(cmake, "--install", openfhe_build)

    cuda = tool(args.cuda_compiler or "nvcc") if args.enable_cuda else None
    if args.enable_cuda and (sources["phantom"] / ".git").exists():
        revision = git(sources["phantom"], "rev-parse", "HEAD")
        if revision != "1f4a198443b3af77118e51f53d5b8f332154b875":
            raise RuntimeError(f"unsupported Phantom revision: {revision}; use the pinned submodule")
    if args.enable_cuda and not args.skip_build:
        run(cmake, "-S", sources["phantom"], "-B", phantom_build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_CXX_COMPILER={cxx}", f"-DCMAKE_CUDA_COMPILER={cuda}",
            f"-DCMAKE_CUDA_HOST_COMPILER={cxx}",
            f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}", "-DPHANTOM_ENABLE_EXAMPLE=OFF",
            "-DPHANTOM_ENABLE_BENCH=OFF", "-DPHANTOM_ENABLE_TEST=OFF", "-DPHANTOM_ENABLE_PYTHON_BINDING=OFF")
        run(cmake, "--build", phantom_build, "--target", "Phantom", "--parallel", args.jobs)

    config = {"native_build": str(native), "seal_dir": str(native / "seal/cmake"),
              "openfhe_dir": str(prefix / ("CMake" if os.name == "nt" else "lib/OpenFHE")),
              "openfhe_build": str(openfhe_build),
              "cc": cc, "cxx": cxx, "cmake": cmake, "ninja": ninja,
              "sources": {name: str(path) for name, path in sources.items()}}
    tracked = {**sources, "cereal": sources["openfhe"] / "third-party/cereal"}
    config["revisions"] = {name: git(path, "rev-parse", "HEAD") if (path / ".git").exists() else "unknown"
                           for name, path in tracked.items()}
    required = [native / "bin" / (name + suffix) for name in ("crossfhe-opt", "crossfhe-translate")]
    required += [Path(config["seal_dir"]) / "SEALConfig.cmake",
                 Path(config["openfhe_dir"]) / "OpenFHEConfig.cmake"]
    if args.enable_cuda:
        libraries = [phantom_build / path for path in ("lib/libPhantom.so", "lib/libPhantom.dylib",
                     "bin/Phantom.dll", "lib/Phantom.dll", "Phantom.dll")]
        library = next((path for path in libraries if path.is_file()), libraries[0])
        required.append(library)
        config.update(phantom_source=str(sources["phantom"]), phantom_build=str(phantom_build),
                      phantom_library=str(library), cuda_compiler=str(cuda) if cuda else None,
                      cuda_architectures=args.cuda_architectures)
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required artifact missing: {path}")
    destination = ROOT / ".crossfhe/toolchain.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: {destination}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"bootstrap failed: {error}", file=sys.stderr)
        raise SystemExit(1)
