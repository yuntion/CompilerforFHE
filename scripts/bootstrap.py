"""Build pinned native tools and register their paths for the Python driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PINS={
    "compile_via_mlir":"3b7dd66313a709a08ba8bf09ba94cc197b15a994",
    "heir":"d2e673e4374ed07b3c8791cbba71c6e08dd5ff08",
    "chloe":"a7a4b9595699cc4386abaa1355fa059c6386b124",
    "libra":"7edb01a668db8532157b06d95a546ffaf5d66aa7",
    "seal":"119dc32e135cb89c1062076a69310d4413ebc824",
    "openfhe":"aa391988d354d4360f390f223a90e0d1b98839d7",
    "cereal":"984e3f194862b17916536b5fade40cba6e47a6fe",
}


def run(args,env=None):
    print("+ "+subprocess.list2cmdline([str(a) for a in args]),flush=True)
    subprocess.run([str(a) for a in args],env=env,check=True)


def checkout(url,commit,directory):
    git=["git"]+(["-c","http.sslBackend=openssl"] if os.name=="nt" else [])
    directory=Path(directory)
    if not (directory/".git").exists():
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"nonempty dependency directory without Git metadata: {directory}")
        directory.parent.mkdir(parents=True,exist_ok=True)
        run(git+["clone","--no-checkout",url,directory])
        run(git+["-C",directory,"checkout",commit])
    actual=subprocess.check_output(git+["-C",str(directory),"rev-parse","HEAD"],text=True).strip()
    if actual!=commit: raise RuntimeError(f"dependency revision mismatch: {directory}: {actual}")


def executable(name):
    if name=="cmake":
        try:
            import cmake
        except ImportError:
            pass
        else:
            bundled=Path(cmake.CMAKE_BIN_DIR)/(name+ (".exe" if os.name=="nt" else ""))
            if bundled.is_file(): return bundled
    if name=="ninja":
        try:
            import ninja
        except ImportError:
            pass
        else:
            bundled=Path(ninja.BIN_DIR)/(name+ (".exe" if os.name=="nt" else ""))
            if bundled.is_file(): return bundled
    found=shutil.which(name)
    if found: return Path(found)
    raise RuntimeError(f"{name} missing")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,default=ROOT/".crossfhe")
    parser.add_argument("--llvm-source",type=Path,default=ROOT/"third_party/llvm-project")
    parser.add_argument("--seal-source",type=Path)
    parser.add_argument("--openfhe-source",type=Path)
    parser.add_argument("--native-build",type=Path)
    parser.add_argument("--openfhe-prefix",type=Path)
    parser.add_argument("--openfhe-build",type=Path)
    parser.add_argument("--toolchain",type=Path,help="compiler bin directory")
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--skip-build",action="store_true",help="register already built artifacts")
    args=parser.parse_args()
    work=args.work_dir.resolve(); work.mkdir(parents=True,exist_ok=True)
    llvm=args.llvm_source.resolve()
    native=(args.native_build or ROOT/"build").resolve()
    prefix=(args.openfhe_prefix or work/"openfhe-install").resolve()
    if not (llvm/"mlir/include/mlir/IR/OpBase.td").is_file():
        raise RuntimeError("use the LLVM snapshot bundled in Compile_via_MLIR via --llvm-source")
    toolchain=args.toolchain
    if toolchain is None and os.name=="nt":
        name="llvm-mingw-20260922-ucrt-x86_64"
        toolchain=work/"toolchains"/name/"bin"
        if not (toolchain/"clang++.exe").is_file():
            archive=work/(name+".zip")
            url=f"https://github.com/mstorsjo/llvm-mingw/releases/download/20260922/{name}.zip"
            if not archive.exists(): urllib.request.urlretrieve(url,archive)
            if archive.stat().st_size!=190725905: raise RuntimeError("unexpected toolchain archive size")
            hasher=hashlib.sha256()
            with archive.open("rb") as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b""): hasher.update(chunk)
            digest=hasher.hexdigest()
            if digest!="e3ad77d117a4bea19a7a3b333341824d79a5a371004a10e25b8504e7b3047666":
                raise RuntimeError("toolchain archive SHA-256 mismatch")
            with zipfile.ZipFile(archive) as source:
                destination=(work/"toolchains").resolve()
                for member in source.infolist():
                    if not (destination/member.filename).resolve().is_relative_to(destination):
                        raise RuntimeError("invalid toolchain archive path")
                source.extractall(destination)
    suffix=".exe" if os.name=="nt" else ""
    cxx=(toolchain/("clang++"+suffix)).resolve() if toolchain else executable("clang++" if shutil.which("clang++") else "g++")
    cc=(toolchain/("clang"+suffix)).resolve() if toolchain else executable("clang" if shutil.which("clang") else "gcc")
    cmake,ninja=executable("cmake"),executable("ninja")
    env=os.environ.copy()
    env["PATH"]=os.pathsep.join(str(p) for p in (cxx.parent,cmake.parent,ninja.parent))+os.pathsep+env.get("PATH","")
    if not args.skip_build:
        seal=(args.seal_source or ROOT/"third_party/SEAL").resolve()
        ofhe=(args.openfhe_source or ROOT/"third_party/OpenFHE").resolve()
        checkout("https://github.com/microsoft/SEAL.git",PINS["seal"],seal)
        checkout("https://github.com/openfheorg/openfhe-development.git",PINS["openfhe"],ofhe)
        checkout("https://github.com/openfheorg/cereal.git",PINS["cereal"],ofhe/"third-party/cereal")
        common=["-G","Ninja","-DCMAKE_BUILD_TYPE=Release",f"-DCMAKE_MAKE_PROGRAM={ninja}",
                f"-DCMAKE_C_COMPILER={cc}",f"-DCMAKE_CXX_COMPILER={cxx}"]
        run([cmake,"-S",ROOT,"-B",native]+common+[
            f"-DPython3_EXECUTABLE={sys.executable}",f"-DCROSSFHE_LLVM_SOURCE={llvm}",f"-DCROSSFHE_SEAL_SOURCE={seal}",
            "-DLLVM_BUILD_TOOLS=OFF","-DLLVM_INCLUDE_UTILS=OFF","-DLLVM_BUILD_UTILS=OFF",
            "-DMLIR_INCLUDE_TESTS=OFF","-DMLIR_INCLUDE_INTEGRATION_TESTS=OFF","-DLLVM_PARALLEL_LINK_JOBS=1"],env)
        run([cmake,"--build",native,"--target","crossfhe-opt","crossfhe-translate","seal","--parallel",str(args.jobs)],env)
        openfhe_build=(args.openfhe_build or work/"openfhe-build").resolve()
        extra=["-DBUILD_STATIC=ON","-DBUILD_SHARED=OFF","-DBUILD_UNITTESTS=OFF","-DBUILD_EXAMPLES=OFF",
               "-DBUILD_BENCHMARKS=OFF","-DWITH_OPENMP=OFF","-DGIT_SUBMOD_AUTO=OFF",f"-DCMAKE_INSTALL_PREFIX={prefix}"]
        if os.name=="nt":
            extra += ["-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -Wno-error -include cstdlib",
                      "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -Wno-error"]
        run([cmake,"-S",ofhe,"-B",openfhe_build]+common+extra,env)
        run([cmake,"--build",openfhe_build,"--target","OPENFHEpke_static","OPENFHEbinfhe_static","OPENFHEcore_static","--parallel",str(args.jobs)],env)
        run([cmake,"--install",openfhe_build],env)
    openfhe_config=prefix/("CMake" if os.name=="nt" else "lib/OpenFHE")
    seal_config=native/"seal/cmake"
    for file in (native/"bin"/("crossfhe-opt"+suffix),native/"bin"/("crossfhe-translate"+suffix),
                 seal_config/"SEALConfig.cmake",openfhe_config/"OpenFHEConfig.cmake",cxx,cmake,ninja):
        if not file.is_file(): raise RuntimeError(f"required artifact missing: {file}")
    config={"native_build":str(native),"seal_dir":str(seal_config),"openfhe_dir":str(openfhe_config),
            "cxx":str(cxx),"cc":str(cc),"cmake":str(cmake),"ninja":str(ninja),"revisions":PINS}
    destination=ROOT/".crossfhe/toolchain.json"; destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    print(f"Ready: {destination}")


if __name__=="__main__":
    try: main()
    except (RuntimeError,OSError,subprocess.CalledProcessError) as error:
        print(f"bootstrap failed: {error}",file=sys.stderr); raise SystemExit(1)
