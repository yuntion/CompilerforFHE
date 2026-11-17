"""Create a source-only handoff; no ciphertexts, private keys or toolchains."""
import argparse
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES={"compiler","front","backend","tools","scripts","tests","docs","examples","build-tools"}
ROOT_FILES={"README.md","CMakeLists.txt","pyproject.toml","requirements.txt",".gitignore",
            ".gitmodules","third_party/README.md"}
EXCLUDED={"__pycache__",".pytest_cache",".venv",".crossfhe","build","artifacts",".git"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    candidates=[ROOT/name for name in sorted(ROOT_FILES)]
    for directory in sorted(DIRECTORIES):
        candidates.extend((ROOT/directory).rglob("*"))
    files=[]
    for path in sorted(candidates):
        relative=path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts): continue
        if path.suffix in (".pyc",".exe",".dll",".ct",".bin",".seal",".bc",".a",".lib",".obj"): continue
        files.append((path,relative))
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path,relative in files: archive.write(path,str(Path("Compile_via_MLIR")/relative))
    print(f"Packaged {len(files)} source files: {args.output.resolve()}")


if __name__=="__main__": main()
