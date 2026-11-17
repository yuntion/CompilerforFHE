"""Standard-library-only test setup shared by the runner and pytest."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ("native_build", "seal_dir", "openfhe_dir", "openfhe_build",
             "phantom_source", "phantom_build", "phantom_library", "cuda_compiler")


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")
    return path


def setting(value, name):
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"missing toolchain setting: {name}; run python scripts/bootstrap.py first")
    return value


def config_path(config, name):
    return (ROOT / Path(setting(config.get(name), name)).expanduser()).resolve()


def executable(value, label):
    found = shutil.which(setting(value, label))
    if not found:
        raise RuntimeError(f"{label} is not executable: {value}; run python scripts/bootstrap.py again")
    return str(Path(found).resolve())


def check_python(packages):
    missing = [name for name in sorted(packages) if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"Python {sys.executable} is missing: {', '.join(missing)}. "
            "Activate your test environment; run python -m pip install -r requirements.txt."
        )


def load_toolchain(path=None):
    path = Path(path or os.environ.get("CROSSFHE_CONFIG", ROOT / ".crossfhe/toolchain.json")).expanduser().resolve()
    require_file(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise RuntimeError(f"cannot read toolchain config {path}: {error}") from error
    if not isinstance(config, dict):
        raise RuntimeError(f"toolchain config must be a JSON object: {path}")
    for name in PATH_KEYS:
        if name in config:
            config[name] = str(config_path(config, name))
    return config


def validate_toolchain(config, requirements, *, built=True):
    """Check only dependencies of the selected tests, never silently skip."""
    requirements = set(requirements)
    if not requirements.intersection({"native", "seal", "openfhe", "gpu"}):
        return
    for name in ("cmake", "ninja", "cxx"):
        executable(config.get(name), name)
    native = config_path(config, "native_build")
    if built:
        suffix = ".exe" if os.name == "nt" else ""
        for name in ("crossfhe-opt", "crossfhe-translate"):
            executable(str(native / "bin" / (name + suffix)), name)
    for marker, key, filename in (("seal", "seal_dir", "SEALConfig.cmake"),
                                  ("openfhe", "openfhe_dir", "OpenFHEConfig.cmake")):
        if marker in requirements:
            require_file(config_path(config, key) / filename)
    if "gpu" in requirements:
        if not config.get("phantom_library"):
            raise RuntimeError("GPU tests require Phantom; run python scripts/bootstrap.py --enable-cuda first")
        require_file(config_path(config, "phantom_source") / "include/phantom.h")
        executable(config.get("cuda_compiler"), "cuda_compiler")
        if built:
            require_file(config_path(config, "phantom_library"))


def read_cache(build):
    cache = require_file(Path(build) / "CMakeCache.txt")
    values = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def check_build_source(build, source):
    cache = read_cache(build)
    actual = cache.get("CMAKE_HOME_DIRECTORY")
    if not actual or Path(actual).resolve() != Path(source).resolve():
        raise RuntimeError(
            f"build source mismatch: {build} uses {actual}, expected {source}. "
            "Use a new matching build directory and register it with scripts/bootstrap.py."
        )
    return cache


def openfhe_build(config):
    """Use the registered build, checking its source and install destination."""
    build = config_path(config, "openfhe_build")
    source = setting(config.get("sources", {}).get("openfhe"), "sources.openfhe")
    cache = check_build_source(build, source)
    installed = config_path(config, "openfhe_dir")
    prefix = installed.parent if installed.name == "CMake" else installed.parents[1]
    if Path(cache.get("CMAKE_INSTALL_PREFIX", "")).resolve() != prefix:
        raise RuntimeError(
            f"OpenFHE install prefix mismatch: {build} does not install to {prefix}; "
            "run python scripts/bootstrap.py again"
        )
    return build
