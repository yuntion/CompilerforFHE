"""Build and run CrossFHE tests with per-run logs and JUnit reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_support import (
    check_build_source, check_python, config_path, executable, load_toolchain,
    openfhe_build, require_file, validate_toolchain,
)

SUITES = ("quick", "front", "compiler", "cpu", "gpu", "all", "scripts")
SUITE_MARKERS = {
    "quick": "not fhe", "front": "not fhe", "compiler": "not fhe",
    "cpu": "not gpu", "gpu": "gpu",
}


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def preflight(args, env):
    packages = {"pytest"} if args.suite == "scripts" else {"pytest", "numpy", "onnx"}
    if args.suite in ("quick", "compiler", "cpu", "gpu", "all"):
        packages.add("torch")
    check_python(packages)
    if args.suite in ("front", "scripts"):
        return {}, None, None
    config = load_toolchain(args.config)
    requirements = {"native"}
    if args.suite in ("cpu", "all"):
        requirements.update(("seal", "openfhe"))
    if args.suite in ("gpu", "all"):
        requirements.add("gpu")
    validate_toolchain(config, requirements, built=args.skip_build)
    cmake = executable(config.get("cmake"), "cmake")
    directories = [str(Path(executable(config.get(name), name)).parent)
                   for name in ("cmake", "cxx", "ninja")]
    env["PATH"] = os.pathsep.join(dict.fromkeys(directories)) + os.pathsep + env.get("PATH", "")
    check_build_source(config_path(config, "native_build"), ROOT)
    suffix = ".exe" if os.name == "nt" else ""
    ctest = executable(str(Path(cmake).with_name("ctest" + suffix)), "ctest")
    return config, cmake, ctest


def build_steps(config, suite, jobs):
    """Validate all source caches before starting any build."""
    cmake = executable(config.get("cmake"), "cmake")
    native = config_path(config, "native_build")
    cache = check_build_source(native, ROOT)
    sources = config.get("sources", {})
    for name, key in (("llvm", "CROSSFHE_LLVM_SOURCE"), ("seal", "CROSSFHE_SEAL_SOURCE")):
        if sources.get(name) and cache.get(key):
            if Path(sources[name]).resolve() != Path(cache[key]).resolve():
                raise RuntimeError(f"build source mismatch: cached {key} differs from sources.{name}")
    targets = ["crossfhe-opt", "crossfhe-translate"]
    if suite in ("cpu", "all"):
        targets.append("seal")
    steps = [("build", [cmake, "--build", native, "--target", *targets, "--parallel", jobs])]
    if suite in ("cpu", "all"):
        build = openfhe_build(config)
        steps += [
            ("build-openfhe", [cmake, "--build", build, "--target", "OPENFHEpke_static",
                               "OPENFHEbinfhe_static", "OPENFHEcore_static", "--parallel", jobs]),
            ("install-openfhe", [cmake, "--install", build]),
        ]
    if suite in ("gpu", "all"):
        build = config_path(config, "phantom_build")
        check_build_source(build, config_path(config, "phantom_source"))
        steps.append(("build-phantom", [cmake, "--build", build, "--target", "Phantom", "--parallel", jobs]))
    return steps


def run_step(name, command, env, directory, summary):
    command = [str(part) for part in command]
    print(f"\n[{name}] {shlex.join(command)}", flush=True)
    record = {"name": name, "command": command, "cwd": str(ROOT), "started_at": timestamp()}
    summary["steps"].append(record)
    started = time.perf_counter()
    try:
        with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace") as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    record["returncode"] = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    record["returncode"] = 130
                    raise
    finally:
        record["seconds"] = round(time.perf_counter() - started, 3)
        record["finished_at"] = timestamp()
        (directory / f"{name}-command.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if record["returncode"]:
        raise RuntimeError(f"{name} failed (exit {record['returncode']}); see {directory / (name + '.log')}")


def check_report(path):
    require_file(path)
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            counts[key] += case.find(tag) is not None
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Examples: python scripts/run_tests.py --suite cpu; "
        "python scripts/run_tests.py -- -k matvec. "
        "Arguments after -- are passed to pytest; suite markers and report paths are managed here."
    ))
    parser.add_argument("--suite", choices=SUITES, default="quick")
    parser.add_argument("--config", type=Path, default=Path(
        os.environ.get("CROSSFHE_CONFIG", ROOT / ".crossfhe/toolchain.json")))
    parser.add_argument("--jobs", type=int, default=4, help="parallel build jobs (default: 4)")
    parser.add_argument("--skip-build", action="store_true", help="test existing native binaries")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".crossfhe/test-runs",
                        help="parent directory for new, unique run directories")
    argv = list(sys.argv[1:] if argv is None else argv)
    separator = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:separator])
    pytest_args = argv[separator + 1:]
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    managed_options = {"--basetemp", "--junitxml", "--junit-xml", "--run-fhe", "--run-gpu",
                       "--collect-only", "--co", "-m"}
    for arg in pytest_args:
        if (arg.split("=", 1)[0] in managed_options
                or arg.startswith("-m") and not arg.startswith("--")):
            parser.error(f"{arg} is managed by run_tests.py; use --suite to select tests")
    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{timestamp()[:10]}-{args.suite}-", dir=args.output_dir))
    summary = {"suite": args.suite, "python": sys.executable, "config": str(args.config),
               "started_at": timestamp(), "skip_build": args.skip_build,
               "steps": [], "reports": {}, "passed": False}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "front") + os.pathsep + env.get("PYTHONPATH", "")
    env["CROSSFHE_CONFIG"] = str(args.config)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"Suite: {args.suite}\nPython: {sys.executable}\nReports: {directory}", flush=True)
    exit_code = 1
    try:
        config, cmake, ctest = preflight(args, env)
        if args.suite not in ("front", "scripts"):
            native = config_path(config, "native_build")
            if not args.skip_build:
                for name, command in build_steps(config, args.suite, args.jobs):
                    run_step(name, command, env, directory, summary)
            suffix = ".exe" if os.name == "nt" else ""
            for tool in ("crossfhe-opt", "crossfhe-translate"):
                executable(str(native / "bin" / (tool + suffix)), tool)
            if args.suite in ("gpu", "all"):
                require_file(config_path(config, "phantom_library"))
            run_step("ctest", [ctest, "--test-dir", native, "--output-on-failure",
                     "--no-tests=error", "--output-junit", directory / "ctest-junit.xml"],
                     env, directory, summary)
        paths = {"front": ["tests/front"], "compiler": ["tests/compiler"],
                 "scripts": ["tests/scripts"]}.get(args.suite, [
                     "tests/front", "tests/compiler", "tests/backend", "tests/e2e", "tests/scripts"])
        command = [sys.executable, "-m", "pytest", *paths, "-q", "-ra"]
        if args.suite in SUITE_MARKERS:
            command += ["-m", SUITE_MARKERS[args.suite]]
        if args.suite in ("cpu", "gpu", "all"):
            command.append("--run-fhe")
        if args.suite in ("gpu", "all"):
            command.append("--run-gpu")
        command += ["--basetemp", str(directory / "cases"),
                    "--junitxml", str(directory / "pytest-junit.xml"), *pytest_args]
        run_step("pytest", command, env, directory, summary)
        for name in ("ctest", "pytest"):
            report = directory / f"{name}-junit.xml"
            if report.is_file():
                counts = check_report(report)
                summary["reports"][name] = counts
                if not counts["tests"] or any(counts[key] for key in ("failures", "errors", "skipped")):
                    raise RuntimeError(f"{name} did not complete without failures or skips: {counts}")
            elif name == "pytest" or args.suite not in ("front", "scripts"):
                raise RuntimeError(f"test report missing: {report}")
        summary["passed"] = True
        exit_code = 0
    except (RuntimeError, OSError, ValueError, ET.ParseError) as error:
        summary["error"] = str(error)
        print(f"\nTests failed: {error}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        summary["error"] = "interrupted"
        exit_code = 130
        print("\nTests interrupted.", file=sys.stderr, flush=True)
    finally:
        # Also preserve counts when the test process itself failed.
        for name in ("ctest", "pytest"):
            report = directory / f"{name}-junit.xml"
            if report.is_file() and name not in summary["reports"]:
                try:
                    summary["reports"][name] = check_report(report)
                except (OSError, ET.ParseError):
                    pass
        summary["finished_at"] = timestamp()
        summary["exit_code"] = exit_code
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\n{'PASS' if summary['passed'] else 'FAIL'} — reports: {directory}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
