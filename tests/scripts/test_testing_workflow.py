"""Regression tests for test selection, setup failures and the runner contract."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_tests as runner
from scripts import test_support as support


def cache(path, source, prefix=None):
    path.mkdir(parents=True)
    text = f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n"
    if prefix:
        text += f"CMAKE_INSTALL_PREFIX:PATH={prefix}\n"
    (path / "CMakeCache.txt").write_text(text)


@pytest.mark.parametrize("content,diagnostic", [
    ("{bad", "cannot read toolchain config"),
    ("[]", "must be a JSON object"),
    ('{"native_build": 1}', "missing toolchain setting: native_build"),
])
def test_invalid_config_has_actionable_diagnostic(tmp_path, content, diagnostic):
    path = tmp_path / "toolchain.json"
    path.write_text(content)
    with pytest.raises(RuntimeError, match=diagnostic):
        support.load_toolchain(path)


def test_executable_diagnostic_identifies_invalid_tool(tmp_path):
    with pytest.raises(RuntimeError, match="cmake is not executable"):
        support.executable(str(tmp_path / "missing-cmake"), "cmake")


def test_build_source_mismatch_is_rejected(tmp_path):
    build = tmp_path / "build"
    cache(build, tmp_path / "old-source")
    with pytest.raises(RuntimeError, match="build source mismatch"):
        support.check_build_source(build, tmp_path / "current-source")


def test_old_openfhe_config_requires_matching_source_and_prefix(tmp_path):
    prefix, source = tmp_path / "install", tmp_path / "source"
    matching, other = tmp_path / "custom-build", tmp_path / "other-build"
    cache(matching, source, prefix)
    cache(other, source, tmp_path / "other-install")
    config = {"openfhe_dir": str(prefix / "lib/OpenFHE"), "sources": {"openfhe": str(source)}}
    assert support.openfhe_build(config) == matching
    cache(tmp_path / "ambiguous-build", source, prefix)
    with pytest.raises(RuntimeError, match="unique OpenFHE build"):
        support.openfhe_build(config)


def test_all_builds_update_installed_openfhe_before_tests(tmp_path):
    native, openfhe, phantom = (tmp_path / name for name in ("native", "openfhe", "phantom"))
    prefix, source = tmp_path / "install", tmp_path / "openfhe-source"
    cache(native, runner.ROOT)
    cache(openfhe, source, prefix)
    cache(phantom, tmp_path / "phantom-source")
    config = {"cmake": sys.executable, "native_build": str(native), "openfhe_build": str(openfhe),
              "openfhe_dir": str(prefix / "lib/OpenFHE"), "sources": {"openfhe": str(source)},
              "phantom_build": str(phantom), "phantom_source": str(tmp_path / "phantom-source")}
    steps = runner.build_steps(config, "all", 3)
    assert [name for name, _ in steps] == ["build", "build-openfhe", "install-openfhe", "build-phantom"]
    assert "seal" in steps[0][1]
    assert steps[2][1][-2:] == ["--install", openfhe]
    # An inconsistent GPU source must block the entire build plan.
    config["phantom_source"] = str(tmp_path / "wrong-source")
    with pytest.raises(RuntimeError, match="build source mismatch"):
        runner.build_steps(config, "all", 3)


@pytest.fixture
def isolated_pytest(tmp_path):
    (tmp_path / "conftest.py").write_text((runner.ROOT / "tests/conftest.py").read_text())
    (tmp_path / "pytest.ini").write_text(
        f"[pytest]\npythonpath = {runner.ROOT}\naddopts = --strict-markers\n"
        "markers =\n native\n torch\n fhe\n gpu\n seal\n openfhe\n"
    )
    (tmp_path / "test_selection.py").write_text(
        "import pytest\n"
        "def test_front(): pass\n"
        "@pytest.mark.native\ndef test_native(): pass\n"
        "@pytest.mark.fhe\n@pytest.mark.seal\ndef test_seal(): pass\n"
        "@pytest.mark.fhe\n@pytest.mark.openfhe\ndef test_openfhe(): pass\n"
        "@pytest.mark.fhe\n@pytest.mark.gpu\ndef test_gpu(): pass\n"
    )

    def run(*args):
        env = os.environ.copy()
        env["CROSSFHE_CONFIG"] = str(tmp_path / "missing.json")
        env.pop("PYTHONPATH", None)
        env.pop("PYTEST_ADDOPTS", None)
        return subprocess.run([sys.executable, "-m", "pytest", "-q", *args],
                              cwd=tmp_path, env=env, capture_output=True, text=True)
    return run


@pytest.mark.parametrize("args,selected,excluded", [
    (["--collect-only"], ["test_front", "test_native"], ["test_seal", "test_openfhe", "test_gpu"]),
    (["--collect-only", "--run-fhe"], ["test_seal", "test_openfhe"], ["test_gpu"]),
    (["--collect-only", "--run-fhe", "--run-gpu", "-m", "gpu"], ["test_gpu"], ["test_native", "test_seal"]),
    (["--collect-only", "--run-fhe", "-m", "seal"], ["test_seal"], ["test_openfhe", "test_gpu"]),
])
def test_collection_obeys_encryption_and_backend_selection(isolated_pytest, args, selected, excluded):
    result = isolated_pytest(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in selected:
        assert "::" + name in result.stdout
    for name in excluded:
        assert "::" + name not in result.stdout


def test_front_selection_does_not_require_toolchain(isolated_pytest):
    result = isolated_pytest("-k", "test_front")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.parametrize("args", [
    ["-k", "test_native"], ["--run-fhe", "-m", "seal"],
    ["--run-fhe", "--run-gpu", "-m", "gpu"],
])
def test_selected_missing_dependency_is_one_setup_error(isolated_pytest, args):
    result = isolated_pytest(*args)
    assert result.returncode == 4
    assert "test setup: required file missing:" in result.stderr
    assert "skipped" not in result.stdout


@pytest.fixture
def mini_repo(tmp_path, monkeypatch):
    front = tmp_path / "tests/front"
    front.mkdir(parents=True)
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n fhe\n")
    (front / "test_probe.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "def test_fail(): assert False, 'probe failure'\n"
        "def test_skip(): pytest.skip('probe skip')\n"
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "check_python", lambda packages: support.check_python({"pytest"}))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    return tmp_path


@pytest.mark.parametrize("selection,expected,count_key", [
    ("test_pass", 0, "tests"), ("test_fail", 1, "failures"),
    ("test_skip", 1, "skipped"), ("no_such_probe", 1, None),
])
def test_runner_reports_real_pytest_outcomes(mini_repo, selection, expected, count_key):
    output = mini_repo / "reports"
    result = runner.main(["--suite", "front", "--output-dir", str(output), "--", "-k", selection])
    assert result == expected
    summary_path, = output.glob("*/summary.json")
    summary = json.loads(summary_path.read_text())
    assert summary["passed"] == (expected == 0)
    assert summary["steps"][0]["command"][-2:] == ["-k", selection]
    if count_key:
        assert summary["reports"]["pytest"][count_key] == 1


def test_repeated_runs_keep_separate_reports(mini_repo):
    args = ["--suite", "front", "--output-dir", str(mini_repo / "reports"), "--", "-k", "test_pass"]
    assert runner.main(args) == runner.main(args) == 0
    reports = list((mini_repo / "reports").glob("*/summary.json"))
    assert len(reports) == 2
    assert all(path.parent.joinpath("pytest-junit.xml").is_file() for path in reports)


def test_failed_build_stops_before_testing(mini_repo, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda args, env: ({"native_build": str(mini_repo)}, None, None))
    monkeypatch.setattr(runner, "build_steps", lambda *args: [
        ("build", [sys.executable, "-c", "raise SystemExit(7)"])
    ])
    output = mini_repo / "reports"
    assert runner.main(["--output-dir", str(output)]) == 1
    path, = output.glob("*/summary.json")
    summary = json.loads(path.read_text())
    assert len(summary["steps"]) == 1
    assert summary["steps"][0]["returncode"] == 7
    assert not summary["reports"]


@pytest.mark.parametrize("arg", ["--run-fhe", "--run-gpu", "--basetemp", "--junitxml", "--collect-only", "-mfhe"])
def test_managed_options_cannot_override_suite_or_reports(arg):
    with pytest.raises(SystemExit) as error:
        runner.main(["--", arg])
    assert error.value.code == 2
