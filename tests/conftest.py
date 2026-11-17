import pytest

from scripts.test_support import check_python, load_toolchain, validate_toolchain


def pytest_addoption(parser):
    parser.addoption("--run-fhe", action="store_true", help="run real encrypted CPU integration tests")
    parser.addoption("--run-gpu", action="store_true", help="also run Phantom GPU tests (requires --run-fhe)")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu") and not config.getoption("--run-fhe"):
        raise pytest.UsageError("--run-gpu requires --run-fhe")
    selected, deselected = [], []
    for item in items:
        if (("fhe" in item.keywords and not config.getoption("--run-fhe"))
                or ("gpu" in item.keywords and not config.getoption("--run-gpu"))):
            deselected.append(item)
        else:
            selected.append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    # Check after pytest's -m/-k filtering so frontend-only selections need no toolchain.
    if config.option.collectonly:
        return
    requirements = {name for item in items for name in ("native", "torch", "seal", "openfhe", "gpu")
                    if name in item.keywords}
    try:
        check_python({"torch"} if "torch" in requirements else set())
        if requirements.intersection({"native", "seal", "openfhe", "gpu"}):
            toolchain = load_toolchain()
            validate_toolchain(toolchain, requirements)
            config._crossfhe_toolchain = toolchain
    except RuntimeError as error:
        raise pytest.UsageError(f"test setup: {error}") from error


@pytest.fixture(scope="session")
def config(pytestconfig):
    return pytestconfig._crossfhe_toolchain
