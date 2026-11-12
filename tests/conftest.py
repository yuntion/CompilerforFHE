import pytest


def pytest_addoption(parser):
    parser.addoption("--run-fhe",action="store_true",help="compile and run real encrypted integration tests")


def pytest_collection_modifyitems(config,items):
    if config.getoption("--run-fhe"): return
    for item in items:
        if "fhe" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="requires --run-fhe and scripts/bootstrap.py"))
