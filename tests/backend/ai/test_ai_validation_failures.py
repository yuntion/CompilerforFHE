"""Infrastructure failures must remain distinct from a rejected proposal."""
import subprocess

import pytest

from backend.ai import validation


def test_runner_timeout_preserves_diagnostics_and_infrastructure_type(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["runner"], 1, output=b"partial output", stderr=b"diagnostic")

    monkeypatch.setattr(validation, "command", timeout)
    log = tmp_path/"timeout.log"
    with pytest.raises(validation.ValidationInfrastructureError, match="timed out"):
        validation._invoke(["runner"], {}, log)
    assert "partial output" in log.read_text() and "diagnostic" in log.read_text()
