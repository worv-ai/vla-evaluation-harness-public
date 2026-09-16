"""The report preserves smoke failures and records immutable image identities."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_report_smoke_status_and_entrypoint(tmp_path, monkeypatch, exit_code):
    spec = importlib.util.spec_from_file_location(
        "docker_report", Path(__file__).resolve().parents[1] / "docker/report.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def read(command, **kwargs):
        if command[1:3] == ["image", "inspect"]:
            return json.dumps([{"Id": "sha256:fixed", "Size": 123, "Config": {"Env": ["PATH=/env/bin:/usr/bin"]}}])
        assert "sha256:fixed" in command
        return json.dumps({"directories": [], "packages": []})

    def run(command, **kwargs):
        assert command[-1] == "sha256:fixed"
        assert "--entrypoint" not in command  # retain Xvfb/tini wrappers
        assert "PATH=/tmp/vla-smoke-bin:/env/bin:/usr/bin" in command
        return SimpleNamespace(returncode=exit_code, stdout="smoke output", stderr="diagnostic")

    monkeypatch.setattr(module.subprocess, "check_output", read)
    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["report", "before", "after", "--config", "/eval.yaml", "--action", "[0]", "--output", str(output)],
    )
    assert module.main() == exit_code
    report = json.loads(output.read_text())
    assert report["after"]["id"] == "sha256:fixed"
    assert report["smoke"]["status"] == ("failed" if exit_code else "passed")
    assert report["smoke"]["stderr"] == "diagnostic"
