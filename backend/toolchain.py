"""Toolchain configuration and native command execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read_config(path=None):
    path = Path(path or os.environ.get("CROSSFHE_CONFIG",ROOT/".crossfhe/toolchain.json"))
    if not path.is_file():
        raise RuntimeError(f"missing toolchain config {path}; run scripts/bootstrap.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def environment(config):
    env = os.environ.copy()
    directories = {str(Path(config[name]).parent) for name in ("cxx","cmake","ninja")}
    env["PATH"] = os.pathsep.join(sorted(directories)) + os.pathsep + env.get("PATH","")
    return env


def command(args, config, *, cwd=None, log=None, timeout=1800):
    started = time.perf_counter()
    result = subprocess.run([str(a) for a in args],cwd=cwd,env=environment(config),
                            capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
    text = result.stdout + result.stderr
    if log:
        Path(log).write_text(text,encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args[0]}\n{text[-8000:]}")
    return result.stdout, time.perf_counter()-started


def _native(config, name):
    suffix = ".exe" if os.name=="nt" else ""
    tool = Path(config["native_build"])/"bin"/(name+suffix)
    if not tool.is_file(): raise RuntimeError(f"native tool missing: {tool}")
    return tool


