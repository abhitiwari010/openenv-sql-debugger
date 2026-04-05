#!/usr/bin/env python3
"""
Pre-submission checks for the OpenEnv hackathon SQL agent project.

Run from repo root:
  python hf/pre-validation-script.py

Or:
  cd hf && python pre-validation-script.py

Checks:
  - Required files exist (inference.py, Dockerfile, openenv.yaml, README, models, etc.)
  - inference.py contains mandatory stdout format helpers and tags
  - Python syntax compiles
  - openenv validate (if CLI installed)

Does not call external LLM APIs (no HF_TOKEN required for this script).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ok(msg: str) -> None:
    print(f"[OK] {msg}", flush=True)


def fail(msg: str) -> bool:
    print(f"[FAIL] {msg}", flush=True)
    return False


def check_files() -> bool:
    good = True
    required = [
        ROOT / "inference.py",
        ROOT / "Dockerfile",
        ROOT / "openenv.yaml",
        ROOT / "README.md",
        ROOT / "models.py",
        ROOT / "client.py",
        ROOT / "server" / "app.py",
        ROOT / "server" / "environment.py",
        ROOT / "core" / "tasks.py",
        ROOT / "core" / "grader.py",
    ]
    for p in required:
        if not p.is_file():
            good &= fail(f"Missing file: {p.relative_to(ROOT)}")
    if good:
        ok("Required project files present")
    return good


def check_inference_stdout_contract() -> bool:
    path = ROOT / "inference.py"
    text = path.read_text(encoding="utf-8")
    tokens = [
        "def log_start",
        "def log_step",
        "def log_end",
        "[START]",
        "[STEP]",
        "[END]",
        "OpenAI(",
        "chat.completions.create",
        "HF_TOKEN",
        "API_BASE_URL",
        "MODEL_NAME",
    ]
    good = True
    for t in tokens:
        if t not in text:
            good &= fail(f"inference.py missing required fragment: {t!r}")
    if good:
        ok("inference.py stdout contract + OpenAI client config markers")
    return good


def check_syntax() -> bool:
    good = True
    for py in [
        ROOT / "inference.py",
        ROOT / "server" / "app.py",
        ROOT / "server" / "environment.py",
        ROOT / "models.py",
    ]:
        try:
            ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as e:
            good &= fail(f"Syntax error in {py.relative_to(ROOT)}: {e}")
    if good:
        ok("Core Python files parse (syntax)")
    return good


def check_openenv_validate() -> bool:
    try:
        r = subprocess.run(
            ["openenv", "validate"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        print(
            "[WARN] openenv CLI not on PATH — install openenv-core and ensure `openenv` is available",
            flush=True,
        )
        return True
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return fail("openenv validate returned non-zero")
    ok("openenv validate")
    return True


def main() -> int:
    print("=== Pre-submission validation ===", flush=True)
    print(f"ROOT: {ROOT}", flush=True)
    all_ok = True
    all_ok &= check_files()
    all_ok &= check_inference_stdout_contract()
    all_ok &= check_syntax()
    all_ok &= check_openenv_validate()
    if all_ok:
        print("\n=== All automated checks passed ===", flush=True)
        return 0
    print("\n=== Fix failures above before submitting ===", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
