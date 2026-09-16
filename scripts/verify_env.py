from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION_FILE = ROOT / ".python-version"


def expected_python() -> tuple[int, int]:
    raw_version = PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
    major, minor, *_ = raw_version.split(".")
    return int(major), int(minor)


def is_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")


def main() -> None:
    expected_major, expected_minor = expected_python()
    actual = sys.version_info

    if (actual.major, actual.minor) != (expected_major, expected_minor):
        raise SystemExit(
            "Wrong Python version: expected "
            f"{expected_major}.{expected_minor}.x, got {actual.major}.{actual.minor}.{actual.micro}"
        )

    if not is_virtualenv():
        raise SystemExit("This verification must be run with the project virtual environment Python.")

    subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "check"],
        cwd=ROOT,
        check=True,
    )

    print(f"Python: {actual.major}.{actual.minor}.{actual.micro}")
    print(f"Executable: {sys.executable}")
    print("Environment verification passed.")


if __name__ == "__main__":
    main()

