from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = ROOT / ".venv"
PYTHON_VERSION_FILE = ROOT / ".python-version"
RUNTIME_REQUIREMENTS = ROOT / "requirements.txt"
DEV_REQUIREMENTS = ROOT / "requirements-dev.txt"
BUILD_REQUIREMENTS = [
    "pip==24.0",
    "setuptools==69.5.1",
    "wheel==0.43.0",
]
PINNED_REQUIREMENT = re.compile(r"^[A-Za-z0-9_.-]+(\[[A-Za-z0-9_,.-]+\])?\s*(==|===).+")


def read_expected_python() -> tuple[int, int]:
    raw_version = PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip()
    parts = raw_version.split(".")
    if len(parts) < 2:
        raise SystemExit(".python-version must contain at least major.minor, for example 3.10")
    return int(parts[0]), int(parts[1])


def ensure_expected_python() -> None:
    expected_major, expected_minor = read_expected_python()
    actual = sys.version_info
    if (actual.major, actual.minor) != (expected_major, expected_minor):
        raise SystemExit(
            "This project expects Python "
            f"{expected_major}.{expected_minor}.x, but this interpreter is "
            f"{actual.major}.{actual.minor}.{actual.micro}: {sys.executable}"
        )


def requirement_lines(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing requirements file: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def validate_pinned_requirements(path: Path, seen: set[Path] | None = None) -> None:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)

    for line in requirement_lines(path):
        if line.startswith(("-r ", "--requirement ")):
            nested = line.split(maxsplit=1)[1]
            validate_pinned_requirements((path.parent / nested).resolve(), seen)
            continue
        if line.startswith(("-c ", "--constraint ", "--index-url ", "--extra-index-url ")):
            continue
        if " @ " in line or line.startswith(("-e ", "--editable ")):
            continue
        if not PINNED_REQUIREMENT.match(line):
            raise SystemExit(
                f"Unpinned dependency in {path.name}: {line!r}. "
                "Use package==version for reproducible installs."
            )


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"+ {printable}")
    subprocess.run(command, cwd=ROOT, check=True)


def create_venv(venv_dir: Path, recreate: bool) -> Path:
    if recreate and venv_dir.exists():
        import shutil

        shutil.rmtree(venv_dir)

    if not venv_python(venv_dir).exists():
        builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade=False)
        builder.create(venv_dir)

    return venv_python(venv_dir)


def install_requirements(python: Path, include_dev: bool) -> None:
    run(
        [
            str(python),
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            *BUILD_REQUIREMENTS,
        ]
    )

    validate_pinned_requirements(RUNTIME_REQUIREMENTS)
    run([str(python), "-m", "pip", "--disable-pip-version-check", "install", "-r", str(RUNTIME_REQUIREMENTS)])

    if include_dev:
        validate_pinned_requirements(DEV_REQUIREMENTS)
        run([str(python), "-m", "pip", "--disable-pip-version-check", "install", "-r", str(DEV_REQUIREMENTS)])


def install_project(python: Path) -> None:
    run([str(python), "-m", "pip", "--disable-pip-version-check", "install", "--editable", str(ROOT)])


def activation_hint(venv_dir: Path) -> str:
    try:
        relative = venv_dir.relative_to(ROOT)
        windows_path = f".\\{relative}"
        posix_path = relative.as_posix()
    except ValueError:
        windows_path = str(venv_dir)
        posix_path = venv_dir.as_posix()
    return (
        "Activate the environment with one of these commands:\n"
        f"  Windows PowerShell: {windows_path}\\Scripts\\Activate.ps1\n"
        f"  Linux/macOS: source {posix_path}/bin/activate"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the reproducible Python environment for Carbono Solo.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV, help="Virtual environment path. Default: .venv")
    parser.add_argument("--no-dev", action="store_true", help="Install only runtime dependencies.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the virtual environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_expected_python()

    venv_dir = args.venv
    if not venv_dir.is_absolute():
        venv_dir = ROOT / venv_dir

    python = create_venv(venv_dir, args.recreate)
    install_requirements(python, include_dev=not args.no_dev)
    install_project(python)

    print()
    print(f"Environment ready: {venv_dir}")
    print(activation_hint(venv_dir))
    print(f"Verify with: {python} scripts/verify_env.py")


if __name__ == "__main__":
    main()
