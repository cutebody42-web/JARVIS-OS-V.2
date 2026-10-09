"""Repair the optional SQLAlchemy C extensions when Windows blocks them.

Some Windows App Control policies reject SQLAlchemy's unsigned ``.pyd``
extensions.  SQLAlchemy supports a pure-Python build, so the bootstrap can
replace only that package when (and only when) the import failure identifies
both a SQLAlchemy C extension and an App Control block.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
SQLALCHEMY_REQUIREMENT = "SQLAlchemy>=2.0,<3"
_PROBE = "import sqlalchemy; print(sqlalchemy.__version__)"
_VERSION_PROBE = (
    "from importlib.metadata import version; "
    "print(version('SQLAlchemy'))"
)
_SQLALCHEMY_2_VERSION = re.compile(
    r"2(?:\.[0-9]+)+"
    r"(?:(?:a|b|rc|post|dev)[0-9]+)*"
    r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?",
    re.IGNORECASE,
)
_SQLALCHEMY_CEXT_MODULES = frozenset(
    {
        "_cache_key_cy",
        "_collections_cy",
        "_immutabledict_cy",
        "_processors_cy",
        "_result_cy",
        "_row_cy",
        "_util_cy",
        # Names used by older SQLAlchemy 2.x binary distributions.
        "collections",
        "immutabledict",
        "processors",
        "resultproxy",
        "util",
    }
)
_BLOCKED_MODULE = re.compile(
    r"dll load failed while importing\s+([a-z0-9_.]+)", re.IGNORECASE
)
_APP_CONTROL_MARKERS = (
    "an application control policy has blocked this file",
    "application control policy has blocked",
    "app control for business",
    "windows defender application control",
    "wdac policy has blocked",
)


class SQLAlchemyImportError(RuntimeError):
    """SQLAlchemy could not be imported and no safe fallback succeeded."""


def _result_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def is_sqlalchemy_app_control_failure(
    output: str, *, platform: str | None = None
) -> bool:
    """Return whether *output* is the narrow failure this fallback repairs."""

    current_platform = sys.platform if platform is None else platform
    if current_platform != "win32":
        return False

    folded = output.casefold()
    if "sqlalchemy" not in folded:
        return False
    if not any(marker in folded for marker in _APP_CONTROL_MARKERS):
        return False

    match = _BLOCKED_MODULE.search(output)
    if match is None:
        return False
    module_name = match.group(1).rsplit(".", 1)[-1].casefold()
    return module_name in _SQLALCHEMY_CEXT_MODULES


def _probe_sqlalchemy(python: str, *, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python, "-c", _PROBE],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


def _installed_sqlalchemy_version(python: str, *, root: Path) -> str:
    result = subprocess.run(
        [python, "-c", _VERSION_PROBE],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    version = result.stdout.strip()
    if result.returncode != 0 or not _SQLALCHEMY_2_VERSION.fullmatch(version):
        details = _result_output(result).strip() or "no version metadata was returned"
        raise SQLAlchemyImportError(
            "Cannot safely select the SQLAlchemy source release to reinstall. "
            f"Expected an installed version matching {SQLALCHEMY_REQUIREMENT}; got: "
            f"{details}"
        )
    return version


def _format_import_failure(result: subprocess.CompletedProcess[str]) -> str:
    details = _result_output(result).strip() or "SQLAlchemy import produced no diagnostics."
    return f"SQLAlchemy import failed with exit code {result.returncode}:\n{details}"


def ensure_sqlalchemy_import(
    python: str | os.PathLike[str] = sys.executable,
    *,
    root: Path = ROOT,
    platform: str | None = None,
) -> bool:
    """Verify SQLAlchemy and repair its App-Control-blocked C extensions.

    Returns ``True`` when a pure-Python reinstall was performed and ``False``
    when the first import succeeded.  Other import failures are reported
    unchanged; they never trigger a broad or speculative reinstall.
    """

    python_path = os.fspath(python)
    initial = _probe_sqlalchemy(python_path, root=root)
    if initial.returncode == 0:
        return False

    initial_output = _result_output(initial)
    if not is_sqlalchemy_app_control_failure(initial_output, platform=platform):
        raise SQLAlchemyImportError(
            _format_import_failure(initial)
            + "\nThe pure-Python fallback was not attempted because this was not "
            "a recognized SQLAlchemy App Control C-extension failure."
        )

    installed_version = _installed_sqlalchemy_version(python_path, root=root)
    print(
        "[setup] Windows App Control blocked a SQLAlchemy C extension; "
        "installing SQLAlchemy's supported pure-Python build.",
        flush=True,
    )
    install_env = os.environ.copy()
    install_env["DISABLE_SQLALCHEMY_CEXT"] = "1"
    install_env.pop("REQUIRE_SQLALCHEMY_CEXT", None)
    subprocess.check_call(
        [
            python_path,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            "--no-binary",
            "SQLAlchemy",
            f"SQLAlchemy=={installed_version}",
        ],
        cwd=root,
        env=install_env,
    )

    verified = _probe_sqlalchemy(python_path, root=root)
    if verified.returncode != 0:
        raise SQLAlchemyImportError(
            "SQLAlchemy still cannot be imported after its pure-Python reinstall.\n"
            + _format_import_failure(verified)
        )
    print(
        f"[setup] SQLAlchemy import verified ({verified.stdout.strip()}).",
        flush=True,
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify SQLAlchemy and repair a Windows App Control C-extension block."
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python interpreter to verify"
    )
    args = parser.parse_args(argv)
    try:
        ensure_sqlalchemy_import(args.python)
    except (SQLAlchemyImportError, subprocess.CalledProcessError) as exc:
        print(f"[setup] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
