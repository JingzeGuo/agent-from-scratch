"""OS sandbox for run_command: writes stay in the workspace and network is off.

macOS uses sandbox-exec (Seatbelt); Linux uses bubblewrap. Both apply the same
policy: read anything, write only the workspace and temporary directories,
keep PROTECTED_DIRS read-only, and deny network access.
"""

import shutil
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path

# Commands run in bash when installed: it supports pipefail, while /bin/sh is
# dash on Debian and Ubuntu.
SHELL = shutil.which("bash") or "/bin/sh"

# Used outside the sandbox later (git hooks, agent memory), so never writable.
PROTECTED_DIRS = (".git", ".agents")

SEATBELT_PROFILE = """(version 1)
(deny default)
(allow process-fork process-exec signal sysctl-read)
(allow file-read*)
(allow file-write*
  (require-all
    (require-any
      (subpath (param "WORKSPACE"))
      (subpath (param "TMP_DIR"))
      (subpath "/private/tmp")
      (literal "/dev/null"))
{protected}))
""".format(
    protected="\n".join(
        f'    (require-not (subpath (param "PROTECTED_{i}")))'
        for i in range(len(PROTECTED_DIRS))
    )
)


def sandbox_argv(command: str, workspace_root: Path) -> list[str] | None:
    """Wrap a shell command in the platform sandbox, or None if unavailable."""
    if not sandbox_available():
        return None
    return _wrap(workspace_root.resolve(), [SHELL, "-c", command])


@cache
def sandbox_available() -> bool:
    """Whether this host can run sandboxed commands (probed once)."""
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        return False
    # Installed is not enough: containers and AppArmor can block namespaces.
    probe = _wrap(Path(tempfile.gettempdir()).resolve(), [SHELL, "-c", "true"])
    try:
        return subprocess.run(probe, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _wrap(root: Path, argv: list[str]) -> list[str]:
    if sys.platform == "darwin":
        params = {
            "WORKSPACE": root,
            "TMP_DIR": Path(tempfile.gettempdir()).resolve(),
            **{f"PROTECTED_{i}": root / name for i, name in enumerate(PROTECTED_DIRS)},
        }
        defines = [
            arg for key, value in params.items() for arg in ("-D", f"{key}={value}")
        ]
        return ["sandbox-exec", "-p", SEATBELT_PROFILE, *defines, *argv]
    protected = [
        arg
        for name in PROTECTED_DIRS
        if (root / name).exists()
        for arg in ("--ro-bind", str(root / name), str(root / name))
    ]
    return [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", str(root), str(root),
        *protected,
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent",
        *argv,
    ]  # fmt: skip
