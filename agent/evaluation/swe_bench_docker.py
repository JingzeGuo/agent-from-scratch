"""Run SWE-bench patch generation inside the instance's prebuilt evaluation image.

The image's /testbed holds the repository at the base commit with its test
dependencies (and compiled extensions) installed. It is copied to a host
workspace, its Git history is replaced by a single baseline commit so no later
revision can leak, and the workspace is mounted back at /testbed. File tools
edit the host copy; commands run in the container without network access.
"""

import math
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from uuid import uuid4

CONTAINER_ROOT = PurePosixPath("/testbed")
PLATFORM = "linux/amd64"


def instance_image(instance_id: str) -> str:
    return f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()


class SweBenchContainer:
    def __init__(self, image: str, workspace: Path) -> None:
        self.image = image
        self.workspace = workspace
        self.name = f"agent-swe-{uuid4().hex[:12]}"

    def start(self) -> None:
        """Copy /testbed to the workspace, reset its history, and mount it back."""
        _docker("create", "--platform", PLATFORM, "--name", self.name, self.image)
        try:
            _docker("cp", f"{self.name}:{CONTAINER_ROOT}", str(self.workspace))
        finally:
            _docker("rm", "-f", self.name)
        self.workspace = self.workspace.resolve()
        shutil.rmtree(self.workspace / ".git")
        for args in (
            ["init", "--quiet"],
            ["add", "-A"],
            ["-c", "user.name=agent", "-c", "user.email=agent@localhost",
             "commit", "--quiet", "--no-verify", "-m", "baseline"],
        ):  # fmt: skip
            subprocess.run(
                ["git", *args], cwd=self.workspace, check=True, capture_output=True
            )
        _docker(
            "run", "--detach", "--platform", PLATFORM, "--network", "none",
            "--name", self.name,
            "--volume", f"{self.workspace}:{CONTAINER_ROOT}",
            self.image, "sleep", "infinity",
        )  # fmt: skip

    def launch(self, command: str, cwd: Path, timeout_seconds: float) -> list[str]:
        """A login shell activates the image's test environment."""
        workdir = CONTAINER_ROOT / cwd.relative_to(self.workspace).as_posix()
        return [
            "docker", "exec", "--workdir", str(workdir), self.name,
            # Kill the command inside the container, not just the docker client.
            "timeout", "--signal=KILL", str(math.ceil(timeout_seconds)),
            "bash", "-lc", command,
        ]  # fmt: skip

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def _docker(*args: str) -> None:
    completed = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=600
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"docker {args[0]} failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
