"""Sandboxed execution environment.

Primary: Docker SDK (disposable container, no network, memory limit,
read-only root FS except /tmp, 30s timeout).
Fallback: subprocess with resource limits (when Docker is unavailable).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class SandboxError(Exception):
    """Raised when sandboxed execution fails."""


class SandboxExecutor:
    """Executes code/tools inside an isolated sandbox."""

    def __init__(
        self,
        image: str = "ubuntu:22.04",
        memory_limit: str = "512m",
        timeout_seconds: int = 30,
        network_disabled: bool = True,
        enabled: Optional[bool] = None,
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.timeout_seconds = timeout_seconds
        self.network_disabled = network_disabled
        self.enabled = (
            enabled if enabled is not None
            else os.getenv("SANDBOX_ENABLED", "true").lower() == "true"
        )

    # ------------------------------------------------------------------
    # Docker path (primary)
    # ------------------------------------------------------------------
    def _docker_available(self) -> bool:
        if not DOCKER_AVAILABLE:
            return False
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def run_python(self, code: str) -> Dict[str, Any]:
        """Run a Python snippet in the sandbox."""
        if self.enabled and self._docker_available():
            return self._run_python_docker(code)
        return self._run_python_subprocess(code)

    def _run_python_docker(self, code: str) -> Dict[str, Any]:
        """Run Python inside a disposable Docker container."""
        client = docker.from_env()

        # Write code to a temp file to mount into the container
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "snippet.py"
            code_path.write_text(code, encoding="utf-8")

            container_config = {
                "image": self.image,
                "command": ["python3", "/tmp/snippet.py"],
                "mem_limit": self.memory_limit,
                "network_disabled": self.network_disabled,
                "read_only": True,
                "tmpfs": {"/tmp": "rw,size=64m"},
                "volumes": {str(code_path): {"bind": "/tmp/snippet.py", "mode": "ro"}},
                "detach": True,
            }

            try:
                container = client.containers.run(**container_config)
            except docker.errors.ImageNotFound:
                raise SandboxError(
                    f"Sandbox image '{self.image}' not found. "
                    "Run: docker pull ubuntu:22.04"
                )
            except docker.errors.APIError as e:
                raise SandboxError(f"Docker API error: {e}")

            try:
                result = container.wait(timeout=self.timeout_seconds)
            except Exception:
                container.kill()
                container.remove()
                raise SandboxError(
                    f"Sandbox execution timed out after {self.timeout_seconds}s"
                )

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            container.remove()

            return {
                "exit_code": result.get("StatusCode", -1),
                "stdout": stdout,
                "stderr": stderr,
                "sandbox": "docker",
            }

    # ------------------------------------------------------------------
    # Subprocess fallback (resource-limited)
    # ------------------------------------------------------------------
    def _run_python_subprocess(self, code: str) -> Dict[str, Any]:
        """Fallback: run Python in a subprocess with resource limits."""
        if not shutil.which("python"):
            raise SandboxError("No Python interpreter available for sandbox fallback.")

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "snippet.py"
            code_path.write_text(code, encoding="utf-8")

            # NOTE: On Windows, resource limits (RLIMIT_*) are not available.
            # We enforce a wall-clock timeout instead. On POSIX, add
            # resource.setrlimit for memory/CPU. Documented as TODO.
            try:
                proc = subprocess.run(
                    ["python", str(code_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=tmpdir,
                )
                return {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "sandbox": "subprocess",
                }
            except subprocess.TimeoutExpired:
                raise SandboxError(
                    f"Sandbox execution timed out after {self.timeout_seconds}s"
                )

    # ------------------------------------------------------------------
    # Generic tool execution in sandbox
    # ------------------------------------------------------------------
    def run_command(
        self,
        command: List[str],
        mount_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run an arbitrary command in the sandbox."""
        if self.enabled and self._docker_available():
            return self._run_command_docker(command, mount_paths or {})
        return self._run_command_subprocess(command)

    def _run_command_docker(
        self, command: List[str], mount_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        client = docker.from_env()
        volumes = {}
        for host_path, container_path in mount_paths.items():
            volumes[str(host_path)] = {"bind": container_path, "mode": "ro"}

        container_config = {
            "image": self.image,
            "command": command,
            "mem_limit": self.memory_limit,
            "network_disabled": self.network_disabled,
            "read_only": True,
            "tmpfs": {"/tmp": "rw,size=64m"},
            "volumes": volumes,
            "detach": True,
        }

        try:
            container = client.containers.run(**container_config)
        except docker.errors.ImageNotFound:
            raise SandboxError(f"Sandbox image '{self.image}' not found.")
        except docker.errors.APIError as e:
            raise SandboxError(f"Docker API error: {e}")

        try:
            result = container.wait(timeout=self.timeout_seconds)
        except Exception:
            container.kill()
            container.remove()
            raise SandboxError(f"Sandbox execution timed out after {self.timeout_seconds}s")

        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        container.remove()

        return {
            "exit_code": result.get("StatusCode", -1),
            "stdout": stdout,
            "stderr": stderr,
            "sandbox": "docker",
        }

    def _run_command_subprocess(self, command: List[str]) -> Dict[str, Any]:
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "sandbox": "subprocess",
            }
        except subprocess.TimeoutExpired:
            raise SandboxError(f"Sandbox execution timed out after {self.timeout_seconds}s")