"""Least-privilege local workspace bridge and outbound Laplace Client agent."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast
from urllib.parse import urlsplit, urlunsplit

from .sync_protocol import clean_logical_path

JsonObject: TypeAlias = dict[str, object]
DEFAULT_TOKEN_ENV = "LAPLACE_CLIENT_TOKEN"
DEFAULT_COMMANDS = (
    "git",
    "python",
    "python3",
    "pytest",
    "ruff",
    "mypy",
    "make",
    "cmake",
    "iverilog",
    "verilator",
    "yosys",
)
SYSTEM_EXECUTABLE_ROOTS = (Path("/usr"), Path("/bin"))


class ClientBridgeError(RuntimeError):
    """Local workspace or authenticated transport operation was rejected."""


def _atomic_json(path: Path, value: object, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            os.chmod(path, mode)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class WorkspaceGrant:
    workspace_id: str
    root: str
    writable: bool
    allowed_commands: tuple[str, ...]


class WorkspaceRegistry:
    """Persistent explicit grants; no directory is exposed implicitly."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path.resolve()

    def _load(self) -> dict[str, WorkspaceGrant]:
        if not self.state_path.exists():
            return {}
        try:
            raw: object = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClientBridgeError("workspace_registry_invalid") from exc
        values = raw.get("workspaces") if isinstance(raw, dict) else None
        if not isinstance(values, dict):
            raise ClientBridgeError("workspace_registry_invalid")
        try:
            return {
                str(key): WorkspaceGrant(
                    workspace_id=str(value["workspace_id"]),
                    root=str(value["root"]),
                    writable=bool(value["writable"]),
                    allowed_commands=tuple(value["allowed_commands"]),
                )
                for key, value in values.items()
                if isinstance(value, dict)
            }
        except (KeyError, TypeError) as exc:
            raise ClientBridgeError("workspace_registry_invalid") from exc

    def _save(self, grants: Mapping[str, WorkspaceGrant]) -> None:
        _atomic_json(
            self.state_path,
            {
                "schema_version": 1,
                "workspaces": {
                    key: asdict(value) for key, value in sorted(grants.items())
                },
            },
        )

    def list(self) -> tuple[WorkspaceGrant, ...]:
        return tuple(self._load().values())

    def get(self, workspace_id: str) -> WorkspaceGrant:
        try:
            return self._load()[workspace_id]
        except KeyError as exc:
            raise ClientBridgeError("workspace_not_granted") from exc

    def grant(
        self,
        root: Path,
        *,
        writable: bool,
        allowed_commands: Sequence[str] = DEFAULT_COMMANDS,
    ) -> WorkspaceGrant:
        expanded = root.expanduser()
        if expanded.is_symlink():
            raise ClientBridgeError("workspace_root_invalid")
        resolved = expanded.resolve(strict=True)
        if not resolved.is_dir():
            raise ClientBridgeError("workspace_root_invalid")
        commands = tuple(sorted(set(allowed_commands)))
        if any(not item or Path(item).name != item for item in commands):
            raise ClientBridgeError("allowed_command_invalid")
        workspace_id = "ws-" + secrets.token_hex(12)
        grant = WorkspaceGrant(workspace_id, str(resolved), writable, commands)
        grants = self._load()
        grants[workspace_id] = grant
        self._save(grants)
        return grant

    def revoke(self, workspace_id: str) -> None:
        grants = self._load()
        if workspace_id not in grants:
            raise ClientBridgeError("workspace_not_granted")
        del grants[workspace_id]
        self._save(grants)


class LocalWorkspace:
    """Operations confined to one immutable canonical grant root."""

    def __init__(self, grant: WorkspaceGrant) -> None:
        self.grant = grant
        recorded = Path(grant.root)
        if recorded.is_symlink():
            raise ClientBridgeError("workspace_root_invalid")
        self.root = recorded.resolve(strict=True)
        if self.root != recorded or not self.root.is_dir():
            raise ClientBridgeError("workspace_root_invalid")

    def resolve(self, logical_path: str) -> Path:
        try:
            clean = clean_logical_path(logical_path)
        except ValueError as exc:
            raise ClientBridgeError("workspace_path_invalid") from exc
        current = self.root
        for part in Path(clean).parts:
            current = current / part
            if current.is_symlink():
                raise ClientBridgeError("workspace_symlink_rejected")
        candidate = (self.root / clean).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ClientBridgeError("workspace_path_escape") from exc
        return candidate

    def _parent_descriptor(self, logical_path: str, *, create: bool) -> tuple[int, str]:
        try:
            clean = clean_logical_path(logical_path)
        except ValueError as exc:
            raise ClientBridgeError("workspace_path_invalid") from exc
        parts = Path(clean).parts
        if not parts:
            raise ClientBridgeError("workspace_path_invalid")
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.root, os.O_RDONLY | directory | nofollow)
        try:
            for part in parts[:-1]:
                try:
                    child = os.open(
                        part,
                        os.O_RDONLY | directory | nofollow,
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    if not create:
                        raise ClientBridgeError("workspace_path_invalid")
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    child = os.open(
                        part,
                        os.O_RDONLY | directory | nofollow,
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise ClientBridgeError("workspace_symlink_rejected") from exc
                os.close(descriptor)
                descriptor = child
            return descriptor, parts[-1]
        except Exception:
            os.close(descriptor)
            raise

    def read_text(self, logical_path: str, *, maximum_bytes: int = 2_000_000) -> str:
        # Preserve the stable public rejection category before the race-safe openat walk.
        self.resolve(logical_path)
        descriptor, name = self._parent_descriptor(logical_path, create=False)
        try:
            try:
                file_descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ClientBridgeError("workspace_read_rejected") from exc
            try:
                stat = os.fstat(file_descriptor)
                if stat.st_size > maximum_bytes:
                    raise ClientBridgeError("workspace_read_rejected")
                content = os.read(file_descriptor, maximum_bytes + 1)
            finally:
                os.close(file_descriptor)
        finally:
            os.close(descriptor)
        if len(content) > maximum_bytes:
            raise ClientBridgeError("workspace_read_rejected")
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ClientBridgeError("workspace_binary_rejected") from exc

    def list_files(self, logical_path: str = ".", *, limit: int = 1000) -> list[str]:
        if not 1 <= limit <= 10_000:
            raise ClientBridgeError("workspace_list_limit_invalid")
        base = self.root if logical_path == "." else self.resolve(logical_path)
        if base.is_symlink() or not base.is_dir():
            raise ClientBridgeError("workspace_directory_required")
        output: list[str] = []
        for item in sorted(base.rglob("*")):
            if item.is_symlink() or ".git" in item.parts or not item.is_file():
                continue
            resolved = item.resolve(strict=True)
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise ClientBridgeError("workspace_path_escape") from exc
            output.append(item.relative_to(self.root).as_posix())
            if len(output) >= limit:
                break
        return output

    def search_text(
        self,
        query: str,
        *,
        limit: int = 100,
        maximum_file_bytes: int = 1_000_000,
    ) -> list[JsonObject]:
        if not query or len(query) > 500 or not 1 <= limit <= 1_000:
            raise ClientBridgeError("workspace_query_invalid")
        found: list[JsonObject] = []
        for relative in self.list_files(limit=5000):
            try:
                text = self.read_text(relative, maximum_bytes=maximum_file_bytes)
            except ClientBridgeError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if query.casefold() in line.casefold():
                    found.append({"path": relative, "line": number, "text": line[:500]})
                    if len(found) >= limit:
                        return found
        return found

    def write_text(
        self, logical_path: str, content: str, *, maximum_bytes: int = 2_000_000
    ) -> None:
        if not self.grant.writable:
            raise ClientBridgeError("workspace_read_only")
        encoded = content.encode("utf-8")
        if len(encoded) > maximum_bytes:
            raise ClientBridgeError("workspace_write_too_large")
        descriptor, name = self._parent_descriptor(logical_path, create=True)
        temporary = f".{name}.laplace-{secrets.token_hex(8)}.tmp"
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            output = os.open(temporary, flags, 0o600, dir_fd=descriptor)
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(output, view)
                    view = view[written:]
                os.fsync(output)
            finally:
                os.close(output)
            os.replace(temporary, name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        finally:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except FileNotFoundError:
                pass
            os.close(descriptor)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        timeout: float = 120.0,
        maximum_output_bytes: int = 1_000_000,
        cancelled: Callable[[], bool] | None = None,
    ) -> JsonObject:
        if (
            not argv
            or len(argv) > 128
            or any(not isinstance(item, str) or len(item) > 10_000 for item in argv)
        ):
            raise ClientBridgeError("workspace_command_invalid")
        executable = Path(argv[0]).name
        if executable != argv[0] or executable not in self.grant.allowed_commands:
            raise ClientBridgeError("workspace_command_not_allowed")
        sandbox = shutil.which("bwrap") if sys.platform.startswith("linux") else None
        if sandbox is None:
            raise ClientBridgeError("workspace_command_sandbox_unavailable")
        resolved_executable = self._sandbox_executable(executable)
        if not 0.1 <= timeout <= 600 or not 1 <= maximum_output_bytes <= 2_000_000:
            raise ClientBridgeError("workspace_command_limits_invalid")
        working = self.root if cwd == "." else self.resolve(cwd)
        if working.is_symlink() or not working.is_dir():
            raise ClientBridgeError("workspace_command_cwd_invalid")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"LANG", "LC_ALL", "LC_CTYPE", "TZ"}
        }
        environment.update(
            {
                "HOME": "/tmp/laplace-home",
                "PATH": self._sandbox_path(),
                "XDG_CACHE_HOME": "/tmp/laplace-cache",
                "XDG_CONFIG_HOME": "/tmp/laplace-config",
            }
        )
        sandbox_argv = self._sandbox_argv(
            Path(sandbox), resolved_executable, argv[1:], working
        )
        started = time.monotonic()
        was_cancelled = False
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(  # nosec B603 - fixed allowlist and argv, no shell
                sandbox_argv,
                cwd=self.root,
                env=environment,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=os.name != "nt",
            )
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    was_cancelled = True
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    break
                if time.monotonic() - started > timeout:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise ClientBridgeError("workspace_command_timeout")
                time.sleep(0.1)
            if was_cancelled:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            else:
                process.wait()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout_bytes = stdout_file.read(maximum_output_bytes + 1)
            stdout_limited = stdout_bytes[:maximum_output_bytes]
            remaining = maximum_output_bytes - len(stdout_limited)
            stderr_bytes = stderr_file.read(remaining + 1)
            stderr_limited = stderr_bytes[:remaining]
        return {
            "returncode": process.returncode,
            "stdout": stdout_limited.decode("utf-8", errors="replace"),
            "stderr": stderr_limited.decode("utf-8", errors="replace"),
            "truncated": (
                len(stdout_bytes) > maximum_output_bytes
                or len(stderr_bytes) > remaining
            ),
            "cancelled": was_cancelled,
            "duration_seconds": round(time.monotonic() - started, 3),
        }

    def _sandbox_path(self) -> str:
        candidates = [
            self.root / ".venv/bin",
            self.root / "venv/bin",
            Path("/usr/local/bin"),
            Path("/usr/bin"),
            Path("/bin"),
        ]
        return os.pathsep.join(str(item) for item in candidates if item.is_dir())

    def _sandbox_executable(self, executable: str) -> Path:
        resolved = shutil.which(executable, path=self._sandbox_path())
        if resolved is None:
            raise ClientBridgeError("workspace_command_tool_unavailable")
        path = Path(resolved).resolve(strict=True)
        for root in (self.root, *SYSTEM_EXECUTABLE_ROOTS):
            try:
                path.relative_to(root.resolve(strict=True))
                return path
            except ValueError:
                continue
        raise ClientBridgeError("workspace_command_tool_outside_sandbox")

    def _sandbox_argv(
        self,
        sandbox: Path,
        executable: Path,
        arguments: Sequence[str],
        working: Path,
    ) -> list[str]:
        command = [
            str(sandbox),
            "--die-with-parent",
            "--unshare-all",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/laplace-home",
            "--dir",
            "/tmp/laplace-cache",
            "--dir",
            "/tmp/laplace-config",
        ]
        current = Path("/")
        for part in self.root.parts[1:-1]:
            current /= part
            if current != Path("/tmp"):
                command.extend(("--dir", str(current)))
        command.extend(
            (
                "--bind" if self.grant.writable else "--ro-bind",
                str(self.root),
                str(self.root),
                "--chdir",
                str(working),
                "--",
                str(executable),
                *arguments,
            )
        )
        return command

    def git_inspect(self) -> JsonObject:
        if "git" not in self.grant.allowed_commands:
            raise ClientBridgeError("workspace_command_not_allowed")
        commands = {
            "status": ("git", "status", "--short", "--branch"),
            "head": ("git", "rev-parse", "HEAD"),
            "diff_stat": ("git", "diff", "--stat"),
        }
        return {
            key: self.run(command, timeout=30, maximum_output_bytes=200_000)
            for key, command in commands.items()
        }

    def execute(
        self,
        action: str,
        arguments: Mapping[str, object],
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> JsonObject:
        def integer(name: str, default: int) -> int:
            value = arguments.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ClientBridgeError(f"workspace_{name}_invalid")
            return value

        def number(name: str, default: float) -> float:
            value = arguments.get(name, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ClientBridgeError(f"workspace_{name}_invalid")
            return float(value)

        if action == "list":
            return {
                "files": self.list_files(
                    str(arguments.get("path", ".")),
                    limit=integer("limit", 1000),
                )
            }
        if action == "read":
            return {"content": self.read_text(str(arguments.get("path", "")))}
        if action == "search":
            return {
                "matches": self.search_text(
                    str(arguments.get("query", "")),
                    limit=integer("limit", 100),
                )
            }
        if action == "write":
            path = arguments.get("path")
            content = arguments.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                raise ClientBridgeError("workspace_write_invalid")
            self.write_text(path, content)
            return {"written": path, "bytes": len(content.encode("utf-8"))}
        if action == "git":
            return self.git_inspect()
        if action == "run":
            raw_argv = arguments.get("argv")
            if not isinstance(raw_argv, list) or not all(
                isinstance(item, str) for item in raw_argv
            ):
                raise ClientBridgeError("workspace_command_invalid")
            return self.run(
                raw_argv,
                cwd=str(arguments.get("cwd", ".")),
                timeout=number("timeout", 120.0),
                maximum_output_bytes=integer("maximum_output_bytes", 1_000_000),
                cancelled=cancelled,
            )
        raise ClientBridgeError("workspace_action_not_supported")


def detected_capabilities(registry: WorkspaceRegistry) -> JsonObject:
    command_sandbox = sys.platform.startswith("linux") and shutil.which("bwrap") is not None
    grants = registry.list()

    def available(grant: WorkspaceGrant, name: str) -> bool:
        try:
            LocalWorkspace(grant)._sandbox_executable(name)
            return True
        except (ClientBridgeError, OSError):
            return False

    if grants:
        tools = {
            name: command_sandbox and any(available(grant, name) for grant in grants)
            for name in DEFAULT_COMMANDS
        }
    else:
        system_path = os.pathsep.join(("/usr/local/bin", "/usr/bin", "/bin"))
        tools = {
            name: command_sandbox and shutil.which(name, path=system_path) is not None
            for name in DEFAULT_COMMANDS
        }
    actions = ["list", "read", "search", "write", "cancel"]
    if command_sandbox:
        actions.extend(("git", "run"))
    return {
        "protocol_version": 1,
        "platform": sys.platform,
        "hostname": socket.gethostname(),
        "tools": tools,
        "command_sandbox": {
            "available": command_sandbox,
            "implementation": "bubblewrap" if command_sandbox else None,
            "network": "isolated" if command_sandbox else None,
        },
        "actions": actions,
        "workspaces": [
            {
                "workspace_id": item.workspace_id,
                "writable": item.writable,
                "allowed_commands": list(item.allowed_commands),
                "available_commands": [
                    name
                    for name in item.allowed_commands
                    if command_sandbox and available(item, name)
                ],
            }
            for item in grants
        ],
    }


@dataclass(frozen=True)
class ClientConnection:
    endpoint: str
    token_env_var: str
    device_id: str
    name: str


class LaplaceClientTransport:
    """Outbound HTTPS/loopback transport using the normal Laplace bearer identity."""

    def __init__(self, endpoint: str, token_env_var: str, *, timeout: float = 30.0) -> None:
        parsed = urlsplit(endpoint)
        loopback = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
        if (
            (not loopback and (parsed.scheme != "https" or not parsed.hostname))
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ClientBridgeError("client_endpoint_must_use_https_or_loopback")
        self.origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        self.token_env_var = token_env_var
        self.timeout = timeout

    def _token(self) -> str:
        token = os.environ.get(self.token_env_var, "")
        if len(token) < 24:
            raise ClientBridgeError(f"missing_client_token_env:{self.token_env_var}")
        return token

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> JsonObject:
        data = (
            json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            self.origin + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "laplace-client/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # nosec B310
                body = response.read(2_100_001)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4_096).decode("utf-8", errors="replace")
            raise ClientBridgeError(f"client_http_error:{exc.code}:{detail[:200]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ClientBridgeError(f"client_connection_failed:{type(exc).__name__}") from exc
        if len(body) > 2_100_000:
            raise ClientBridgeError("client_response_too_large")
        try:
            raw: object = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClientBridgeError("client_response_invalid") from exc
        if not isinstance(raw, dict):
            raise ClientBridgeError("client_response_invalid")
        return cast(JsonObject, raw)


def _connection_path(state_path: Path) -> Path:
    return state_path.resolve().with_name("connection.json")


def _load_connection(state_path: Path) -> ClientConnection:
    path = _connection_path(state_path)
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError
        return ClientConnection(
            endpoint=str(raw["endpoint"]),
            token_env_var=str(raw["token_env_var"]),
            device_id=str(raw["device_id"]),
            name=str(raw["name"]),
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ClientBridgeError("client_not_paired") from exc


def pair_client(
    registry: WorkspaceRegistry,
    *,
    endpoint: str,
    token_env_var: str,
    name: str,
) -> JsonObject:
    connection_path = _connection_path(registry.state_path)
    existing: ClientConnection | None = None
    if connection_path.exists():
        existing = _load_connection(registry.state_path)
    transport = LaplaceClientTransport(endpoint, token_env_var)
    payload: JsonObject = {
        "name": name,
        "capabilities": detected_capabilities(registry),
        "device_id": (
            existing.device_id
            if existing is not None
            and existing.endpoint == transport.origin
            and existing.token_env_var == token_env_var
            else None
        ),
    }
    result = transport.request("POST", "/api/v1/client/devices/pair", payload)
    device_id = result.get("device_id")
    if not isinstance(device_id, str):
        raise ClientBridgeError("client_pair_response_invalid")
    _atomic_json(
        connection_path,
        {
            "schema_version": 1,
            "endpoint": transport.origin,
            "token_env_var": token_env_var,
            "device_id": device_id,
            "name": name,
        },
    )
    return {"paired": True, **result, "credentials_stored": False}


def _cancel_checker(
    transport: LaplaceClientTransport, operation_id: str
) -> Callable[[], bool]:
    last_checked = 0.0
    last_value = False

    def check() -> bool:
        nonlocal last_checked, last_value
        now = time.monotonic()
        if now - last_checked < 0.5:
            return last_value
        last_checked = now
        value = transport.request("GET", f"/api/v1/client/operations/{operation_id}")
        last_value = bool(value.get("cancellation_requested")) or value.get("state") == "CANCELLED"
        return last_value

    return check


def serve_once(registry: WorkspaceRegistry, connection: ClientConnection) -> JsonObject:
    transport = LaplaceClientTransport(connection.endpoint, connection.token_env_var)
    capabilities = detected_capabilities(registry)
    transport.request(
        "POST",
        f"/api/v1/client/devices/{connection.device_id}/heartbeat",
        {"capabilities": capabilities},
    )
    claimed = transport.request(
        "GET", f"/api/v1/client/devices/{connection.device_id}/operations/next"
    )
    operation = claimed.get("operation")
    if operation is None:
        return {"status": "IDLE", "device_id": connection.device_id}
    if not isinstance(operation, dict):
        raise ClientBridgeError("client_operation_invalid")
    operation_id = str(operation.get("operation_id", ""))
    workspace_id = str(operation.get("workspace_id", ""))
    action = str(operation.get("action", ""))
    arguments = operation.get("arguments")
    if not isinstance(arguments, dict):
        raise ClientBridgeError("client_operation_invalid")
    failed = False
    try:
        workspace = LocalWorkspace(registry.get(workspace_id))
        result = workspace.execute(
            action,
            cast(JsonObject, arguments),
            cancelled=_cancel_checker(transport, operation_id),
        )
    except (ClientBridgeError, OSError) as exc:
        failed = True
        result = {"error": str(exc), "error_type": type(exc).__name__}
    completed = transport.request(
        "POST",
        (
            f"/api/v1/client/devices/{connection.device_id}/operations/"
            f"{operation_id}/result"
        ),
        {"result": result, "failed": failed},
    )
    return {"status": "PROCESSED", "operation": completed}


def serve_forever(
    registry: WorkspaceRegistry,
    connection: ClientConnection,
    *,
    poll_interval: float,
) -> None:
    backoff = poll_interval
    while True:
        try:
            serve_once(registry, connection)
            backoff = poll_interval
            time.sleep(poll_interval)
        except ClientBridgeError as exc:
            print(json.dumps({"event": "reconnect", "error": str(exc)}), file=sys.stderr)
            time.sleep(backoff)
            backoff = min(30.0, max(poll_interval, backoff * 2))


def _default_state() -> Path:
    value = os.environ.get("LAPLACE_CLIENT_STATE")
    return Path(value) if value else Path.home() / ".laplace" / "client" / "workspaces.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laplace-client")
    parser.add_argument("--state", type=Path, default=_default_state())
    sub = parser.add_subparsers(dest="command", required=True)
    grant = sub.add_parser("grant")
    grant.add_argument("root", type=Path)
    grant.add_argument("--read-only", action="store_true")
    grant.add_argument("--allow", action="append", default=[])
    revoke = sub.add_parser("revoke")
    revoke.add_argument("workspace_id")
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("workspace_id")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.add_argument("--cwd", default=".")
    pair = sub.add_parser("pair")
    pair.add_argument("--endpoint", required=True)
    pair.add_argument("--token-env-var", default=DEFAULT_TOKEN_ENV)
    pair.add_argument("--name", default=socket.gethostname())
    serve = sub.add_parser("serve")
    serve.add_argument("--poll-interval", type=float, default=1.0)
    serve.add_argument("--once", action="store_true")
    unpair = sub.add_parser("unpair")
    unpair.add_argument("--keep-local-state", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    registry = WorkspaceRegistry(args.state)
    try:
        if args.command == "grant":
            commands = tuple(args.allow) if args.allow else DEFAULT_COMMANDS
            payload: object = asdict(
                registry.grant(
                    args.root,
                    writable=not args.read_only,
                    allowed_commands=commands,
                )
            )
        elif args.command == "revoke":
            registry.revoke(args.workspace_id)
            payload = {"revoked": args.workspace_id}
        elif args.command == "status":
            connection = None
            try:
                connection = asdict(_load_connection(registry.state_path))
            except ClientBridgeError:
                pass
            payload = {
                "workspaces": [asdict(item) for item in registry.list()],
                "capabilities": detected_capabilities(registry),
                "connection": connection,
            }
        elif args.command == "run":
            if not args.argv:
                raise ClientBridgeError("workspace_command_invalid")
            payload = LocalWorkspace(registry.get(args.workspace_id)).run(
                args.argv, cwd=args.cwd
            )
        elif args.command == "pair":
            payload = pair_client(
                registry,
                endpoint=args.endpoint,
                token_env_var=args.token_env_var,
                name=args.name,
            )
        elif args.command == "serve":
            active_connection = _load_connection(registry.state_path)
            if not 0.1 <= args.poll_interval <= 60:
                raise ClientBridgeError("client_poll_interval_invalid")
            if args.once:
                payload = serve_once(registry, active_connection)
            else:
                serve_forever(registry, active_connection, poll_interval=args.poll_interval)
                payload = {"status": "STOPPED"}
        else:
            active_connection = _load_connection(registry.state_path)
            transport = LaplaceClientTransport(
                active_connection.endpoint, active_connection.token_env_var
            )
            payload = transport.request(
                "DELETE", f"/api/v1/client/devices/{active_connection.device_id}"
            )
            if not args.keep_local_state:
                _connection_path(registry.state_path).unlink(missing_ok=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (ClientBridgeError, OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
