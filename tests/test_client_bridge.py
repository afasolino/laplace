from __future__ import annotations

from pathlib import Path

import pytest

from research_workspace.client_bridge import (
    ClientBridgeError,
    LocalWorkspace,
    WorkspaceRegistry,
    detected_capabilities,
)


def test_workspace_grant_read_write_and_revoke(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("hello\n", encoding="utf-8")
    registry = WorkspaceRegistry(tmp_path / "state" / "workspaces.json")

    grant = registry.grant(root, writable=True, allowed_commands=("git",))
    workspace = LocalWorkspace(grant)

    assert workspace.read_text("a.txt") == "hello\n"
    workspace.write_text("sub/b.txt", "world\n")
    assert (root / "sub" / "b.txt").read_text(encoding="utf-8") == "world\n"
    assert set(workspace.list_files()) == {"a.txt", "sub/b.txt"}
    assert workspace.search_text("world")[0]["path"] == "sub/b.txt"

    registry.revoke(grant.workspace_id)
    with pytest.raises(ClientBridgeError, match="workspace_not_granted"):
        registry.get(grant.workspace_id)


def test_workspace_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    registry = WorkspaceRegistry(tmp_path / "state.json")
    grant = registry.grant(root, writable=True)
    workspace = LocalWorkspace(grant)

    with pytest.raises(ClientBridgeError):
        workspace.read_text("../outside.txt")

    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ClientBridgeError, match="workspace_symlink_rejected"):
        workspace.read_text("link.txt")


def test_registry_rejects_symlink_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    registry = WorkspaceRegistry(tmp_path / "state.json")
    with pytest.raises(ClientBridgeError, match="workspace_root_invalid"):
        registry.grant(alias, writable=True)


def test_read_only_and_command_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    registry = WorkspaceRegistry(tmp_path / "state.json")
    grant = registry.grant(root, writable=False, allowed_commands=("git",))
    workspace = LocalWorkspace(grant)

    with pytest.raises(ClientBridgeError, match="workspace_read_only"):
        workspace.write_text("x.txt", "x")
    with pytest.raises(ClientBridgeError, match="workspace_command_not_allowed"):
        workspace.run(("python", "-c", "print(1)"))
    with pytest.raises(ClientBridgeError, match="workspace_command_not_allowed"):
        workspace.run(("/usr/bin/git", "status"))


def test_command_sandbox_mounts_only_system_runtime_and_granted_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    registry = WorkspaceRegistry(tmp_path / "state.json")
    workspace = LocalWorkspace(
        registry.grant(root, writable=False, allowed_commands=("python3",))
    )

    command = workspace._sandbox_argv(  # noqa: SLF001 - security contract test
        Path("/usr/bin/bwrap"),
        Path("/usr/bin/python3"),
        ("-V",),
        root,
    )

    assert "--unshare-all" in command
    assert "--ro-bind" in command
    assert "--bind" not in command
    assert command[-2:] == ["/usr/bin/python3", "-V"]
    assert str(root) in command
    assert "/etc" not in command
    assert "/home" not in command
    capabilities = detected_capabilities(registry)
    assert capabilities["command_sandbox"] == {
        "available": True,
        "implementation": "bubblewrap",
        "network": "isolated",
    }
