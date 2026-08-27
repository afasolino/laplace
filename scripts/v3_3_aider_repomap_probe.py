#!/usr/bin/env python3
"""Render the exact pinned Aider RepoMap against a repo-local immutable snapshot."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import re
import subprocess
import time
from pathlib import Path

import aider.repomap as aider_repomap
import grep_ast
from aider.repomap import RepoMap
from grep_ast import filename_to_lang

UPSTREAM_REVISION = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_FILE_BYTES = 2_000_000


class Char4Model:
    @staticmethod
    def token_count(text: str) -> int:
        return (len(text) + 3) // 4


class SnapshotIO:
    """Small read-only IO surface required by Aider RepoMap."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)

    def read_text(self, filename: str) -> str | None:
        path = Path(filename)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            return None
        if resolved.is_symlink() or not resolved.is_file():
            return None
        try:
            if resolved.stat().st_size > _MAX_FILE_BYTES:
                return None
            return resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    @staticmethod
    def tool_output(*_args: object, **_kwargs: object) -> None:
        return None

    @staticmethod
    def tool_warning(*_args: object, **_kwargs: object) -> None:
        return None

    @staticmethod
    def tool_error(*_args: object, **_kwargs: object) -> None:
        return None


def _revision(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise SystemExit("aider_checkout_revision_unavailable")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--aider-checkout", type=Path, required=True)
    parser.add_argument("--query", default="")
    parser.add_argument("--focus", action="append", default=[])
    parser.add_argument("--token-budget", type=int, default=1000)
    parser.add_argument("--repeat-count", type=int, default=1)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    checkout = args.aider_checkout.resolve(strict=True)
    if (
        not repo.is_dir()
        or not checkout.is_dir()
        or not 1 <= args.token_budget <= 100_000
        or not 1 <= args.repeat_count <= 20
    ):
        raise SystemExit("invalid_probe_arguments")
    if _revision(checkout) != UPSTREAM_REVISION:
        raise SystemExit("aider_revision_mismatch")

    module_file = Path(aider_repomap.__file__).resolve()
    try:
        module_file.relative_to(checkout)
    except ValueError as exc:
        raise SystemExit("aider_import_outside_pinned_checkout") from exc

    grep_ast_version = importlib.metadata.version("grep-ast")
    if grep_ast_version != "0.9.0":
        raise SystemExit("grep_ast_version_mismatch")
    grep_ast_file_raw = getattr(grep_ast, "__file__", None)
    if not isinstance(grep_ast_file_raw, str):
        raise SystemExit("grep_ast_import_missing_file")
    grep_ast_file = Path(grep_ast_file_raw).resolve()

    files = sorted(
        str(path)
        for path in repo.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.relative_to(repo).parts
        and filename_to_lang(str(path))
    )
    mentioned_fnames = set(args.focus)
    mentioned_idents = set(_TOKEN.findall(args.query))

    repomap = RepoMap(
        map_tokens=args.token_budget,
        root=str(repo),
        main_model=Char4Model(),
        io=SnapshotIO(repo),
        max_context_window=args.token_budget + 4096,
        map_mul_no_files=1,
        refresh="always",
    )
    suppressed_stdout = io.StringIO()

    def render_once() -> tuple[str, float]:
        started = time.perf_counter()
        with contextlib.redirect_stdout(suppressed_stdout):
            rendered = repomap.get_ranked_tags_map(
                [],
                files,
                args.token_budget,
                mentioned_fnames,
                mentioned_idents,
                True,
            ) or ""
        return rendered, time.perf_counter() - started

    if args.repeat_count == 1:
        text, elapsed = render_once()
        wall_samples = [elapsed]
        stable = True
        unique_hashes = 1
    else:
        # The warm-up populates Aider's normal tag cache and is not timed.
        warm_text, _ = render_once()
        texts: list[str] = []
        wall_samples: list[float] = []
        for _ in range(args.repeat_count):
            measured_text, measured_elapsed = render_once()
            texts.append(measured_text)
            wall_samples.append(measured_elapsed)
        hashes = {
            hashlib.sha256(item.encode("utf-8")).hexdigest()
            for item in [warm_text, *texts]
        }
        stable = len(hashes) == 1
        unique_hashes = len(hashes)
        text = texts[0]
        elapsed = wall_samples[0]

    result = {
        "provider": "aider-repomap-v33",
        "upstream_revision": UPSTREAM_REVISION,
        "module_file": str(module_file),
        "grep_ast_version": grep_ast_version,
        "grep_ast_module_file": str(grep_ast_file),
        "token_budget": args.token_budget,
        "context_tokens": Char4Model.token_count(text),
        "chars": len(text),
        "files_considered": len(files),
        "wall_time_seconds": elapsed,
        "wall_time_seconds_samples": wall_samples,
        "repeat_count": args.repeat_count,
        "within_process_stable": stable,
        "unique_map_hashes": unique_hashes,
        "text": text,
        "suppressed_upstream_stdout_chars": len(suppressed_stdout.getvalue()),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
