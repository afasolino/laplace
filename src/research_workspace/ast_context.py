"""Bounded structural code context backed by Aider's ``grep-ast``.

Laplace owns path authorization and output budgets.  Tree parsing, language
coverage and scope expansion are delegated to the mature upstream library.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_PATTERN_CHARS = 1_000
_DEFAULT_MAX_CHARS = 12_000
_MAX_OUTPUT_CHARS = 24_000


class AstContextError(RuntimeError):
    """A bounded AST-context request was invalid or unavailable."""


def _repository_file(repository_root: Path, relative_path: str) -> Path:
    root = repository_root.expanduser().resolve()
    if not relative_path or len(relative_path) > 1_000:
        raise AstContextError("ast_context_invalid_path")
    supplied = Path(relative_path)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise AstContextError("ast_context_path_escape")
    candidate = root / supplied
    if candidate.is_symlink():
        raise AstContextError("ast_context_symlink_refused")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AstContextError("ast_context_file_unavailable") from exc
    if resolved != root and root not in resolved.parents:
        raise AstContextError("ast_context_path_escape")
    if not resolved.is_file():
        raise AstContextError("ast_context_file_unavailable")
    return resolved


def _tree_context_factory() -> Callable[..., Any]:
    try:
        from grep_ast.grep_ast import TreeContext  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised by dependency gate
        raise AstContextError(
            "ast_context_dependency_missing:install_pip_editable_with_v3_extra"
        ) from exc
    return cast(Callable[..., Any], TreeContext)


def render_ast_context(
    repository_root: Path,
    relative_path: str,
    pattern: str,
    *,
    ignore_case: bool = False,
    max_chars: int = _DEFAULT_MAX_CHARS,
    tree_context_factory: Callable[..., Any] | None = None,
) -> JsonObject:
    """Render scope-aware context for matching lines in one repository file."""

    if not pattern or len(pattern) > _MAX_PATTERN_CHARS:
        raise AstContextError("ast_context_invalid_pattern")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise AstContextError("ast_context_invalid_regex") from exc
    if not 512 <= max_chars <= _MAX_OUTPUT_CHARS:
        raise AstContextError("ast_context_invalid_max_chars")

    path = _repository_file(repository_root, relative_path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AstContextError("ast_context_file_unavailable") from exc
    if size > _MAX_FILE_BYTES:
        raise AstContextError("ast_context_file_too_large")
    try:
        code = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AstContextError("ast_context_file_not_utf8") from exc

    factory = tree_context_factory or _tree_context_factory()
    try:
        context = factory(
            str(path),
            code,
            color=False,
            verbose=False,
            line_number=True,
            parent_context=True,
            child_context=True,
            last_line=True,
            margin=2,
            mark_lois=True,
            header_max=10,
            show_top_of_file_parent_scope=True,
            loi_pad=1,
        )
        matches = context.grep(pattern, ignore_case)
        context.add_lines_of_interest(matches)
        context.add_context()
        rendered = str(context.format())
    except (ValueError, re.error) as exc:
        raise AstContextError(f"ast_context_upstream_error:{type(exc).__name__}") from exc

    truncated = len(rendered) > max_chars
    if truncated:
        rendered = rendered[: max_chars - 1].rstrip() + "…"
    root = repository_root.expanduser().resolve()
    return {
        "status": "OK",
        "provider": "grep-ast",
        "path": str(path.relative_to(root)),
        "pattern": pattern,
        "match_count": len(matches),
        "context": rendered,
        "truncated": truncated,
        "max_chars": max_chars,
    }


def tool_definition() -> JsonObject:
    """MCP schema for the local read-only structural-context tool."""

    return {
        "name": "ast_context",
        "description": (
            "Read bounded tree-sitter structural context around regex matches in one "
            "authorized repository file using grep-ast."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 1_000},
                "pattern": {"type": "string", "minLength": 1, "maxLength": 1_000},
                "ignore_case": {"type": "boolean"},
                "max_chars": {"type": "integer", "minimum": 512, "maximum": 24_000},
            },
            "required": ["path", "pattern"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="laplace-ast-context")
    parser.add_argument("pattern")
    parser.add_argument("path")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--ignore-case", action="store_true")
    parser.add_argument("--max-chars", type=int, default=_DEFAULT_MAX_CHARS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        value = render_ast_context(
            args.repo,
            args.path,
            args.pattern,
            ignore_case=args.ignore_case,
            max_chars=args.max_chars,
        )
    except AstContextError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
