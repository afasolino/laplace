"""Deterministic, advisory structural context for local repositories.

This is a small native RepoMap implementation.  It deliberately keeps exact
file reads and mutation tools outside this service: the output is a bounded
hint about symbols and relationships, never an authority about file content.
Tree-sitter language packs were evaluated as an optional upstream reference,
but the baseline has no parser dependency and uses conservative standard
library extraction for the languages required by Laplace.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

JsonObject: TypeAlias = dict[str, object]
Language = Literal["python", "c", "cpp", "verilog", "systemverilog"]
SymbolKind = Literal[
    "class",
    "function",
    "method",
    "struct",
    "enum",
    "namespace",
    "macro",
    "module",
    "interface",
    "package",
]
EdgeKind = Literal["import", "include", "reference", "instantiates"]

_SUPPORTED_SUFFIXES: dict[str, Language] = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cp": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".ipp": "cpp",
    ".v": "verilog",
    ".vh": "verilog",
    ".sv": "systemverilog",
    ".svh": "systemverilog",
}
_SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HASH = re.compile(r"^[a-f0-9]{64}$")
_MAX_PATH_CHARS = 1_024
_MAX_SYMBOL_NAME = 256
_MAX_SIGNATURE = 240


class RepositoryContextError(RuntimeError):
    """A repository context request failed closed."""

    def __init__(self, category: str, evidence: JsonObject | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.evidence = evidence or {}


class RepositoryContextValidationError(RepositoryContextError):
    """The repository context request or source path was invalid."""


class RepositoryContextStaleError(RepositoryContextError):
    """A previously built context no longer matches repository bytes."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise RepositoryContextError("context_serialization_failed") from exc


def _estimated_tokens(text: str) -> int:
    """Use a conservative, deterministic character estimate in the baseline."""

    return (len(text) + 3) // 4


def _language_for_path(path: Path) -> Language | None:
    return _SUPPORTED_SUFFIXES.get(path.suffix.lower())


def _safe_relative(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RepositoryContextValidationError("path_outside_repository") from exc
    if not relative or len(relative) > _MAX_PATH_CHARS or relative.startswith("../"):
        raise RepositoryContextValidationError("invalid_repository_path")
    return relative


def _line_signature(content: str, line: int) -> str:
    lines = content.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()[:_MAX_SIGNATURE]
    return ""


@dataclass(frozen=True)
class RepositorySymbol:
    path: str
    language: Language
    name: str
    kind: SymbolKind
    line: int
    end_line: int
    signature: str
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise RepositoryContextValidationError("invalid_symbol_path")
        if self.language not in {"python", "c", "cpp", "verilog", "systemverilog"}:
            raise RepositoryContextValidationError("invalid_symbol_language")
        if not _IDENTIFIER.fullmatch(self.name) or len(self.name) > _MAX_SYMBOL_NAME:
            raise RepositoryContextValidationError("invalid_symbol_name")
        if self.kind not in {
            "class",
            "function",
            "method",
            "struct",
            "enum",
            "namespace",
            "macro",
            "module",
            "interface",
            "package",
        }:
            raise RepositoryContextValidationError("invalid_symbol_kind")
        if self.line < 1 or self.end_line < self.line:
            raise RepositoryContextValidationError("invalid_symbol_line")
        if not _HASH.fullmatch(self.source_sha256):
            raise RepositoryContextValidationError("invalid_symbol_source_hash")

    def to_json(self) -> JsonObject:
        return {
            "path": self.path,
            "language": self.language,
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class RepositoryEdge:
    source_path: str
    target_path: str | None
    kind: EdgeKind
    name: str
    line: int
    resolved: bool

    def __post_init__(self) -> None:
        if not self.source_path or Path(self.source_path).is_absolute():
            raise RepositoryContextValidationError("invalid_edge_source_path")
        if self.target_path is not None and Path(self.target_path).is_absolute():
            raise RepositoryContextValidationError("invalid_edge_target_path")
        if self.kind not in {"import", "include", "reference", "instantiates"}:
            raise RepositoryContextValidationError("invalid_edge_kind")
        if not self.name or len(self.name) > _MAX_SYMBOL_NAME:
            raise RepositoryContextValidationError("invalid_edge_name")
        if self.line < 1 or not isinstance(self.resolved, bool):
            raise RepositoryContextValidationError("invalid_edge_fields")
        if self.resolved and self.target_path is None:
            raise RepositoryContextValidationError("resolved_edge_target_missing")

    def to_json(self) -> JsonObject:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "kind": self.kind,
            "name": self.name,
            "line": self.line,
            "resolved": self.resolved,
        }


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    language: Language
    source_sha256: str
    size_bytes: int
    symbols: tuple[RepositorySymbol, ...]
    edges: tuple[RepositoryEdge, ...]

    def to_json(self) -> JsonObject:
        return {
            "path": self.path,
            "language": self.language,
            "source_sha256": self.source_sha256,
            "size_bytes": self.size_bytes,
            "symbols": [symbol.to_json() for symbol in self.symbols],
            "edges": [edge.to_json() for edge in self.edges],
        }


@dataclass(frozen=True)
class RepositoryIndex:
    repository_root: str
    snapshot_hash: str
    files: tuple[RepositoryFile, ...]
    symbols: tuple[RepositorySymbol, ...]
    edges: tuple[RepositoryEdge, ...]

    def to_json(self) -> JsonObject:
        return {
            "repository_root": self.repository_root,
            "snapshot_hash": self.snapshot_hash,
            "files": [item.to_json() for item in self.files],
            "symbols": [item.to_json() for item in self.symbols],
            "edges": [item.to_json() for item in self.edges],
        }


@dataclass(frozen=True)
class RepoMap:
    snapshot_hash: str
    text: str
    estimated_tokens: int
    token_budget: int
    entries: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _HASH.fullmatch(self.snapshot_hash):
            raise RepositoryContextValidationError("invalid_repo_map_snapshot_hash")
        if self.estimated_tokens > self.token_budget or self.token_budget <= 0:
            raise RepositoryContextValidationError("repo_map_token_budget_exceeded")

    def to_json(self) -> JsonObject:
        return {
            "snapshot_hash": self.snapshot_hash,
            "text": self.text,
            "estimated_tokens": self.estimated_tokens,
            "token_budget": self.token_budget,
            "entries": list(self.entries),
            "authority": "advisory",
            "exact_file_reads_required_for_mutation_or_verification": True,
        }


@dataclass(frozen=True)
class RepositoryContext:
    index: RepositoryIndex
    repo_map: RepoMap

    @property
    def snapshot_hash(self) -> str:
        return self.index.snapshot_hash

    @property
    def symbols(self) -> tuple[RepositorySymbol, ...]:
        return self.index.symbols

    @property
    def edges(self) -> tuple[RepositoryEdge, ...]:
        return self.index.edges

    def to_json(self) -> JsonObject:
        return {
            "index": self.index.to_json(),
            "repo_map": self.repo_map.to_json(),
            "authority": "advisory",
        }


@dataclass(frozen=True)
class _Reference:
    kind: EdgeKind
    name: str
    line: int
    target_hint: str | None = None


@dataclass(frozen=True)
class _Analysis:
    path: str
    language: Language
    content: str
    source_sha256: str
    size_bytes: int
    symbols: tuple[RepositorySymbol, ...]
    references: tuple[_Reference, ...]


def _python_analysis(path: str, content: str, digest: str, size: int) -> _Analysis:
    symbols: list[RepositorySymbol] = []
    references: list[_Reference] = []
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError:
        return _Analysis(path, "python", content, digest, size, (), ())
    defined_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind: SymbolKind = "class" if isinstance(node, ast.ClassDef) else "function"
            line = int(node.lineno)
            end_line = int(getattr(node, "end_lineno", line) or line)
            symbols.append(
                RepositorySymbol(
                    path=path,
                    language="python",
                    name=node.name,
                    kind=kind,
                    line=line,
                    end_line=end_line,
                    signature=_line_signature(content, line),
                    source_sha256=digest,
                )
            )
            defined_names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                references.append(_Reference("import", alias.name, int(node.lineno)))
        elif isinstance(node, ast.ImportFrom):
            module = ("." * node.level) + (node.module or "")
            for alias in node.names:
                references.append(_Reference("import", f"{module}:{alias.name}", int(node.lineno)))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined_names:
            references.append(_Reference("reference", node.id, int(node.lineno)))
    return _Analysis(
        path,
        "python",
        content,
        digest,
        size,
        tuple(sorted(symbols, key=lambda item: (item.line, item.name, item.kind))),
        tuple(references),
    )


_C_DECLARATION = re.compile(
    r"^\s*(?:(?:static|inline|extern|virtual|constexpr|const|unsigned|signed|long|short)\s+)*"
    r"(?:[A-Za-z_][\w:<>]*\s+|[A-Za-z_][\w:<>]*\s*[*&]\s*)+"
    r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{"
)
_C_TYPE = re.compile(r"^\s*(?P<kind>class|struct|namespace|enum)\s+(?P<name>[A-Za-z_]\w*)")
_C_MACRO = re.compile(r"^\s*#\s*define\s+(?P<name>[A-Za-z_]\w*)")
_INCLUDE = re.compile(r"^\s*#\s*include\s*[<\"](?P<name>[^>\"]+)[>\"]")


def _c_analysis(path: str, language: Language, content: str, digest: str, size: int) -> _Analysis:
    symbols: list[RepositorySymbol] = []
    references: list[_Reference] = []
    defined: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), start=1):
        type_match = _C_TYPE.match(line)
        if type_match:
            kind = cast(SymbolKind, type_match.group("kind"))
            name = type_match.group("name")
            symbols.append(
                RepositorySymbol(path, language, name, kind, line_number, line_number, line.strip()[:_MAX_SIGNATURE], digest)
            )
            defined.add(name)
        macro_match = _C_MACRO.match(line)
        if macro_match:
            name = macro_match.group("name")
            symbols.append(RepositorySymbol(path, language, name, "macro", line_number, line_number, line.strip()[:_MAX_SIGNATURE], digest))
            defined.add(name)
        function_match = _C_DECLARATION.match(line)
        if function_match:
            name = function_match.group("name")
            if name not in {"if", "for", "while", "switch", "catch"}:
                symbols.append(RepositorySymbol(path, language, name, "function", line_number, line_number, line.strip()[:_MAX_SIGNATURE], digest))
                defined.add(name)
        include_match = _INCLUDE.match(line)
        if include_match:
            references.append(_Reference("include", include_match.group("name"), line_number))
        for token in _TOKEN.findall(line):
            if token not in defined and token not in {"if", "else", "for", "while", "return", "class", "struct", "namespace", "include", "define"}:
                references.append(_Reference("reference", token, line_number))
    return _Analysis(path, language, content, digest, size, tuple(sorted(set(symbols), key=lambda item: (item.line, item.name, item.kind))), tuple(references))


_RTL_DECLARATION = re.compile(r"^\s*(?P<kind>module|interface|package)\s+(?P<name>[A-Za-z_]\w*)")
_RTL_INCLUDE = re.compile(r"^\s*`include\s*[\"](?P<name>[^\"]+)[\"]")
_RTL_INSTANCE = re.compile(r"^\s*(?P<module>[A-Za-z_]\w*)\s*(?:#\s*\([^;]*\)\s*)?(?P<instance>[A-Za-z_]\w*)\s*\(")


def _rtl_analysis(path: str, language: Language, content: str, digest: str, size: int) -> _Analysis:
    symbols: list[RepositorySymbol] = []
    references: list[_Reference] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        declaration = _RTL_DECLARATION.match(line)
        if declaration:
            kind = cast(SymbolKind, declaration.group("kind"))
            symbols.append(RepositorySymbol(path, language, declaration.group("name"), kind, line_number, line_number, line.strip()[:_MAX_SIGNATURE], digest))
        include = _RTL_INCLUDE.match(line)
        if include:
            references.append(_Reference("include", include.group("name"), line_number))
        instance = _RTL_INSTANCE.match(line)
        if instance and instance.group("module") not in {"module", "interface", "if", "for", "while", "assign"}:
            references.append(_Reference("instantiates", instance.group("module"), line_number, instance.group("instance")))
    return _Analysis(path, language, content, digest, size, tuple(sorted(symbols, key=lambda item: (item.line, item.name, item.kind))), tuple(references))


class RepositoryContextService:
    """Build and cache a content-addressed structural repository index."""

    def __init__(self, repository_root: Path, *, max_file_bytes: int = 2_000_000) -> None:
        self.repository_root = repository_root.resolve()
        if not self.repository_root.is_dir():
            raise RepositoryContextValidationError("repository_root_not_directory")
        if max_file_bytes < 1:
            raise RepositoryContextValidationError("invalid_max_file_bytes")
        self.max_file_bytes = max_file_bytes
        self._lock = threading.RLock()
        self._snapshot: tuple[tuple[str, int, str], ...] | None = None
        self._index: RepositoryIndex | None = None
        self._map_cache: dict[tuple[object, ...], RepoMap] = {}

    def _discover(self) -> tuple[tuple[str, int, str], ...]:
        records: list[tuple[str, int, str]] = []
        for directory, dirnames, filenames in os.walk(self.repository_root, followlinks=False):
            dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRECTORIES)
            for filename in sorted(filenames):
                path = Path(directory) / filename
                language = _language_for_path(path)
                if language is None or path.is_symlink():
                    continue
                relative = _safe_relative(self.repository_root, path)
                try:
                    size = path.stat().st_size
                    with path.open("rb") as handle:
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                except (OSError, ValueError) as exc:
                    raise RepositoryContextError("repository_file_unreadable", {"path": relative}) from exc
                records.append((relative, size, digest))
        return tuple(sorted(records))

    def snapshot_hash(self) -> str:
        snapshot = self._discover()
        digest = hashlib.sha256(_canonical(snapshot).encode("utf-8")).hexdigest()
        with self._lock:
            if self._snapshot != snapshot:
                self._snapshot = snapshot
                self._index = None
                self._map_cache.clear()
        return digest

    def _read_analysis(self, relative: str, size: int, digest: str) -> _Analysis:
        path = self.repository_root / relative
        language = _language_for_path(path)
        if language is None:
            raise RepositoryContextValidationError("unsupported_repository_language")
        if size > self.max_file_bytes:
            return _Analysis(relative, language, "", digest, size, (), ())
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RepositoryContextError("repository_source_unreadable", {"path": relative}) from exc
        if len(content.encode("utf-8")) > self.max_file_bytes:
            return _Analysis(relative, language, "", digest, size, (), ())
        if language == "python":
            return _python_analysis(relative, content, digest, size)
        if language in {"c", "cpp"}:
            return _c_analysis(relative, language, content, digest, size)
        return _rtl_analysis(relative, language, content, digest, size)

    @staticmethod
    def _resolve_include(root: Path, source_path: str, name: str) -> str | None:
        candidates = [root / Path(source_path).parent / name, root / name]
        for candidate in candidates:
            try:
                relative = candidate.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if candidate.is_file() and not candidate.is_symlink() and _language_for_path(candidate) is not None:
                return relative
        return None

    @staticmethod
    def _resolve_python_import(paths: set[str], source_path: str, name: str) -> str | None:
        module = name.split(":", 1)[0]
        if module.startswith("."):
            dots = len(module) - len(module.lstrip("."))
            base = Path(source_path).parent
            for _ in range(max(0, dots - 1)):
                base = base.parent
            module = module[dots:]
        module_path = module.replace(".", "/")
        candidates = [f"{module_path}.py", f"{module_path}/__init__.py"]
        return next((candidate for candidate in candidates if candidate in paths), None)

    def _build_index(self, snapshot: tuple[tuple[str, int, str], ...], snapshot_hash: str) -> RepositoryIndex:
        analyses = [self._read_analysis(*record) for record in snapshot]
        all_paths = {analysis.path for analysis in analyses}
        symbols = tuple(symbol for analysis in analyses for symbol in analysis.symbols)
        by_name: dict[str, set[str]] = defaultdict(set)
        for symbol in symbols:
            by_name[symbol.name].add(symbol.path)
        files: list[RepositoryFile] = []
        all_edges: list[RepositoryEdge] = []
        for analysis in analyses:
            edges: list[RepositoryEdge] = []
            for reference in analysis.references:
                target: str | None
                if reference.kind == "include":
                    target = self._resolve_include(self.repository_root, analysis.path, reference.name)
                elif reference.kind == "import" and analysis.language == "python":
                    target = self._resolve_python_import(all_paths, analysis.path, reference.name)
                elif reference.kind in {"instantiates", "reference"}:
                    targets = by_name.get(reference.name, set())
                    target = next(iter(targets)) if len(targets) == 1 else None
                else:
                    target = None
                resolved = target is not None
                if reference.kind == "reference" and not resolved:
                    continue
                edge = RepositoryEdge(analysis.path, target, reference.kind, reference.name, reference.line, resolved)
                edges.append(edge)
                all_edges.append(edge)
            files.append(
                RepositoryFile(
                    analysis.path,
                    analysis.language,
                    analysis.source_sha256,
                    analysis.size_bytes,
                    analysis.symbols,
                    tuple(sorted(edges, key=lambda item: (item.line, item.kind, item.name, item.target_path or ""))),
                )
            )
        return RepositoryIndex(
            str(self.repository_root),
            snapshot_hash,
            tuple(sorted(files, key=lambda item: item.path)),
            tuple(sorted(symbols, key=lambda item: (item.path, item.line, item.name, item.kind))),
            tuple(sorted(all_edges, key=lambda item: (item.source_path, item.line, item.kind, item.name, item.target_path or ""))),
        )

    def index(self) -> RepositoryIndex:
        snapshot = self._discover()
        snapshot_hash = hashlib.sha256(_canonical(snapshot).encode("utf-8")).hexdigest()
        with self._lock:
            if self._snapshot != snapshot:
                self._snapshot = snapshot
                self._index = None
                self._map_cache.clear()
            if self._index is None:
                self._index = self._build_index(snapshot, snapshot_hash)
            return self._index

    def find_symbol(self, name: str) -> tuple[RepositorySymbol, ...]:
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise RepositoryContextValidationError("invalid_symbol_query")
        return tuple(symbol for symbol in self.index().symbols if symbol.name == name)

    def find_references(self, name: str) -> tuple[RepositoryEdge, ...]:
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise RepositoryContextValidationError("invalid_reference_query")
        return tuple(edge for edge in self.index().edges if edge.name == name and edge.kind in {"reference", "instantiates"})

    @staticmethod
    def _rank_symbol(symbol: RepositorySymbol, query_terms: set[str], focus_paths: set[str], incoming: int) -> tuple[int, str, int, str]:
        name = symbol.name.casefold()
        path = symbol.path.casefold()
        score = incoming
        if symbol.path in focus_paths:
            score += 10_000
        if name in query_terms:
            score += 5_000
        score += sum(500 for term in query_terms if term in name)
        score += sum(20 for term in query_terms if term in path)
        return score, symbol.path, symbol.line, symbol.name

    def build_repo_map(
        self,
        *,
        query: str = "",
        focus_paths: Sequence[str] = (),
        token_budget: int = 1_000,
    ) -> RepoMap:
        if not isinstance(query, str) or len(query) > 4_000:
            raise RepositoryContextValidationError("invalid_repo_map_query")
        if not isinstance(focus_paths, Sequence) or isinstance(focus_paths, (str, bytes)):
            raise RepositoryContextValidationError("invalid_repo_map_focus_paths")
        if not isinstance(token_budget, int) or isinstance(token_budget, bool) or not 1 <= token_budget <= 100_000:
            raise RepositoryContextValidationError("invalid_repo_map_token_budget")
        normalized_focus: set[str] = set()
        for path in focus_paths:
            if not isinstance(path, (str, os.PathLike)):
                raise RepositoryContextValidationError("invalid_repo_map_focus_path")
            normalized_focus.add(
                _safe_relative(self.repository_root, (self.repository_root / Path(path)).resolve())
            )
        index = self.index()
        key = (index.snapshot_hash, query, tuple(sorted(normalized_focus)), token_budget)
        with self._lock:
            cached = self._map_cache.get(key)
            if cached is not None:
                return cached
        query_terms = set(_TOKEN.findall(query.casefold()))
        incoming = Counter(edge.name for edge in index.edges)
        symbols = sorted(
            index.symbols,
            key=lambda symbol: self._rank_symbol(symbol, query_terms, normalized_focus, incoming[symbol.name]),
            reverse=True,
        )
        edges_by_source: dict[str, list[RepositoryEdge]] = defaultdict(list)
        for edge in index.edges:
            edges_by_source[edge.source_path].append(edge)
        candidates: list[tuple[tuple[int, str, int, str], str]] = []
        for symbol in symbols:
            line = f"{symbol.path}:{symbol.line} {symbol.kind} {symbol.name} :: {symbol.signature}"
            candidates.append((self._rank_symbol(symbol, query_terms, normalized_focus, incoming[symbol.name]), line))
        for path, edges in sorted(edges_by_source.items()):
            for edge in sorted(edges, key=lambda item: (item.line, item.kind, item.name, item.target_path or "")):
                line = f"{edge.source_path}:{edge.line} {edge.kind} {edge.name} -> {edge.target_path or '?'}"
                candidates.append(((incoming[edge.name], edge.source_path, edge.line, edge.name), line))
        represented_paths = {symbol.path for symbol in symbols}
        for file in index.files:
            if file.path not in represented_paths:
                candidates.append(((0, file.path, 0, ""), f"{file.path} ({file.language}, no reliable symbols)"))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        header = f"# Laplace RepoMap advisory\n# snapshot {index.snapshot_hash}\n"
        lines: list[str] = [header]
        entries: list[str] = []
        used = _estimated_tokens(header)
        for _rank, candidate in candidates:
            candidate = candidate[: max(4, token_budget * 4)]
            candidate_tokens = _estimated_tokens(candidate + "\n")
            if used + candidate_tokens > token_budget:
                continue
            lines.append(candidate + "\n")
            entries.append(candidate)
            used += candidate_tokens
        text = "".join(lines)
        if _estimated_tokens(text) > token_budget:
            text = text[: token_budget * 4]
        result = RepoMap(index.snapshot_hash, text, _estimated_tokens(text), token_budget, tuple(entries))
        with self._lock:
            self._map_cache[key] = result
        return result

    def build_context(
        self,
        *,
        query: str = "",
        focus_paths: Sequence[str] = (),
        token_budget: int = 1_000,
    ) -> RepositoryContext:
        index = self.index()
        return RepositoryContext(index, self.build_repo_map(query=query, focus_paths=focus_paths, token_budget=token_budget))

    def assert_fresh(self, context: RepositoryContext | RepoMap) -> None:
        expected = context.snapshot_hash
        current = self.snapshot_hash()
        if current != expected:
            raise RepositoryContextStaleError(
                "repository_context_stale",
                {"expected_snapshot_hash": expected, "current_snapshot_hash": current},
            )
