# C5 — Strengthen Repository Intelligence

## Goal

Eliminate ambiguity between "production-grade structural repository intelligence" and "lightweight advisory regex map".

## Required decision

Inspect the existing parser architecture, dependencies, supported languages, performance constraints, and current tests.

For Python, retain AST-based parsing if correct.

For C, C++, Verilog, and SystemVerilog choose one of these two paths:

### Path A — integrate tree-sitter
Use a maintained tree-sitter package already compatible with Python 3.11 and the repository environment. Prefer `tree-sitter-language-pack` if it satisfies the required grammars and licensing.

Use tree-sitter for structural extraction of symbols/references/dependencies. Keep a conservative fallback only when parsing is unavailable or fails.

### Path B — explicitly reduce the contract
Only if tree-sitter integration is demonstrably unsuitable, change the public contract/documentation so the existing implementation is explicitly a lightweight advisory RepoMap and is never represented as complete semantic parsing.

Path B requires a written evidence artifact explaining the rejection and known unsupported constructs.

Do not keep the current ambiguous middle state.

## Required parser tests

For C/C++:
- multiline declarations;
- namespaces/classes/templates;
- qualified names;
- macros/preprocessor noise;
- includes.

For Verilog/SystemVerilog:
- parameterized modules;
- multiline ports;
- module instantiation;
- generate blocks;
- packages/imports;
- interfaces/modports where grammar supports them;
- preprocessing noise.

Repository intelligence remains advisory; exact source remains authoritative.

## Cache/invalidation

Ensure cached results do not require unnecessary full reparsing of unchanged files. File hash/stat optimization may be used, but correctness must be hash-backed before reuse.

Add stale-index and changed-file tests.
