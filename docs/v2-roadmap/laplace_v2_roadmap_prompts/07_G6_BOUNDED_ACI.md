# G6 — Improved Bounded Agent-Computer Interface

Improve model-facing repository tools without relaxing the sandbox.

## Execute

1. Inspect SWE-agent ACI and mini-SWE-agent for tool-shape ideas; record commits/licenses.
2. Prefer a small typed set: `repo_map`, `find_symbol`, `find_references`, `search_text`, `read_region`, `inspect_diff`, `edit_region`, `create_text_file`, `verify`, `git_state`.
3. Keep model-controlled generic shell/network unavailable.
4. Preserve worktree/path/owner/verifier binding, cancellation and budgets.
5. If v1 traces show real output truncation or awkward large-file construction, A/B a bounded chunk/append/edit primitive. Do not enlarge generation limits or add chunked writes without evidence.
6. Keep outputs bounded and structured.

## Gate

Cover traversal, symlink escape, `.git`, malformed input, huge output, timeout, cancellation, races/concurrent sessions and verifier misuse.

Compare frozen coding/repair tasks with old versus new ACI. Enable only with preserved correctness/security and measurable efficiency benefit.

Certify and commit before G7.
