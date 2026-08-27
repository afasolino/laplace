# Laplace v3 human regression

Run from `/home/giando/work/laplace-v2` on `feature/laplace-v3`. Keep `/home/giando/work/laplace` untouched.

## 0. Deterministic gate

```bash
source .venv/bin/activate
unset PYTHONPATH
python -m pip install -e '.[dev,v3]'
python -m pytest -q tests/test_chat_v3_consolidation.py tests/test_ast_context_v3.py tests/test_laplace_web_v3.py tests/test_zetsu_codex_v3.py tests/test_zetsu_sdk_stdio_v3.py
python -m mypy src/research_workspace
python -m ruff check src/research_workspace/chat_cli.py src/research_workspace/chat_input.py src/research_workspace/chat_discovery.py src/research_workspace/chat_verification.py src/research_workspace/ast_context.py src/research_workspace/laplace_web.py src/research_workspace/zetsu_sdk_stdio.py src/research_workspace/zetsu_codex.py tests/test_chat_v3_consolidation.py tests/test_ast_context_v3.py tests/test_laplace_web_v3.py tests/test_zetsu_codex_v3.py tests/test_zetsu_sdk_stdio_v3.py
git diff --check
```

## 1. User-friendly terminal

Start the normal certified runtime, then:

```bash
laplace chat --repo-id laplace-v2 --access read
```

Verify manually:
1. `/help`, `/skills`, `/capabilities`, `/verification`, `/frontends` render without a model turn.
2. Paste a three-paragraph instruction: it must remain one prompt/one turn.
3. Alt+Enter inserts a newline; Enter submits.
4. Type a prefix of a slash command and trigger completion; fuzzy completion must work.
5. Submit a command used earlier, start typing it again, and accept the history suggestion.
6. Exit and reopen: command/input history remains available.

Then write safety:

```bash
laplace chat --repo-id laplace-v2 --access write
```

Expected: immediate `write_access_requires_verification`.

Start with a verifier and perform one tiny test-only edit. Confirm edit -> verification failure if deliberately broken -> repair -> verification pass -> completion still works.

## 2. grep-ast structural context

```bash
laplace ast-context 'derive_task_label' src/research_workspace/task_labels.py --repo "$PWD"
```

Expected:
- `provider` is `grep-ast`;
- path is repository-relative;
- output includes enclosing structural context, not only matching lines.

Negative checks:

```bash
laplace ast-context x ../outside.py --repo "$PWD" ; test $? -eq 2
```

and create no symlinks/artifacts outside the repository.

## 3. Gradio web UI

With the Operator running and its normal token available:

```bash
laplace web --repo-id laplace-v2 --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

Verify:
1. page shows Mode, Access, Model lane, verifier, chat, session resume and cancel controls;
2. default access is read-only;
3. Capabilities matches terminal `/capabilities`;
4. plain Chat produces a response through the resident Operator;
5. Agent/read performs a read-only repository task;
6. Agent/write with an empty verifier is rejected;
7. Agent/write with `pytest tests/test_task_labels.py` can perform a tiny test-only mutation and returns the verified result;
8. New session clears browser history; Resume `last` restores persisted history;
9. Cancel targets the remote agent session;
10. `--host 0.0.0.0` is refused. No Gradio `share=True` tunnel is ever enabled.

## 4. Official SDK Zetsu-Codex

Ensure the local Zetsu/Operator runtime is healthy first:

```bash
laplace zetsu status --repo "$PWD"
laplace zetsu test --repo "$PWD"
```

Install the new stdio transport using Codex's own configuration CLI:

```bash
laplace codex install --repo "$PWD" --replace
laplace codex status
codex mcp get zetsu --json
```

Expected transport: stdio, command `laplace-zetsu-mcp`, with repository/state-root/endpoint arguments and **no bearer token stored in Codex config**.

Launch:

```bash
laplace codex launch
```

Inside Codex:
1. `/mcp` shows `zetsu` connected;
2. tool list includes normal Zetsu tools plus `ast_context` when repository-agent capability is authorized;
3. ask Codex: `Use ast_context to show where derive_task_label is defined. Do not modify anything.` It must call `ast_context` and return bounded structural context;
4. ask for a Zetsu `search`/retrieval call and confirm it goes through the existing authenticated backend;
5. ask for a read-only `agent_task`; confirm repository identity/worktree evidence is preserved;
6. run one mutation with explicit verifier through `agent_task`; confirm latest mutation verification remains mandatory;
7. use an unauthorized repo or invalid token and confirm MCP startup/tool discovery fails closed.

## 5. Optional Hermes interoperability

Configure Hermes Agent to use the same stdio command `laplace-zetsu-mcp --repo <repo> --state-root <state> --endpoint <endpoint>`. Reload MCP and verify the same tool list. Do not install/copy Hermes runtime into Laplace.

## 6. Regression of existing invariants

Re-run the existing focused suite used for v2 certification:

```bash
python -m pytest -q \
  tests/test_chat_v3_consolidation.py \
  tests/test_chat_cli_v2.py \
  tests/test_chat_operator_client_v2.py \
  tests/test_chat_async_agent_turns.py \
  tests/test_operator_agent_conversation.py \
  tests/test_agent_verification_policy.py \
  tests/test_zetsu_agent_checkpoint.py \
  tests/test_repository_lifecycle_fix.py \
  tests/test_zetsu_agent_mcp.py
```

Finally:

```bash
git status --short
git diff --check
git diff --stat
git -C /home/giando/work/laplace status --short
```

The production checkout must remain unchanged.
