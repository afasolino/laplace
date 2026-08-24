# Laplace quick start

## 1. Install locally

Python 3.11 or newer is required. No model is downloaded automatically.

```bash
uv venv
uv pip install --python .venv/bin/python -e '.[dev,browser]'
```

Pip fallback:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev,browser]'
```

Install Playwright Chromium only when browser tests or screenshots are required:

```bash
.venv/bin/playwright install chromium
```

## 2. Create the external first account

Choose an external state directory; it must not be inside the Git checkout.

```bash
install -d -m 0700 "$HOME/.local/state/laplace/auth"
PYTHONPATH=src .venv/bin/python -m research_workspace.user_admin bootstrap \
  --registry "$HOME/.local/state/laplace/auth/registered_users.yaml" \
  --email afasolino@unisa.it \
  --user-id usr_afasolino \
  --display-name "Alfonso Fasolino" \
  --capability-tier operator \
  --capability chat \
  --capability agent \
  --capability research \
  --capability operator \
  --capability admin \
  --capability personal_corpus \
  --capability shared_corpus_ingest \
  --capability repository_admin \
  --capability model_admin \
  --role admin \
  --default-lane quality
```

The one-time activation code is printed once. It is not written to the registry or audit log. Do not paste it into a guide, shell history argument, screenshot, or issue.

Add the normal Chat/Agent/personal-corpus account separately:

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.user_admin add \
  --registry "$HOME/.local/state/laplace/auth/registered_users.yaml" \
  --email frcapone@unisa.it \
  --user-id usr_frcapone \
  --display-name "Francesco R. Capone" \
  --capability-tier plus \
  --capability chat \
  --capability agent \
  --capability personal_corpus \
  --role user \
  --default-lane standard
```

Give that activation code only to the named user. Repository access remains empty until an administrator registers and grants a logical repository.

## 3. Validate the selected operating mode

Desktop mode uses a user-owned project, a single-user state root and a configured
local provider. Server mode adds registered identities, capabilities, logical
repositories, isolated worktrees, governed corpora, queues and audit. Ports and
providers come from the selected deployment configuration; a frontend port is never
a provider identity.

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.laplace_cli \
  --validate-config configs/laplace.example.yaml \
  --configuration-mode desktop \
  --diagnostic-export /tmp/laplace-config-diagnostic.json
```

This command does not contact, start or stop a model provider.

## 4. Start the local Operator service

Start the selected local model endpoints separately using the validated serving-profile procedure. Then:

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.operator_server \
  --state-root "$HOME/.local/state/laplace" \
  --deployment-mode local \
  --host 127.0.0.1 \
  --port 8765
```

Open `http://127.0.0.1:8765`, select **First activation**, enter the printed code, and create the password.

## 5. Verify

```bash
curl --fail http://127.0.0.1:8765/api/v1/health
curl --fail http://127.0.0.1:8765/api/v1/readiness
```

Health proves process liveness. Readiness also checks the registered-user registry, session store, external state directories, and lane routing.

For remote use, continue with [REMOTE_ACCESS.md](REMOTE_ACCESS.md). For user and lifecycle administration, read [ADMIN_GUIDE.md](ADMIN_GUIDE.md).

For personal-folder ingestion and isolated Agent work, continue with [PERSONAL_CORPUS.md](PERSONAL_CORPUS.md) and [AGENT_WORKTREES.md](AGENT_WORKTREES.md).

For release review without a GPU, run:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/run_architecture_release_v7_certification.py
```

That command uses only fixtures and temporary state. It never contacts or manages a
model provider; GPU/live-model certification remains
`BLOCKED_BY_USER_CONSTRAINT_GPU_UNAVAILABLE`.

## v8 candidate verification

Run the non-GPU release candidate from a clean dedicated v8 branch:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_release_candidate_v8_certification.py \
  --output-root outputs
```

This command uses fixtures and isolated state. It does not start a model server.
Run the live-GPU command only after the CPU report says
`GO_FOR_CONTROLLED_LIVE_GPU_CERTIFICATION`.

## Zetsu from a Codex project

From the repository that Codex will work on, export the existing owner-bound
bearer credential and install/refresh the managed integration:

```bash
export LAPLACE_ZETSU_TOKEN='<secret value>'
laplace zetsu configure --endpoint https://laplace.example.org/mcp --json
laplace zetsu status --json
laplace zetsu test --json
```

The command preserves unrelated `$CODEX_HOME/config.toml` content, registers the
user-level MCP endpoint, and installs the repository-local Zetsu Skill. Use Codex
locally for ordinary checkout/shell/Git work; use Zetsu for compact retrieval,
bounded Qwen `delegate`/`agent_task`, CodeV `rtl_task`, or deterministic evidence.
See [ZETSU.md](ZETSU.md).
