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
  --role admin \
  --default-lane quality
```

The one-time activation code is printed once. It is not written to the registry or audit log. Do not paste it into a guide, shell history argument, screenshot, or issue.

## 3. Start the local Operator service

Start the selected local model endpoints separately using the validated serving-profile procedure. Then:

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.operator_server \
  --state-root "$HOME/.local/state/laplace" \
  --deployment-mode local \
  --host 127.0.0.1 \
  --port 8765
```

Open `http://127.0.0.1:8765`, select **First activation**, enter the printed code, and create the password.

## 4. Verify

```bash
curl --fail http://127.0.0.1:8765/api/v1/health
curl --fail http://127.0.0.1:8765/api/v1/readiness
```

Health proves process liveness. Readiness also checks the registered-user registry, session store, external state directories, and lane routing.

For remote use, continue with [REMOTE_ACCESS.md](REMOTE_ACCESS.md). For user and lifecycle administration, read [ADMIN_GUIDE.md](ADMIN_GUIDE.md).
