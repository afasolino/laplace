# Laplace remote access

Laplace supports three explicit modes. The recommended production design keeps Uvicorn and every vLLM endpoint on loopback.

## Local loopback

Start with `--deployment-mode local --host 127.0.0.1 --port 8765`, then open:

```text
http://127.0.0.1:8765
```

The browser cookie is intentionally non-`Secure` only in this loopback development mode, and `/api/v1/health` reports `development_http=true`.

## SSH tunnel

From the client computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 <server>
```

Keep the Operator service bound to `127.0.0.1:8765`, then open `http://127.0.0.1:8765` on the client. This is the quickest safe remote method. The remote machine's ports 8765, 8102, and 8103 need not be opened in a firewall.

## Production HTTPS reverse proxy

Use [Caddyfile.example](../deploy/caddy/Caddyfile.example) or [laplace.conf.example](../deploy/nginx/laplace.conf.example). Replace the example DNS name and paths. Do not expose any Qwen or CodeV model port, including 8102, 8103, and candidate ports 8206/8207.

The same authenticated origin serves Zetsu at `/mcp` and the Laplace Client
device/operation API under `/api/v1/client/`. Zetsu requires a normal
owner-bound bearer identity and rejects browser sessions. Client agents use an
outbound HTTPS connection with the same authorization model. Keep bearer token
files outside Git with mode 0600. If the established organization ingress
offers MFA, require it there in addition to Laplace authorization; do not add a
parallel Zetsu or Client identity system.

Example Operator arguments:

```bash
PYTHONPATH=src .venv/bin/python -m research_workspace.operator_server \
  --state-root /var/lib/laplace \
  --host 127.0.0.1 \
  --port 8765 \
  --deployment-mode reverse-proxy \
  --external-url https://laplace.example.org \
  --allowed-origin https://laplace.example.org \
  --allowed-host laplace.example.org \
  --trusted-proxy 127.0.0.1 \
  --user-registry /var/lib/laplace/auth/registered_users.yaml \
  --session-store /var/lib/laplace/auth/sessions.sqlite3 \
  --no-pwa
```

Reverse-proxy startup fails if the external URL is absent, is not HTTPS, or does not equal an explicit allowed origin. It also requires explicit allowed Host/Origin CLI flags. Forwarded headers from an address outside `--trusted-proxy` are rejected.

Remote cookies are `laplace_session; HttpOnly; Secure; SameSite=Strict; Path=/`. HSTS is emitted in reverse-proxy mode.

### Certificates

- Public DNS: Caddy can obtain/renew an ACME certificate automatically. Nginx can use a separately managed ACME client and renewal timer.
- Private LAN: use Caddy `tls internal` or a certificate from the organization's private CA; install that CA root only on authorized clients.

Never disable certificate verification for real user access. A certificate should cover the exact configured hostname and have adequate remaining validity.

### Firewall documentation

Do not apply firewall changes automatically. An administrator may allow inbound TCP 443 to the reverse proxy and optionally TCP 22 for SSH administration. Keep TCP 8765, 8102, and 8103 restricted to loopback. Public DNS/router changes require explicit operator authorization.

### Verify

Without a credential:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_remote_access.py \
  --external-url https://laplace.example.org \
  --expected-host laplace.example.org
```

This checks scheme/TLS/hostname/expiry, login-page privacy, health/readiness, invalid Host/Origin rejection, security headers, and that model ports are not reachable externally.

For the cookie and logout checks, supply the email and enter the password without echo:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_remote_access.py \
  --external-url https://laplace.example.org \
  --expected-host laplace.example.org \
  --email afasolino@unisa.it
```

`--password-stdin` is available for a protected automation pipe; never place a password on the command line or in a result bundle.

## Controlled insecure LAN development

Direct non-loopback HTTP is rejected unless `--allow-insecure-lan-http` is explicit. It also requires explicit allowed hosts/origins, registered authentication, and rate limiting. The GUI displays a persistent red warning. This is not the recommended production path.

## Streaming and shutdown

Caddy's `flush_interval -1` and Nginx's `proxy_buffering off` preserve SSE/streaming and cancellation behavior. Proxy timeouts permit bounded chat/research calls without unlimited bodies.

Folder selection and drag-and-drop run in the user's browser over the same authenticated HTTPS origin. The browser sends selected bytes and relative logical names only; the server never receives a client canonical path. Separately, Laplace Client can operate on explicitly granted PC roots through its outbound authenticated agent; see [LAPLACE_CLIENT.md](LAPLACE_CLIENT.md). Configure the proxy request-body limit to match the Operator service and per-file corpus policy while retaining bounded multipart timeouts.

Stop the reverse proxy only if it is dedicated to Laplace and the operator authorized it. Stop the Operator service before model servers. The model lifecycle unit validates ownership before signalling PIDs.
