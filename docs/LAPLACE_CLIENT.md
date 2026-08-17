# Laplace Client

Laplace Client is an outbound agent for an authenticated user's PC. The server
never opens a port on the PC. The user explicitly grants canonical local roots;
no root is available by default.

## Pair and grant

Store an owner-bound Laplace bearer credential in a mode-0600 environment file
outside Git, then:

```bash
export LAPLACE_CLIENT_TOKEN='<secret value>'
laplace-client grant /absolute/project/root --allow python3 --allow git --allow verilator
laplace-client pair --endpoint https://laplace.example.org --name engineering-laptop
laplace-client status
laplace-client serve
```

`grant` prints a random workspace ID. Use `--read-only` when modifications and
commands are unnecessary. `revoke WORKSPACE_ID` removes a local grant. `unpair`
revokes the owner/device record remotely and removes only local connection
state. Credentials are read from `LAPLACE_CLIENT_TOKEN` and are never written
to client state.

The device reports available tools dynamically. Supported operations are
list/read/search, controlled writes, Git inspection, approved commands,
cancellation, and result retrieval. Commands execute on the client, never the
server. The server queue and device records are persistent and owner/device
scoped; reconnects update the same device only for the same endpoint and token
environment.

## Isolation

Logical paths reject traversal and symlinks. Reads and writes use no-follow
descriptor walks and atomic replacement, preventing symlink races and escape
from the granted root. On Linux, commands run only when bubblewrap is available.
The sandbox exposes `/usr` read-only, a private `/tmp`, a minimal `/dev` and
`/proc`, and exactly the granted root, read-write or read-only according to the
grant. It unshares networking. Executables must resolve inside `/usr` or the
granted root; tools elsewhere are reported unavailable. Command execution fails
closed on unsupported platforms.

Validate a Linux client before enrollment:

```bash
PYTHONPATH=src .venv/bin/python scripts/certify_laplace_client_sandbox.py
```

This exercises Python, Git, Verilator RTL lint when installed, cancellation,
read-only enforcement, and denial of an out-of-root file read.
For an authenticated staging endpoint, run
`scripts/certify_laplace_client_transport.py --endpoint https://…`; it verifies
pairing, reconnect, queued execution on the client, result retrieval,
cancellation, and revoke.

For persistence, install `deploy/systemd/laplace-client.service.example` as a
user unit, put the token in `%h/.config/laplace/client.env` with mode 0600, then
run `systemctl --user enable --now laplace-client`.
