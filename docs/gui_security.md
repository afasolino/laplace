# GUI security

The default bind is `127.0.0.1`. API access always uses bearer authentication.
Roles are read, operate, approve, and admin. State-changing requests also
require a credential-bound in-memory CSRF nonce and an allowed Origin.

Security headers deny external scripts/styles, objects, framing, base-URI
changes, camera, microphone, and geolocation. API responses use `no-store`.
Tokens stay in browser `sessionStorage`; the server token file is mode 0600 and
excluded from artifacts and bundles.

Artifact paths must be relative, remain inside the operator state root or
repository `outputs`, use an allowlisted extension, and remain below the size
limit. Non-admin access rejects held-out, prompt, secret, and credential path
components. There is no arbitrary shell endpoint and no result-edit endpoint.

Dynamic frontend content is inserted through `textContent`. The generated
research HTML escapes report and ledger text. Measured configuration becomes
immutable after preparation/start, and frontend failure cannot change
execution semantics.

