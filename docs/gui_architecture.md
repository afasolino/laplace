# GUI architecture

The Operator GUI is a dependency-free responsive PWA shell over the versioned
FastAPI `/api/v1` surface. Vanilla HTML/CSS/JavaScript keeps the runtime small;
filesystem artifacts remain authoritative and SQLite holds only projections,
approvals, and operator events.

The ten views are dashboard, run builder, live run, evidence explorer, research
studio, corpus manager, model/hardware center, comparison, approval queue, and
diagnostics. The layout becomes a bottom scrollable navigation rail below
760 px and avoids horizontal overflow at 390 px.

Browser actions call the authenticated versioned Operator API. The Chat view
uses `/api/v1/chat`; the Agent view creates one `/api/v1/agent/sessions` record
and sends later natural-language turns to that same session's `/messages` route.
Its transcript is stored server-side with owner/repository binding, not in
browser storage. Status, diff, tests, and context inspect deterministic status
or paged-result data and do not poll a model. A user must explicitly select an
edit-capable turn, give a verifier argv, and confirm it before the Agent can
mutate its isolated worktree.

The event view consumes server-sent events. Downloads are authenticated fetches,
not public static links. Research reports render from ledger/report API data
rather than executing artifact HTML. The PWA stores no bearer token, CSRF token,
or terminal/agent session metadata in browser storage.

Start locally with:

```bash
laplace-operator-server --host 127.0.0.1 --port 8765
```

On first start, four role tokens are saved mode 0600; the admin token is shown
once. The PWA caches only the application shell and never caches `/api/`.
