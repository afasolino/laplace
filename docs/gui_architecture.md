# GUI architecture

The Operator GUI is a dependency-free responsive PWA shell over the versioned
FastAPI `/api/v1` surface. Vanilla HTML/CSS/JavaScript keeps the runtime small;
filesystem artifacts remain authoritative and SQLite holds only projections,
approvals, and operator events.

The ten views are dashboard, run builder, live run, evidence explorer, research
studio, corpus manager, model/hardware center, comparison, approval queue, and
diagnostics. The layout becomes a bottom scrollable navigation rail below
760 px and avoids horizontal overflow at 390 px.

Browser actions call `OperatorService`, `DeepResearchService`, or
`ModelServerController`. The event view consumes server-sent events. Downloads
are authenticated fetches, not public static links. Research reports render
from ledger/report API data rather than executing artifact HTML.

Start locally with:

```bash
laplace-operator-server --host 127.0.0.1 --port 8765
```

On first start, four role tokens are saved mode 0600; the admin token is shown
once. The PWA caches only the application shell and never caches `/api/`.

