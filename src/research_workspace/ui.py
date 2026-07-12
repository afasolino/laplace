from __future__ import annotations

from fastapi.responses import HTMLResponse


def offline_dashboard() -> HTMLResponse:
    return HTMLResponse("""<!doctype html><html><head><meta charset='utf-8'><title>Local Research Workspace</title></head>
    <body><h1>Local Research Workspace</h1><nav>System status · Collections · Grounded search · Extraction · Log analysis · Drafting · Benchmarks</nav>
    <p>Loopback-only interface. Use <code>/docs</code> for the local API.</p></body></html>""")
