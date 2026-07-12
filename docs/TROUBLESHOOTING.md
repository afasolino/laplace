# Troubleshooting

- **Missing model:** verify `ollama list`, `qwen3:4b`, and `qwen3-embedding:0.6b`; no cloud fallback exists.
- **IEEE key missing:** `search-ieee` returns `API_KEY_REQUIRED`; set `IEEE_XPLORE_API_KEY` temporarily in the shell.
- **403/429/CAPTCHA/session expiry:** stop the browser workflow, preserve the audit record, and retry only after manual resolution.
- **Invalid PDF or duplicate:** the downloader rejects MIME/magic/hash conflicts; inspect `Data/Downloads` and the ingestion report.
- **Parser failure:** inspect `Data/Quarantine`; native PDF extraction does not silently claim OCR success.
- **GPU memory pressure:** use `RW_CONTEXT_TOKENS=4096`, keep concurrency one, and unload optional models before the text model.
- **Browser profile lock:** close the visible browser and retry; never point to a normal browser profile.
- **OneDrive issues:** keep the application separate from FormalScience and keep browser state at `C:/Users/andre/AppData/Local/FormalScienceBrowser`.
- **Offline mode:** `research-workspace search ... --offline` returns no network results while local retrieval remains available.

