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
- **Chat does not open:** run `laplace --start --no-browser`, verify `http://127.0.0.1:8000/chat`, and inspect the project `Data/Logs/laplace-server.log`; `laplace --stop` only stops the recorded project PID.
- **Chat says fallback:** this is an intentional citation-safety result. Inspect `/api/chat/messages/{message_id}/audit`; the model response was retained, but the visible answer uses exact retrieved evidence because its citation IDs were invalid or empty.
- **A draft appears above a fallback:** this is expected. Expand `Unverified model draft · citation validation failed` to inspect the immutable rejected candidate; the separate green grounded fallback is the answer used for citations. In the terminal use `laplace --ask "..." --show-rejected-draft`.
- **Repeated/late stream events:** the UI keys events by `message_id`, `revision_id`, and `sequence`; refresh the conversation if a browser extension buffers SSE. The server never replaces a stored message in place.
- **Conversation history missing:** confirm the active project and inspect `Data/Metadata/laplace.db`; conversations are never stored in the application repository.
- **Generation remains active:** press Stop. Cancellation sets only the current Laplace generation event and does not terminate Ollama or unrelated processes.
