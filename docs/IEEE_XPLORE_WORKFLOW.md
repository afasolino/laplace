# IEEE Xplore workflow

Set a key only in the process environment; never put it in `.env` tracked files:

```powershell
$env:IEEE_XPLORE_API_KEY = '<value supplied through your approved secret mechanism>'
& .\.venv\Scripts\python.exe -m research_workspace.cli search-ieee "SRAM compute-in-memory" --limit 5
```

Without a key the command returns `API_KEY_REQUIRED`. Discovery uses the official Metadata API, not search-page scraping. Initialize the isolated profile with `ieee browser-init`, then run `ieee login`; if Playwright is installed it reports `MANUAL_LOGIN_REQUIRED`, otherwise `PLAYWRIGHT_REQUIRED`. The profile is `C:/Users/andre/AppData/Local/FormalScienceBrowser`, outside OneDrive and Git.

For a queued candidate: `ieee open <project> <candidate-id>`, confirm the title/DOI/article number in the visible browser, then `ieee approve <project> <candidate-id>`. Only after explicit approval may `ieee download <project> <candidate-id>` proceed; the default batch is one. Files first land in `Data/Downloads/IEEE/Pending`, and only validated downloads can move to `Downloaded`.

