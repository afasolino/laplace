# FormalScience operating guide

The application repository remains at `C:/Users/andre/OneDrive/Desktop/dottorato/attivita/local_research_workspace_codex`. User documents and project state remain under `C:/Users/andre/OneDrive/Desktop/dottorato/FormalScience`: shared `Library` is reference-only; each project lives under `Workspace/<ProjectName>` with `Data` and `Outputs` separated.

The supported user-facing entry point is `laplace`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_laplace.ps1
mkdir .\MyProject
cd .\MyProject
laplace --init .
laplace --doctor
laplace --ingest MyWorks --dry-run
laplace --ingest MyWorks
laplace --search "SRAM CIM quantization"
laplace --ask "Which local evidence supports this claim?"
laplace --start
```

`laplace` searches parents for `.laplace/project.yaml`, keeps the global registry in `%USERPROFILE%\\.laplace`, and refuses unsafe paths, registry collisions, ungrounded citations, and unapproved IEEE downloads. Use `laplace --project PATH ...` for an explicit project. `laplace --stop`, `--backup`, `--clean-cache --yes`, and `--status` manage lifecycle state without touching source PDFs or model/browser data.

The original low-level command remains available for reproducible scripts:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
& .\.venv\Scripts\python.exe -m research_workspace.cli project init MyProject
& .\.venv\Scripts\python.exe -m research_workspace.cli library-ingest MyProject --collection MyWorks
& .\.venv\Scripts\python.exe -m research_workspace.cli local-search "SRAM CIM quantization" --project MyProject --class user_work
& .\.venv\Scripts\python.exe -m research_workspace.cli search "SRAM compute-in-memory speculative decoding" --project MyProject
& .\.venv\Scripts\python.exe -m uvicorn research_workspace.api:create_app --factory --host 127.0.0.1 --port 8000
```

Use `project list/show/validate`, rerun `library-ingest` for incremental indexing, and inspect JSON evidence packets before drafting. `qwen3:4b` is the main local model and `qwen3-embedding:0.6b` is the embedding model.
