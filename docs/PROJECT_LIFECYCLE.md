# Project lifecycle

`laplace --init NAME` creates a portable project in the current directory (or `laplace --init .` initializes the current directory) with `.laplace/project.yaml`, `.laplace/state.json`, the `Config`, `Data/{Parsed,Metadata,VectorStore,Cache,Logs,Quarantine,Downloads}`, and `Outputs/{Drafts,EvidencePackets,Extractions,Comparisons,Reports}` tree. It registers the project in `%USERPROFILE%\\.laplace\\projects.json`, refuses path traversal, Library placement, application-repository placement, and name/path collisions, and never overwrites a non-empty directory without `--force`.

`laplace --validate`, `--status`, `--config`, `--backup`, `--clean-cache --yes`, `--start`, and `--stop` operate on the auto-detected project. The original `research-workspace project init <name>` remains available for the fixed FormalScience/Workspace layout and is intentionally not replaced.

Shared Library PDFs are never copied or modified. `laplace --ingest` stores parsed chunks, metadata, hashes, and source paths in the selected project. Downloads remain project-local until `laplace --promote DOCUMENT_ID DESTINATION --force` is explicitly confirmed; promotion writes an audit record and refuses overwrites.
