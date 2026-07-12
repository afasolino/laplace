# Windows installation

Install Python 3.11+, ensure the locally installed Ollama service exposes only `127.0.0.1:11434`, and confirm `qwen3:4b` plus `qwen3-embedding:0.6b` are present. Then run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_laplace.ps1
laplace --doctor
mkdir .\MyProject
cd .\MyProject
laplace --init .
laplace --validate
```

The installer creates a user launcher and adds only its directory to the user PATH. For a repository-only setup, `scripts/bootstrap.ps1` and the low-level `research-workspace` command remain supported. No NPU driver or vision package is required for the baseline.
