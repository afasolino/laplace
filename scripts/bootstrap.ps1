param([switch]$SkipInstall)
$ErrorActionPreference = 'Stop'
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw 'Python 3.11+ is required. Install Python or uv, then rerun.' }
& $py -m venv .venv
if (-not $SkipInstall) { & .\.venv\Scripts\python.exe -m pip install -e '.[dev]' }
& .\.venv\Scripts\python.exe -m research_workspace.cli config

