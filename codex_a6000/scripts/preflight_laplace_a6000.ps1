$ErrorActionPreference = "Stop"
$root = (& git rev-parse --show-toplevel 2>$null).Trim()
if (-not $root) { throw "Run from inside the cloned Laplace repository." }
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw ".venv\Scripts\python.exe is missing." }
& $python "codex_a6000\scripts\preflight_laplace_a6000.py"
exit $LASTEXITCODE
