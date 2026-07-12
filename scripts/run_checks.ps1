$ErrorActionPreference = 'Stop'
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check src tests
& .\.venv\Scripts\python.exe -m mypy src
& .\.venv\Scripts\python.exe -m research_workspace.cli config
& .\.venv\Scripts\python.exe -m research_workspace.cli probe

