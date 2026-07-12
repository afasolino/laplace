"""Uvicorn factory used by the Laplace project server."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from .api import create_app


def create_project_app() -> FastAPI:
    """Create the API against the project recorded by the launcher state."""
    project_value = os.getenv("FORMALSCIENCE_ACTIVE_PROJECT")
    if not project_value:
        return create_app()
    project = Path(project_value).expanduser().resolve()
    return create_app(project, project / "Data" / "Metadata" / "workspace.db")


create_app_factory = create_project_app
