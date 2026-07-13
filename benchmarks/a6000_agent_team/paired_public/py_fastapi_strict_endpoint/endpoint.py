"""Public fixture for a strict FastAPI request-boundary task."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel


class SquareRequest(BaseModel):
    value: int


app = FastAPI()


@app.post("/square")
def square(request: SquareRequest) -> dict[str, int]:
    return {"result": request.value * request.value}
