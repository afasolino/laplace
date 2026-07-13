"""Public fixture for strict request validation."""

from pydantic import BaseModel


class PolicyRequest(BaseModel):
    retries: int
