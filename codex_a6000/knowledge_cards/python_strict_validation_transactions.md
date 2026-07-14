# Python strict-validation and transaction invariant card

This project-authored card records recurring implementation invariants for bounded benchmark repairs.

## Pydantic v2 strict request models

Use the v2 configuration API and make strictness explicit at model level:

```python
from pydantic import BaseModel, ConfigDict

class Request(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    value: int
```

The required negative cases are string-to-integer coercion, floating-point coercion, and undeclared fields. Avoid reparsing an instance that FastAPI or Pydantic has already validated.

## SQLite idempotence and conflict preservation

A state-recording transaction has three distinct outcomes:

1. absent key: insert and commit, returning `True`;
2. same existing value: leave unchanged, returning `False`;
3. conflicting value: raise the specified conflict exception and preserve the original row.

Do not catch the conflict exception with a broad `except Exception` handler that converts it to another error. Roll back database failures, then re-raise without overwriting the intended domain exception. Commit only after a successful insert or other explicitly committed transition.
