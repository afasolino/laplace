from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Sequence

from .retrieval import embed


@dataclass(frozen=True)
class AuxiliaryStatus:
    provider: str
    status: str
    reason: str


def npu_status() -> AuxiliaryStatus:
    if importlib.util.find_spec("onnxruntime") is None:
        return AuxiliaryStatus(
            "onnxruntime", "NPU_OPTIONAL_NOT_BENEFICIAL", "onnxruntime is not installed"
        )
    return AuxiliaryStatus(
        "onnxruntime",
        "AVAILABLE_UNVALIDATED",
        "provider compatibility requires an explicit model benchmark",
    )


def cpu_embedding(texts: Sequence[str]) -> list[list[float]]:
    """Deterministic auxiliary baseline; never claims NPU acceleration."""
    return [embed(text) for text in texts]


def vision_status() -> dict[str, str]:
    return {
        "status": "OPTIONAL_NOT_INSTALLED",
        "policy": "native extraction remains primary; load one vision model on demand",
    }
