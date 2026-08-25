"""Compatibility names for the neutral durable result store."""

from .result_store import ResultStore, ResultStoreError

ZetsuResultStore = ResultStore
ZetsuResultError = ResultStoreError

__all__ = ["ZetsuResultError", "ZetsuResultStore"]
