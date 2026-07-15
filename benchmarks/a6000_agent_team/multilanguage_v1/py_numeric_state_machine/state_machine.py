from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundedAccumulator:
    lower: float
    upper: float
    value: float = 0.0

    def apply(self, delta: float) -> float:
        if self.lower > self.upper:
            raise ValueError("lower bound exceeds upper bound")
        self.value = min(self.upper, max(self.lower, self.value + delta))
        return self.value
