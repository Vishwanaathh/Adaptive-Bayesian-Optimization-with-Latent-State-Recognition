from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class Dimension:
    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self):
        if self.kind not in {"continuous", "log", "integer", "categorical"}:
            raise ValueError(f"Unsupported dimension type: {self.kind}")
        if self.kind == "categorical":
            if len(self.choices) < 2:
                raise ValueError("Categorical dimensions require at least two choices")
        elif self.low is None or self.high is None or self.high <= self.low:
            raise ValueError(f"Invalid bounds for {self.name}")

class SearchSpace:
    def __init__(self, dimensions: list[Dimension]):
        if not dimensions:
            raise ValueError("Search space cannot be empty")
        names = [d.name for d in dimensions]
        if len(set(names)) != len(names):
            raise ValueError("Dimension names must be unique")
        self.dimensions = dimensions
        self.dim = len(dimensions)

    @classmethod
    def from_config(cls, items):
        dims = []
        for x in items:
            dims.append(Dimension(
                name=x["name"], kind=x["type"],
                low=x.get("low"), high=x.get("high"),
                choices=tuple(x.get("choices", ()))
            ))
        return cls(dims)

    def decode(self, unit_x: np.ndarray) -> dict[str, Any]:
        x = np.asarray(unit_x, dtype=float).reshape(-1)
        if len(x) != self.dim:
            raise ValueError("Point dimensionality mismatch")
        if np.any((x < 0) | (x > 1)):
            raise ValueError("Normalized coordinates must lie in [0, 1]")
        result = {}
        for i, d in enumerate(self.dimensions):
            u = float(x[i])
            if d.kind == "categorical":
                idx = min(int(u * len(d.choices)), len(d.choices) - 1)
                result[d.name] = d.choices[idx]
            elif d.kind == "log":
                result[d.name] = float(np.exp(np.log(d.low) + u * (np.log(d.high)-np.log(d.low))))
            elif d.kind == "integer":
                result[d.name] = int(np.clip(round(d.low + u*(d.high-d.low)), d.low, d.high))
            else:
                result[d.name] = float(d.low + u*(d.high-d.low))
        return result

    def encode(self, config: dict[str, Any]) -> np.ndarray:
        out = []
        for d in self.dimensions:
            v = config[d.name]
            if d.kind == "categorical":
                out.append((d.choices.index(v) + 0.5) / len(d.choices))
            elif d.kind == "log":
                out.append((np.log(float(v))-np.log(d.low))/(np.log(d.high)-np.log(d.low)))
            else:
                out.append((float(v)-d.low)/(d.high-d.low))
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.random((n, self.dim))
