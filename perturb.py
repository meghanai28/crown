from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
import mlx.core as mx
import mlx.nn as nn

from model import BaseModel

@dataclass(frozen=True)
class PerturbationConfig:
    layer_index: int = 14
    rank: int = 4
    scale: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("layer_index cannot be negative")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")

