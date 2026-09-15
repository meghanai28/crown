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
        if self.scale < 0:
            raise ValueError("scale cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")


class RandomLowRankLinear(nn.Module):
    """A frozen base layer plus a small random low-rank update."""

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        scale: float,
        seed: int,
    ) -> None:
        super().__init__()

        if isinstance(base_layer, nn.QuantizedLinear):
            output_dims = base_layer.weight.shape[0]
            packed_input_dims = base_layer.weight.shape[1]
            input_dims = packed_input_dims * 32 // base_layer.bits
        elif isinstance(base_layer, nn.Linear):
            output_dims, input_dims = base_layer.weight.shape
        else:
            raise TypeError(
                "base_layer must be Linear or QuantizedLinear"
            )

        self.base_layer = base_layer
        self.scale = scale

        key_a = mx.random.key(seed)
        key_b = mx.random.key(seed + 1)

        self.a = mx.random.normal(
            shape=(input_dims, rank),
            key=key_a,
        ) / math.sqrt(input_dims)

        self.b = mx.random.normal(
            shape=(rank, output_dims),
            key=key_b,
        ) / math.sqrt(rank)

        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        """Run the base layer and add the low-rank perturbation."""
        a = self.a.astype(x.dtype)
        b = self.b.astype(x.dtype)

        base_output = self.base_layer(x)
        low_rank_output = (x @ a) @ b

        perturbation = (
            self.scale * low_rank_output
        ).astype(base_output.dtype)

        return base_output + perturbation


class LocalPerturbation:
    """Temporarily installs one random perturbation into the model."""

    def __init__(
        self,
        base_model: BaseModel,
        config: PerturbationConfig | None = None,
    ) -> None:
        self.base_model = base_model
        self.config = (
            PerturbationConfig()
            if config is None
            else config
        )
        self._original_q_proj: nn.Module | None = None

    def _target_attention(self) -> nn.Module:
        layers = self.base_model.model.layers
        index = self.config.layer_index

        if index >= len(layers):
            raise IndexError(
                f"layer_index {index} is invalid; "
                f"model has {len(layers)} layers"
            )

        return layers[index].self_attn

    def apply(self) -> None:
        if self._original_q_proj is not None:
            raise RuntimeError("perturbation is already active")

        attention = self._target_attention()
        original_q_proj = attention.q_proj

        attention.q_proj = RandomLowRankLinear(
            base_layer=original_q_proj,
            rank=self.config.rank,
            scale=self.config.scale,
            seed=self.config.seed,
        )

        self._original_q_proj = original_q_proj

    def remove(self) -> None:
        if self._original_q_proj is None:
            raise RuntimeError("perturbation is not active")

        attention = self._target_attention()
        attention.q_proj = self._original_q_proj
        self._original_q_proj = None

    @contextmanager
    def active(self) -> Iterator[None]:
        self.apply()

        try:
            yield
        finally:
            self.remove()


def main() -> None:
    base_model = BaseModel()

    config = PerturbationConfig(seed=7)
    perturbation = LocalPerturbation(base_model, config)

    prompt = (
        "In exactly five words, describe a quiet library."
    )

    attention = (
        base_model.model
        .layers[config.layer_index]
        .self_attn
    )
    original_q_proj = attention.q_proj

    print(f"Before: {type(attention.q_proj).__name__}")
    baseline = base_model.generate(prompt, max_tokens=24)

    with perturbation.active():
        print(f"During: {type(attention.q_proj).__name__}")
        changed = base_model.generate(prompt, max_tokens=24)

    print(f"After: {type(attention.q_proj).__name__}")
    print(
        f"Exactly restored: "
        f"{attention.q_proj is original_q_proj}"
    )

    restored = base_model.generate(prompt, max_tokens=24)

    print(f"Baseline: {baseline}")
    print(f"Perturbed: {changed}")
    print(f"Restored: {restored}")


if __name__ == "__main__":
    main()