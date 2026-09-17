"""A small coordinate system inside the subspace.

Stage 1 found the map: math and coder deltas share a region that generic
post-training does not explain. It also found that random points in that
region are not good. Stage 2a asks whether *specific* points are.

To ask that you need an address you can actually search over. The raw
parameterisation is hopeless: dW = U M V^T with M of shape 32x32 across
196 matrices is 200,704 numbers, and no gradient-free method finds a
good point in that with a few hundred evaluations.

So this builds a dictionary instead. D fixed directions are drawn once
inside the subspace, each one spanning the whole model, and an address is
the D-dimensional vector of weights over them:

    delta_W(w) = sum_i  w_i * direction_i

D is 24 by default, so an address is 24 numbers regardless of model size
-- which is also the shape a navigator would have to emit.

Only the weights change during a search, so the factors are built once
and the forward pass applies the weights in between them:

    (x @ A) * w  @ B

No matrix is rebuilt per evaluation, and the Frobenius normalization
reduces to a DxD elementwise product using two Gram matrices cached at
construction. Changing an address costs almost nothing.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import mlx.core as mx
import mlx.nn as nn

from model import BaseModel
from perturb import (
    DEFAULT_MODULE_PATHS,
    BasisProvider,
    _component_seed,
    _matrix_dimensions,
    _resolve_parent,
    _weight_frobenius_norm,
    get_transformer_layers,
)


class DictionaryDirection(nn.Module):
    """One frozen matrix plus a weighted sum of D fixed directions."""

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        basis: tuple[mx.array, mx.array],
        dictionary_size: int,
        rank: int,
        seed: int,
    ) -> None:
        super().__init__()

        input_dims, output_dims = _matrix_dimensions(base_layer)
        right_basis, left_basis = basis
        basis_rank = right_basis.shape[1]

        if rank > basis_rank:
            raise ValueError(
                f"rank {rank} exceeds basis rank {basis_rank}"
            )

        self.base_layer = base_layer
        self.dictionary_size = dictionary_size
        self.rank = rank
        self.base_weight_norm = _weight_frobenius_norm(base_layer)

        right = right_basis.astype(mx.float32)
        left = left_basis.astype(mx.float32)

        a_parts: list[mx.array] = []
        b_parts: list[mx.array] = []

        for index in range(dictionary_size):
            keys = mx.random.split(
                mx.random.key(seed + index * 7919),
                2,
            )
            projector_p = mx.random.normal(
                shape=(basis_rank, rank),
                key=keys[0],
            ).astype(mx.float32)
            projector_q = mx.random.normal(
                shape=(rank, basis_rank),
                key=keys[1],
            ).astype(mx.float32)

            # Confined to the subspace on both sides, exactly as in the
            # stage 1 sampler.
            part_a = right @ projector_p
            part_b = projector_q @ left.T

            # Unit Frobenius per direction, so a weight vector means the
            # same thing in every matrix.
            norm = mx.sqrt(
                mx.maximum(
                    mx.sum((part_a.T @ part_a) * (part_b @ part_b.T)),
                    1e-30,
                )
            )
            a_parts.append(part_a)
            b_parts.append(part_b / norm)

        self.a = mx.concatenate(a_parts, axis=1)
        self.b = mx.concatenate(b_parts, axis=0)

        # Gram matrices let the norm of any weighted combination be
        # computed without touching the large factors again.
        self._gram_a = self.a.T @ self.a
        self._gram_b = self.b @ self.b.T

        mx.eval(self.a, self.b, self._gram_a, self._gram_b)

        self.weights = mx.zeros((dictionary_size * rank,))
        self.freeze()

    def combination_norm(self, expanded: mx.array) -> mx.array:
        """Frobenius norm of the update at these expanded weights."""

        outer = expanded[:, None] * expanded[None, :]
        return mx.sqrt(
            mx.maximum(
                mx.sum(self._gram_a * self._gram_b * outer),
                1e-30,
            )
        )

    def set_address(
        self,
        address: mx.array,
        *,
        scale: float,
    ) -> None:
        """Point this layer at one address, normalized to `scale`."""

        expanded = mx.repeat(address, self.rank)
        norm = self.combination_norm(expanded)
        gain = (self.base_weight_norm * scale) / norm
        self.weights = expanded * gain
        mx.eval(self.weights)

    def __call__(self, x: mx.array) -> mx.array:
        base_output = self.base_layer(x)
        hidden = (x @ self.a.astype(x.dtype)) * self.weights.astype(
            x.dtype
        )
        update = (hidden @ self.b.astype(x.dtype)).astype(
            base_output.dtype
        )

        return base_output + update


@dataclass
class _Installed:
    parent: nn.Module
    attribute: str
    original: nn.Module
    wrapper: DictionaryDirection


class DictionaryPerturbation:
    """Installs one shared D-direction dictionary across the model.

    The same D-dimensional address drives every target matrix, so an
    address is D numbers for the whole model rather than D per matrix.
    That is what makes it something a navigator could plausibly emit.
    """

    def __init__(
        self,
        base_model: BaseModel,
        basis_provider: BasisProvider,
        *,
        layer_indices: Sequence[int],
        module_paths: Sequence[str] = DEFAULT_MODULE_PATHS,
        dictionary_size: int = 24,
        rank: int = 4,
        seed: int = 0,
    ) -> None:
        self.base_model = base_model
        self.basis_provider = basis_provider
        self.layer_indices = tuple(layer_indices)
        self.module_paths = tuple(module_paths)
        self.dictionary_size = dictionary_size
        self.rank = rank
        self.seed = seed
        self._installed: list[_Installed] = []
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def apply(self) -> None:
        if self._active:
            raise RuntimeError("dictionary is already active")

        layers = get_transformer_layers(self.base_model)
        self._active = True

        try:
            for layer_index in self.layer_indices:
                layer = layers[layer_index]

                for module_path in self.module_paths:
                    parent, attribute = _resolve_parent(
                        layer,
                        module_path,
                    )
                    original = getattr(parent, attribute)
                    label = f"layers.{layer_index}.{module_path}"
                    wrapper = DictionaryDirection(
                        original,
                        basis=self.basis_provider(label, original),
                        dictionary_size=self.dictionary_size,
                        rank=self.rank,
                        seed=_component_seed(
                            self.seed,
                            layer_index,
                            module_path,
                        ),
                    )
                    setattr(parent, attribute, wrapper)
                    self._installed.append(
                        _Installed(
                            parent=parent,
                            attribute=attribute,
                            original=original,
                            wrapper=wrapper,
                        )
                    )
        except Exception:
            self._restore()
            self._active = False
            raise

    def set_address(
        self,
        address: Sequence[float] | mx.array,
        *,
        scale: float,
    ) -> None:
        """Move every installed matrix to one address."""

        vector = mx.array(
            [float(value) for value in address],
            dtype=mx.float32,
        )
        length = mx.sqrt(mx.maximum(mx.sum(vector * vector), 1e-30))
        # Only the direction of an address matters; distance is set by
        # `scale`, which is calibrated separately against KL.
        vector = vector / length
        mx.eval(vector)

        for item in self._installed:
            item.wrapper.set_address(vector, scale=scale)

    def _restore(self) -> None:
        while self._installed:
            item = self._installed.pop()
            setattr(item.parent, item.attribute, item.original)

    def remove(self) -> None:
        if not self._active:
            raise RuntimeError("dictionary is not active")

        self._restore()
        self._active = False

    @contextmanager
    def active(self) -> Iterator[DictionaryPerturbation]:
        self.apply()

        try:
            yield self
        finally:
            self.remove()


def random_address(
    dictionary_size: int,
    seed: int,
) -> list[float]:
    values = mx.random.normal(
        shape=(dictionary_size,),
        key=mx.random.key(seed),
    )
    mx.eval(values)

    return [float(v) for v in values.tolist()]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    return dot / max(norm_a * norm_b, 1e-30)
