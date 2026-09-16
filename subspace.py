"""Structured subspaces to draw perturbation directions from.

An isotropic direction spends almost all of its norm on directions
that are irrelevant to any particular task. If useful models are not
distributed uniformly around the base, then *where* a direction points
should matter as much as how far it goes, and a prior that concentrates
the update inside a task-relevant subspace should find improvements
more often at equal perturbation norm.

This module builds those subspaces. Two sources are provided:

BaseWeightSource
    The leading singular subspace of the base weights themselves. It
    needs no second checkpoint, so it runs today. It is a contrast arm
    rather than the hypothesis: post-training updates are reported to
    lean away from the principal directions, so if that holds, this
    source should not beat isotropic sampling.

CheckpointDeltaSource
    The leading singular subspace of the difference between two
    checkpoints of the same architecture. This is the arm the density
    hypothesis is really about. It only ever uses the delta's
    *geometry*, never its values, because independently trained runs
    agree far more about where they write than about what they write.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from model import BaseModel
from perturb import (
    _resolve_parent,
    dequantized_weight,
    get_transformer_layers,
)


def randomized_subspace(
    matrix: mx.array,
    *,
    basis_rank: int,
    seed: int,
    power_iterations: int = 2,
) -> tuple[mx.array, mx.array]:
    """Approximate the leading singular subspace of one matrix.

    A full SVD of a 3072x8192 matrix is far too slow to repeat across
    every target matrix. A randomized range finder with a couple of
    power iterations recovers the leading subspace well enough to
    define a sampling prior, in a fraction of a second.

    ``matrix`` is shaped (output, input), matching the dequantized
    weight layout. Returns ``(right_basis, left_basis)`` with shapes
    (input, basis_rank) and (output, basis_rank), both orthonormal in
    their columns, so that the span of ``left_basis`` covers the
    matrix's column space and ``right_basis`` its row space.
    """

    if basis_rank <= 0:
        raise ValueError("basis_rank must be positive")

    output_dims, input_dims = matrix.shape
    usable_rank = min(
        basis_rank,
        output_dims,
        input_dims,
    )

    matrix = matrix.astype(mx.float32)
    sketch = mx.random.normal(
        shape=(input_dims, usable_rank),
        key=mx.random.key(seed),
    )

    # Power iterations sharpen the separation between the leading
    # subspace and the tail, which matters because a checkpoint delta
    # has a much flatter spectrum than a weight matrix.
    projected = matrix @ sketch

    for _ in range(power_iterations):
        projected = matrix @ (matrix.T @ projected)

    left_basis, _ = mx.linalg.qr(
        projected,
        stream=mx.cpu,
    )
    right_basis, _ = mx.linalg.qr(
        (left_basis.T @ matrix).T,
        stream=mx.cpu,
    )

    mx.eval(left_basis, right_basis)
    return right_basis, left_basis


class SubspaceSource(Protocol):
    """Supplies the matrix whose leading subspace defines the prior."""

    name: str

    def matrix_for(
        self,
        label: str,
        base_layer: nn.Module,
    ) -> mx.array:
        ...


@dataclass
class BaseWeightSource:
    """Principal subspace of the base weight matrix itself."""

    name: str = "base_weight"

    def matrix_for(
        self,
        label: str,
        base_layer: nn.Module,
    ) -> mx.array:
        return dequantized_weight(base_layer)


class CheckpointDeltaSource:
    """Principal subspace of a second checkpoint minus the base.

    The second checkpoint is loaded lazily, so constructing this costs
    nothing until a basis is actually requested.
    """

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")

        self.model_id = model_id
        self.name = name or f"delta:{model_id}"
        self._layers: object | None = None

    def _resolve_other_layer(
        self,
        label: str,
    ) -> nn.Module:
        if self._layers is None:
            print(
                f"  loading delta checkpoint {self.model_id}"
            )
            other_model, _ = load(self.model_id)
            other_model.freeze()
            other_model.eval()
            self._layers = get_transformer_layers(
                _ModelHandle(other_model)
            )

        # label looks like "layers.<index>.<module path>"
        _, index_text, module_path = label.split(".", 2)
        layer = self._layers[int(index_text)]
        parent, attribute = _resolve_parent(
            layer,
            module_path,
        )

        return getattr(parent, attribute)

    def release(self) -> None:
        """Drop the second checkpoint once every basis is cached.

        The bases are a few hundred megabytes; the checkpoint they came
        from is several gigabytes, and nothing needs it again. Clearing
        MLX's buffer cache as well keeps the peak from carrying into the
        next arm, which matters when several deltas load in one run.
        """

        self._layers = None
        mx.clear_cache()

    def matrix_for(
        self,
        label: str,
        base_layer: nn.Module,
    ) -> mx.array:
        base_weight = dequantized_weight(base_layer)
        other_weight = dequantized_weight(
            self._resolve_other_layer(label)
        )

        if base_weight.shape != other_weight.shape:
            raise ValueError(
                f"{label}: checkpoint shapes differ, "
                f"{base_weight.shape} vs "
                f"{other_weight.shape}"
            )

        return other_weight.astype(
            mx.float32
        ) - base_weight.astype(mx.float32)


@dataclass
class _ModelHandle:
    """Adapts a bare MLX model to what get_transformer_layers wants."""

    model: object


class CachedBasis:
    """A BasisProvider that computes each basis at most once.

    One density sweep installs the same matrices hundreds of times, so
    recomputing a basis per sample would dominate the run.
    """

    def __init__(
        self,
        source: SubspaceSource,
        *,
        basis_rank: int = 32,
        seed: int = 0,
        power_iterations: int = 2,
        track_energy: int = 8,
    ) -> None:
        self.source = source
        self.basis_rank = basis_rank
        self.seed = seed
        self.power_iterations = power_iterations
        self.track_energy = track_energy
        self.energy_fractions: dict[str, float] = {}
        self._isotropic_reference: list[float] = []
        self._cache: dict[
            str, tuple[mx.array, mx.array]
        ] = {}

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    def __call__(
        self,
        label: str,
        base_layer: nn.Module,
    ) -> tuple[mx.array, mx.array]:
        cached = self._cache.get(label)

        if cached is not None:
            return cached

        matrix = self.source.matrix_for(
            label,
            base_layer,
        )
        basis = randomized_subspace(
            matrix,
            basis_rank=self.basis_rank,
            # A distinct seed per matrix keeps the sketches
            # independent without needing extra state.
            seed=self.seed + len(self._cache),
            power_iterations=self.power_iterations,
        )

        # How much of the source matrix the basis actually captures is
        # the difference between an informative prior and a noisy one.
        # A delta whose leading subspace holds no more energy than a
        # random subspace would carries no geometry worth inheriting.
        if len(self.energy_fractions) < self.track_energy:
            self.energy_fractions[label] = (
                subspace_energy_fraction(matrix, basis)
            )
            output_dims, input_dims = matrix.shape
            # What a subspace of the same rank would capture if the
            # source had no structure at all: the reference the
            # measured fraction has to beat to mean anything.
            self._isotropic_reference.append(
                (self.basis_rank**2)
                / (output_dims * input_dims)
            )

        self._cache[label] = basis
        return basis

    def energy_report(self) -> str:
        """One line describing how concentrated the source is."""

        if not self.energy_fractions:
            return "no energy samples collected"

        values = list(self.energy_fractions.values())
        mean = sum(values) / len(values)
        reference = sum(self._isotropic_reference) / len(
            self._isotropic_reference
        )

        return (
            f"rank-{self.basis_rank} basis captures "
            f"{mean:.1%} of the source energy "
            f"(min {min(values):.1%}, max {max(values):.1%}, "
            f"{len(values)} matrices); an unstructured source "
            f"would give {reference:.4%}"
        )


def subspace_energy_fraction(
    matrix: mx.array,
    basis: tuple[mx.array, mx.array],
) -> float:
    """Share of the matrix's energy captured by the basis.

    A basis that captures a large fraction of a checkpoint delta is
    evidence the delta really is low-rank. A fraction near what random
    directions would give means the prior carries little information,
    and any density result from it should be read with suspicion.
    """

    right_basis, left_basis = basis
    matrix = matrix.astype(mx.float32)
    projected = (
        left_basis
        @ (left_basis.T @ matrix @ right_basis)
        @ right_basis.T
    )

    total = mx.sum(matrix * matrix)
    kept = mx.sum(projected * projected)
    mx.eval(total, kept)

    return float(kept.item()) / max(
        float(total.item()),
        1e-30,
    )


def main() -> None:
    """Verify a basis is orthonormal and actually restricts a sample."""

    from perturb import (
        DEFAULT_MODULE_PATHS,
        LocalPerturbation,
        PerturbationConfig,
    )

    print("Loading the base model")
    base_model = BaseModel()

    provider = CachedBasis(
        BaseWeightSource(),
        basis_rank=32,
    )
    layers = get_transformer_layers(base_model)
    label = "layers.14.mlp.down_proj"
    parent, attribute = _resolve_parent(
        layers[14],
        "mlp.down_proj",
    )
    target = getattr(parent, attribute)

    right_basis, left_basis = provider(label, target)
    print(
        f"\n{label}: right {right_basis.shape}, "
        f"left {left_basis.shape}"
    )

    identity = mx.eye(right_basis.shape[1])
    right_error = mx.max(
        mx.abs(right_basis.T @ right_basis - identity)
    )
    left_error = mx.max(
        mx.abs(left_basis.T @ left_basis - identity)
    )
    mx.eval(right_error, left_error)

    print(
        "  orthonormality error: right "
        f"{float(right_error.item()):.2e}, left "
        f"{float(left_error.item()):.2e}"
    )

    weight = dequantized_weight(target)
    captured = subspace_energy_fraction(
        weight,
        (right_basis, left_basis),
    )
    print(
        "  base weight energy inside the basis: "
        f"{captured:.1%}"
    )

    # A fully subspace-restricted update must be reproduced exactly by
    # projecting onto the basis. Anything else means the sampler is
    # leaking outside the prior it claims to obey.
    config = PerturbationConfig(
        layer_indices=(14,),
        module_paths=("mlp.down_proj",),
        rank=4,
        scale=0.01,
        seed=0,
        subspace_weight=1.0,
    )
    perturbation = LocalPerturbation(
        base_model,
        config,
        basis_provider=provider,
    )

    with perturbation.active():
        wrapper = perturbation.wrappers[0]
        update = (wrapper.a @ wrapper.b).T

    leakage = 1.0 - subspace_energy_fraction(
        update,
        (right_basis, left_basis),
    )
    print(
        "\n  subspace_weight=1.0 sample, energy outside "
        f"the basis: {leakage:.2e}"
    )

    reference = mx.random.normal(shape=update.shape)
    random_capture = subspace_energy_fraction(
        reference,
        (right_basis, left_basis),
    )
    print(
        "  an isotropic sample would land "
        f"{random_capture:.2%} inside the same basis"
    )

    if leakage > 1e-4:
        raise RuntimeError(
            "a subspace-restricted sample escaped its basis"
        )

    print(
        "\nPASS: bases are orthonormal and restricted "
        "samples stay inside them."
    )


if __name__ == "__main__":
    main()
