from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

import mlx.core as mx
import mlx.nn as nn

from model import BaseModel


DEFAULT_MODULE_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class PerturbationConfig:
    """Defines one reproducible direction in local weight space.

    If layer_indices is None, four layers spread through model depth
    are selected automatically.

    scale is a relative Frobenius norm. For every targeted matrix W,
    the installed update satisfies approximately

        ||delta_W||_F / ||W||_F = abs(scale).

    Reusing one seed at different scales follows the same direction
    at different distances from the base model.
    """

    layer_indices: tuple[int, ...] | None = None
    module_paths: tuple[str, ...] = DEFAULT_MODULE_PATHS
    rank: int = 4
    scale: float = 0.01
    seed: int = 0

    # Fraction of the update's energy drawn inside a structured
    # subspace; the remainder stays isotropic. 0.0 reproduces the
    # original random-direction baseline and is the control arm.
    # A soft prior is used rather than hard projection because a
    # subspace estimated from one checkpoint is noisy, and hard
    # projection throws away every direction it failed to capture.
    subspace_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.layer_indices is not None:
            if not self.layer_indices:
                raise ValueError(
                    "layer_indices cannot be empty"
                )

            if any(
                index < 0
                for index in self.layer_indices
            ):
                raise ValueError(
                    "layer indices cannot be negative"
                )

            if (
                len(set(self.layer_indices))
                != len(self.layer_indices)
            ):
                raise ValueError(
                    "layer_indices cannot contain duplicates"
                )

        if not self.module_paths:
            raise ValueError(
                "module_paths cannot be empty"
            )

        if any(
            not path.strip()
            for path in self.module_paths
        ):
            raise ValueError(
                "module paths cannot be empty"
            )

        if (
            len(set(self.module_paths))
            != len(self.module_paths)
        ):
            raise ValueError(
                "module_paths cannot contain duplicates"
            )

        if self.rank <= 0:
            raise ValueError("rank must be positive")

        if not math.isfinite(self.scale):
            raise ValueError("scale must be finite")

        if self.seed < 0:
            raise ValueError("seed cannot be negative")

        if not 0.0 <= self.subspace_weight <= 1.0:
            raise ValueError(
                "subspace_weight must lie in [0, 1]"
            )


# Computing the norm of a quantized matrix requires temporarily
# dequantizing it. Cache the resulting scalar so a scale sweep only
# pays that cost once per base layer.
_WEIGHT_NORM_CACHE: dict[int, float] = {}


def _matrix_dimensions(
    layer: nn.Module,
) -> tuple[int, int]:
    if isinstance(layer, nn.QuantizedLinear):
        output_dims = layer.weight.shape[0]
        packed_input_dims = layer.weight.shape[1]
        input_dims = (
            packed_input_dims * 32 // layer.bits
        )
        return input_dims, output_dims

    if isinstance(layer, nn.Linear):
        output_dims, input_dims = layer.weight.shape
        return input_dims, output_dims

    raise TypeError(
        "target module must be Linear or QuantizedLinear; "
        f"received {type(layer).__name__}"
    )


def _dequantized_weight(
    layer: nn.Module,
) -> mx.array:
    if not isinstance(layer, nn.QuantizedLinear):
        return layer.weight

    options = {
        "group_size": layer.group_size,
        "bits": layer.bits,
    }

    if hasattr(layer, "mode"):
        options["mode"] = layer.mode

    return mx.dequantize(
        layer.weight,
        layer.scales,
        layer.biases,
        **options,
    )


def dequantized_weight(layer: nn.Module) -> mx.array:
    """Public view of a target matrix, shaped (output, input)."""

    return _dequantized_weight(layer)


def _factored_frobenius_norm(
    a: mx.array,
    b: mx.array,
) -> mx.array:
    """Norm of (a @ b) without materializing the dense product."""

    return mx.sqrt(
        mx.maximum(
            mx.sum((a.T @ a) * (b @ b.T)),
            1e-30,
        )
    )


def _weight_frobenius_norm(
    layer: nn.Module,
) -> float:
    cache_key = id(layer)

    if cache_key in _WEIGHT_NORM_CACHE:
        return _WEIGHT_NORM_CACHE[cache_key]

    weight = _dequantized_weight(layer)
    norm = mx.linalg.norm(
        weight.astype(mx.float32)
    )
    mx.eval(norm)

    value = float(norm.item())

    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(
            "base weight has an invalid Frobenius norm"
        )

    _WEIGHT_NORM_CACHE[cache_key] = value
    return value


def _component_seed(
    seed: int,
    layer_index: int,
    module_path: str,
) -> int:
    """Derive a stable seed without Python's randomized hash()."""

    label = f"{seed}:{layer_index}:{module_path}"
    digest = hashlib.sha256(
        label.encode("utf-8")
    ).digest()

    return (
        int.from_bytes(digest[:8], "little")
        % (2**31 - 2)
    )


class NormalizedLowRankLinear(nn.Module):
    """A frozen base layer plus one normalized low-rank update."""

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        rank: int,
        scale: float,
        seed: int,
        basis: tuple[mx.array, mx.array] | None = None,
        subspace_weight: float = 0.0,
    ) -> None:
        super().__init__()

        input_dims, output_dims = (
            _matrix_dimensions(base_layer)
        )

        if rank > min(input_dims, output_dims):
            raise ValueError(
                f"rank {rank} is larger than the smallest "
                "matrix dimension"
            )

        if subspace_weight > 0.0 and basis is None:
            raise ValueError(
                "a positive subspace_weight needs a basis"
            )

        self.base_layer = base_layer
        self.rank = rank
        self.scale = scale
        self.seed = seed
        self.subspace_weight = subspace_weight
        self.input_dims = input_dims
        self.output_dims = output_dims

        keys = mx.random.split(
            mx.random.key(seed),
            4,
        )

        # Two independent rank-r updates are mixed by concatenating
        # along the rank axis, because
        #     [a1 | a2] @ [[b1], [b2]] == a1 @ b1 + a2 @ b2.
        # Each part is first normalized to unit Frobenius norm and
        # then weighted by sqrt, so subspace_weight is the fraction
        # of the update's *energy* drawn inside the subspace.
        parts_a: list[mx.array] = []
        parts_b: list[mx.array] = []

        isotropic_gain = math.sqrt(1.0 - subspace_weight)
        subspace_gain = math.sqrt(subspace_weight)

        if isotropic_gain > 0.0:
            a_iso = mx.random.normal(
                shape=(input_dims, rank),
                key=keys[0],
            ).astype(mx.float32)
            b_iso = mx.random.normal(
                shape=(rank, output_dims),
                key=keys[1],
            ).astype(mx.float32)
            b_iso = b_iso * (
                isotropic_gain
                / _factored_frobenius_norm(a_iso, b_iso)
            )
            parts_a.append(a_iso)
            parts_b.append(b_iso)

        if subspace_gain > 0.0:
            assert basis is not None
            right_basis, left_basis = basis
            basis_rank = right_basis.shape[1]

            if rank > basis_rank:
                raise ValueError(
                    f"rank {rank} exceeds the basis rank "
                    f"{basis_rank}"
                )

            # delta_W = U (Q^T P^T) V^T lies in the span of the
            # basis on both sides, which is what restricts the
            # sample to the subspace.
            projector_p = mx.random.normal(
                shape=(basis_rank, rank),
                key=keys[2],
            ).astype(mx.float32)
            projector_q = mx.random.normal(
                shape=(rank, basis_rank),
                key=keys[3],
            ).astype(mx.float32)

            a_sub = right_basis.astype(mx.float32) @ projector_p
            b_sub = projector_q @ left_basis.astype(mx.float32).T
            b_sub = b_sub * (
                subspace_gain
                / _factored_frobenius_norm(a_sub, b_sub)
            )
            parts_a.append(a_sub)
            parts_b.append(b_sub)

        a = mx.concatenate(parts_a, axis=1)
        b = mx.concatenate(parts_b, axis=0)

        # The effective update is (A @ B).T. Its Frobenius norm can
        # be calculated using only two rank-by-rank Gram matrices,
        # so we never materialize a huge dense delta_W matrix.
        update_norm = _factored_frobenius_norm(a, b)
        mx.eval(update_norm)

        raw_update_norm = float(
            update_norm.item()
        )
        base_weight_norm = _weight_frobenius_norm(
            base_layer
        )

        if (
            not math.isfinite(raw_update_norm)
            or raw_update_norm <= 0.0
        ):
            raise RuntimeError(
                "random update has an invalid Frobenius norm"
            )

        normalization_gain = (
            base_weight_norm / raw_update_norm
        )

        self.a = a
        self.b = b * normalization_gain
        self.base_weight_norm = base_weight_norm
        self.raw_update_norm = raw_update_norm
        self.normalization_gain = normalization_gain

        mx.eval(self.a, self.b)
        self.freeze()

    @property
    def relative_weight_scale(self) -> float:
        return abs(self.scale)

    def __call__(self, x: mx.array) -> mx.array:
        """Run W x plus the fixed low-rank delta_W x."""

        base_output = self.base_layer(x)
        a = self.a.astype(x.dtype)
        b = self.b.astype(x.dtype)
        low_rank_output = (x @ a) @ b
        perturbation = (
            self.scale * low_rank_output
        ).astype(base_output.dtype)

        return base_output + perturbation


class BasisProvider(Protocol):
    """Supplies the (right, left) basis for one target matrix.

    Implementations live in subspace.py. The call is expected to be
    cached, because one density sweep installs the same matrices
    hundreds of times.
    """

    def __call__(
        self,
        label: str,
        base_layer: nn.Module,
    ) -> tuple[mx.array, mx.array]:
        ...


@dataclass
class _InstalledModule:
    parent: nn.Module
    attribute: str
    original: nn.Module
    wrapper: NormalizedLowRankLinear
    label: str


def get_transformer_layers(
    base_model: BaseModel,
) -> Sequence[nn.Module]:
    """Support both older and newer MLX-LM decoder layouts."""

    model = base_model.model

    if hasattr(model, "layers"):
        return model.layers

    inner_model = getattr(model, "model", None)

    if (
        inner_model is not None
        and hasattr(inner_model, "layers")
    ):
        return inner_model.layers

    raise AttributeError(
        "could not find transformer layers on the loaded model"
    )


def all_layer_indices(
    base_model: BaseModel,
) -> tuple[int, ...]:
    """Every transformer layer.

    The four-layer default explores a very small slice of the weight
    space, which is a poor probe for any claim about how densely
    useful models are packed around the base.
    """

    return tuple(
        range(len(get_transformer_layers(base_model)))
    )


def automatic_layer_indices(
    number_of_layers: int,
) -> tuple[int, ...]:
    """Choose four depth-spanning layers."""

    if number_of_layers <= 0:
        raise ValueError(
            "number_of_layers must be positive"
        )

    candidates = (
        number_of_layers // 4,
        number_of_layers // 2,
        (3 * number_of_layers) // 4,
        number_of_layers - 1,
    )

    return tuple(sorted(set(candidates)))


def _resolve_parent(
    root: nn.Module,
    module_path: str,
) -> tuple[nn.Module, str]:
    path_parts = module_path.split(".")
    parent = root

    for part in path_parts[:-1]:
        if not hasattr(parent, part):
            raise AttributeError(
                f"{type(parent).__name__} has no "
                f"submodule {part!r} while resolving "
                f"{module_path!r}"
            )

        parent = getattr(parent, part)

    attribute = path_parts[-1]

    if not hasattr(parent, attribute):
        raise AttributeError(
            f"{type(parent).__name__} has no "
            f"submodule {attribute!r} while resolving "
            f"{module_path!r}"
        )

    return parent, attribute


class LocalPerturbation:
    """Temporarily install one multi-layer random direction."""

    def __init__(
        self,
        base_model: BaseModel,
        config: PerturbationConfig | None = None,
        basis_provider: BasisProvider | None = None,
    ) -> None:
        self.base_model = base_model
        self.config = (
            PerturbationConfig()
            if config is None
            else config
        )
        self.basis_provider = basis_provider

        if (
            self.config.subspace_weight > 0.0
            and basis_provider is None
        ):
            raise ValueError(
                "a positive subspace_weight needs a "
                "basis_provider"
            )
        self._installed: list[_InstalledModule] = []
        self._is_active = False

        layers = get_transformer_layers(base_model)

        if self.config.layer_indices is None:
            self.layer_indices = automatic_layer_indices(
                len(layers)
            )
        else:
            self.layer_indices = tuple(
                self.config.layer_indices
            )

        invalid_indices = [
            index
            for index in self.layer_indices
            if index >= len(layers)
        ]

        if invalid_indices:
            raise IndexError(
                f"invalid layer indices {invalid_indices}; "
                f"model has {len(layers)} layers"
            )

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def wrappers(
        self,
    ) -> tuple[NormalizedLowRankLinear, ...]:
        return tuple(
            item.wrapper
            for item in self._installed
        )

    @property
    def target_labels(self) -> tuple[str, ...]:
        return tuple(
            f"layers.{layer_index}.{module_path}"
            for layer_index in self.layer_indices
            for module_path in self.config.module_paths
        )

    def apply(self) -> None:
        if self._is_active:
            raise RuntimeError(
                "perturbation is already active"
            )

        layers = get_transformer_layers(
            self.base_model
        )
        self._is_active = True

        try:
            for layer_index in self.layer_indices:
                layer = layers[layer_index]

                for module_path in self.config.module_paths:
                    parent, attribute = _resolve_parent(
                        layer,
                        module_path,
                    )
                    original = getattr(
                        parent,
                        attribute,
                    )
                    component_seed = _component_seed(
                        self.config.seed,
                        layer_index,
                        module_path,
                    )
                    label = (
                        f"layers.{layer_index}."
                        f"{module_path}"
                    )
                    basis = (
                        None
                        if self.basis_provider is None
                        or self.config.subspace_weight
                        <= 0.0
                        else self.basis_provider(
                            label,
                            original,
                        )
                    )
                    wrapper = NormalizedLowRankLinear(
                        original,
                        rank=self.config.rank,
                        scale=self.config.scale,
                        seed=component_seed,
                        basis=basis,
                        subspace_weight=(
                            self.config.subspace_weight
                        ),
                    )

                    setattr(
                        parent,
                        attribute,
                        wrapper,
                    )

                    self._installed.append(
                        _InstalledModule(
                            parent=parent,
                            attribute=attribute,
                            original=original,
                            wrapper=wrapper,
                            label=(
                                f"layers.{layer_index}."
                                f"{module_path}"
                            ),
                        )
                    )
        except Exception:
            self._restore_installed_modules()
            self._is_active = False
            raise

    def _restore_installed_modules(self) -> None:
        while self._installed:
            item = self._installed.pop()
            setattr(
                item.parent,
                item.attribute,
                item.original,
            )

    def remove(self) -> None:
        if not self._is_active:
            raise RuntimeError(
                "perturbation is not active"
            )

        self._restore_installed_modules()
        self._is_active = False

    @contextmanager
    def active(
        self,
    ) -> Iterator[LocalPerturbation]:
        self.apply()

        try:
            yield self
        finally:
            self.remove()


def main() -> None:
    """Run a small installation and restoration smoke test."""

    base_model = BaseModel()
    config = PerturbationConfig(
        rank=4,
        scale=0.01,
        seed=7,
    )
    perturbation = LocalPerturbation(
        base_model,
        config,
    )

    print(
        "Target layers:",
        perturbation.layer_indices,
    )
    print(
        "Target matrices:",
        len(perturbation.target_labels),
    )

    with perturbation.active():
        print(
            "Installed wrappers:",
            len(perturbation.wrappers),
        )

    print(
        "Restored:",
        not perturbation.is_active,
    )


if __name__ == "__main__":
    main()