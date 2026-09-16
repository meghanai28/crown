from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

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

        self.base_layer = base_layer
        self.rank = rank
        self.scale = scale
        self.seed = seed
        self.input_dims = input_dims
        self.output_dims = output_dims

        key_a = mx.random.key(seed)
        key_b = mx.random.key(seed + 1)

        a = mx.random.normal(
            shape=(input_dims, rank),
            key=key_a,
        ).astype(mx.float32)
        b = mx.random.normal(
            shape=(rank, output_dims),
            key=key_b,
        ).astype(mx.float32)

        # The effective update is (A @ B).T. Its Frobenius norm can
        # be calculated using only two rank-by-rank Gram matrices,
        # so we never materialize a huge dense delta_W matrix.
        gram_a = a.T @ a
        gram_b = b @ b.T
        update_norm_squared = mx.sum(
            gram_a * gram_b
        )
        update_norm = mx.sqrt(
            mx.maximum(update_norm_squared, 1e-30)
        )
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
    """Support both older and newer MLX-LM Llama layouts."""

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
    ) -> None:
        self.base_model = base_model
        self.config = (
            PerturbationConfig()
            if config is None
            else config
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
                    wrapper = NormalizedLowRankLinear(
                        original,
                        rank=self.config.rank,
                        scale=self.config.scale,
                        seed=component_seed,
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