"""Where do different post-training deltas write, and do they share a region?

The 100-sample behavioural run found Math and Coder deltas to be
statistically identical (p = 0.78 on math, p = 0.66 on code). Behaviour
cannot say why. Geometry can, and for a fraction of the compute: measure
the principal angles between the subspaces directly, no sampling at all.

The measure is the mean squared cosine of the principal angles between
two rank-k subspaces, from 0 (orthogonal) to 1 (identical). Two random
k-dimensional subspaces of R^d already overlap by about k/d, so the raw
number is reported against that floor.

The control that matters
------------------------
`instruct-control` is Qwen2.5-7B minus Qwen2.5-7B-Instruct: a real
post-training delta carrying no task specialization at all. It separates
the two stories that fit the behavioural result equally well.

  If every task delta overlaps instruct-control as much as it overlaps
  the other task deltas, then post-training simply carves out one
  generic region and task identity is irrelevant. That is the
  high-plasticity story.

  If the task deltas share something with each other *beyond* what they
  share with instruct-control, there is task-family structure on top of
  the generic component. That is the shared-reasoning story.

The last section tests exactly that by projecting the instruct subspace
out of the task subspaces and re-measuring. Overlap that survives the
projection is structure that generic adaptation does not explain.
"""

from __future__ import annotations

import mlx.core as mx

from model import BaseModel
from perturb import (
    DEFAULT_MODULE_PATHS,
    LocalPerturbation,
    PerturbationConfig,
    all_layer_indices,
)
from subspace import CachedBasis, CheckpointDeltaSource

BASIS_RANK = 32

# Every one of these is (checkpoint - Qwen2.5-7B-Instruct) and every one
# shares the base's architecture and 4-bit/group-64 quantization, so the
# quantization error largely cancels in the difference.
SOURCES: dict[str, str] = {
    "math": "mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
    "coder": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    # Post-training with no task specialization: the generic baseline.
    "instruct-control": "mlx-community/Qwen2.5-7B-4bit",
}

# Deliberately excluded: Qwen2.5-7B-Medicine. It is LoRA-trained, so its
# delta has rank exactly equal to the adapter's. Its "subspace" is then a
# hyperparameter someone picked, not a discovered fact about where
# adaptation lands, which is the only thing this project is asking about.
# Any delta source added here must come from full fine-tuning, or from a
# LoRA of rank far above BASIS_RANK so that extracting 32 is a real
# compression rather than a copy.

CONTROL = "instruct-control"

GROUPS = {
    "attn": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
}


def mean_squared_cosine(
    basis_a: mx.array,
    basis_b: mx.array,
) -> float:
    """Mean squared cosine of the principal angles between two bases.

    Both have orthonormal columns, so the singular values of A^T B are
    the cosines of the principal angles and the squared Frobenius norm
    is their sum of squares.
    """

    cross = basis_a.T @ basis_b
    total = mx.sum(cross * cross)
    mx.eval(total)

    return float(total.item()) / basis_a.shape[1]


def project_out(
    basis: mx.array,
    removed: mx.array,
) -> tuple[mx.array, float]:
    """Remove one subspace from another and re-orthonormalize.

    Also returns the fraction of the original basis's energy that
    survived. A basis that loses most of its energy here was largely
    inside the subspace being removed, and whatever overlap remains is
    measured on very little signal.
    """

    residual = basis - removed @ (removed.T @ basis)
    kept = mx.sum(residual * residual) / mx.sum(basis * basis)
    mx.eval(kept)

    orthonormal, _ = mx.linalg.qr(residual, stream=mx.cpu)
    mx.eval(orthonormal)

    return orthonormal, float(kept.item())


def build_bases(
    model: BaseModel,
    layer_indices: tuple[int, ...],
    model_id: str,
) -> CachedBasis:
    source = CheckpointDeltaSource(model_id)
    provider = CachedBasis(source, basis_rank=BASIS_RANK)

    warm = LocalPerturbation(
        model,
        PerturbationConfig(
            layer_indices=layer_indices,
            module_paths=DEFAULT_MODULE_PATHS,
            rank=8,
            scale=0.01,
            seed=0,
            subspace_weight=1.0,
        ),
        basis_provider=provider,
    )

    with warm.active():
        pass

    source.release()
    return provider


def labels(layer_indices: tuple[int, ...]) -> list[str]:
    return [
        f"layers.{layer}.{path}"
        for layer in layer_indices
        for paths in GROUPS.values()
        for path in paths
    ]


def pair_overlap(
    providers: dict[str, CachedBasis],
    all_labels: list[str],
    first: str,
    second: str,
) -> float:
    """Mean overlap ratio against chance, over every target matrix."""

    ratios: list[float] = []

    for label in all_labels:
        for index in (0, 1):
            basis_a = providers[first]._cache[label][index]
            basis_b = providers[second]._cache[label][index]
            chance = BASIS_RANK / basis_a.shape[0]
            ratios.append(
                mean_squared_cosine(basis_a, basis_b) / chance
            )

    return sum(ratios) / len(ratios)


def main() -> None:
    print("Loading the base model")
    model = BaseModel()
    layer_indices = all_layer_indices(model)
    all_labels = labels(layer_indices)

    providers: dict[str, CachedBasis] = {}

    for name, model_id in SOURCES.items():
        print(f"\nBuilding {name} bases: {model_id}")
        providers[name] = build_bases(
            model,
            layer_indices,
            model_id,
        )
        print(f"  {providers[name].energy_report()}")

    print(
        "\nA source whose rank-32 basis captures nearly all of its "
        "energy was\ntrained with a low-rank adapter, and its "
        "'subspace' is the adapter's."
    )

    names = list(SOURCES)

    print("\n" + "=" * 64)
    print("Pairwise subspace overlap (x chance; 1.0 = no better than random)")
    print("=" * 64 + "\n")

    header = "".join(f"{n[:9]:>11}" for n in names)
    print(f"  {'':>17}{header}")

    matrix: dict[str, dict[str, float]] = {}

    for first in names:
        row = []
        matrix[first] = {}

        for second in names:
            if first == second:
                row.append(f"{'-':>11}")
                continue

            ratio = pair_overlap(
                providers,
                all_labels,
                first,
                second,
            )
            matrix[first][second] = ratio
            row.append(f"{ratio:>10.1f}x")

        print(f"  {first:>17}{''.join(row)}")

    # --- the decisive test ------------------------------------------------
    task_names = [n for n in names if n != CONTROL]

    print("\n" + "=" * 64)
    print(
        "Does task-delta overlap survive removing the generic direction?"
    )
    print("=" * 64)
    print(
        f"\nProjecting the {CONTROL} subspace out of each task subspace,"
        "\nthen re-measuring how much they still share.\n"
    )
    print(
        f"  {'pair':>22} {'raw':>9} {'after':>9} "
        f"{'energy kept':>12}"
    )

    for index, first in enumerate(task_names):
        for second in task_names[index + 1 :]:
            raw = matrix[first][second]
            residual_ratios: list[float] = []
            kept_fractions: list[float] = []

            for label in all_labels:
                for side in (0, 1):
                    control_basis = providers[CONTROL]._cache[label][side]
                    basis_a, kept_a = project_out(
                        providers[first]._cache[label][side],
                        control_basis,
                    )
                    basis_b, kept_b = project_out(
                        providers[second]._cache[label][side],
                        control_basis,
                    )
                    chance = BASIS_RANK / basis_a.shape[0]
                    residual_ratios.append(
                        mean_squared_cosine(basis_a, basis_b)
                        / chance
                    )
                    kept_fractions.append((kept_a + kept_b) / 2.0)

            after = sum(residual_ratios) / len(residual_ratios)
            kept = sum(kept_fractions) / len(kept_fractions)
            print(
                f"  {first + ' vs ' + second:>22} {raw:>8.1f}x "
                f"{after:>8.1f}x {kept:>11.1%}"
            )

    print(
        """
Reading it:

  after ~= raw     The shared structure is not generic adaptation. Task
                   deltas agree with each other about somewhere the
                   instruct delta does not go, which is real task-family
                   geometry and is what stage 2 would learn.

  after ~= 1.0x    The sharing was entirely the generic post-training
                   direction. Every delta lands in one region that is
                   simply safe to edit, and task identity carries
                   nothing. The navigator has no signal to predict.

  energy kept low  The task subspace was largely inside the control's,
                   so the residual comparison rests on little signal and
                   should not be over-read."""
    )


if __name__ == "__main__":
    main()
