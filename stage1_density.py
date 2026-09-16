"""Stage 1 smoke test: do task-specific subspaces carry task information?

One question, one table. For each prior, sample random directions and
count how often they improve math and how often they improve code:

                     math      code
    isotropic          ?         ?      <- control
    math-delta         ?         ?      <- should rise on math only
    coder-delta        ?         ?      <- should rise on code only

If the diagonal beats the off-diagonal, task-specific subspaces carry
task-specific information. If every row looks the same, they do not. If
both deltas lift both families, the prior is doing something generic and
the answer is still no.

What this is NOT
----------------
It is not adding a delta to the base model. That is task arithmetic, it
obviously works, and it would prove nothing. The delta's *values* are
never used. Only the subspace spanned by its top singular vectors is
used, and directions are drawn at random inside that subspace, as likely
to point away from the specialist checkpoint as toward it. The claim
under test is that the subspace is task-relevant, not that the delta is.

Why every arm gets its own scale
--------------------------------
Arms cannot be compared at equal weight change. Measured on this setup,
the same ||dW||/||W|| = 0.08 produced a next-token KL of 0.185 spread
isotropically and 0.543 confined to a rank-32 subspace: nearly three
times the effect on the model's output. Packing a fixed amount of weight
change into fewer directions hits the model harder. So each arm's scale
is calibrated to reach the same KL, and the arms are then compared at
equal effect on the model rather than equal effect on the weights.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from model import BaseModel
from perturb import (
    DEFAULT_MODULE_PATHS,
    LocalPerturbation,
    PerturbationConfig,
    all_layer_indices,
)
from scoring import normalize_code_value
from subspace import CachedBasis, CheckpointDeltaSource
from tasks import (
    CODE_DEV_TASKS,
    CODE_TEST_TASKS,
    MATH_DEV_TASKS,
    MATH_TEST_TASKS,
    TASK_SUITE_VERSION,
    AtomicTask,
)


# --- settings -------------------------------------------------------------
# Every arm is calibrated to this next-token KL from the base, so arms
# differ in where they point and not in how hard they push.
TARGET_KL = 0.05

SAMPLES_PER_ARM = 10
RANK = 16
BASIS_RANK = 32

ARMS: tuple[tuple[str, str | None], ...] = (
    ("isotropic", None),
    (
        "math-delta",
        "mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
    ),
    (
        "coder-delta",
        "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    ),
)

SYSTEM_PROMPT = "Solve the task carefully."
RESULTS_PATH = Path("results/stage1_density.json")


# --- the objective --------------------------------------------------------
# Two things are deliberately kept out of it.
#
# The answer tag is teacher-forced rather than generated. The earlier
# accuracy search was won by a direction that mostly made the model emit
# <answer> before running out of tokens, which scored as a capability
# gain. Forcing the tag keeps formatting out of the measurement.
#
# The number is a margin against a wrong answer, not a raw probability.
# A direction that only sharpens the output distribution raises the
# probability of any short string, which looks identical to knowing more
# answers. Subtracting a matched distractor cancels that.


@dataclass(frozen=True)
class ScoredItem:
    task: AtomicTask
    distractor: str


def _shape_class(task: AtomicTask, answer: str) -> str:
    if task.family == "math":
        return "int"

    value_type, _ = normalize_code_value(answer)
    return value_type


def build_items(
    tasks: tuple[AtomicTask, ...],
    paired: tuple[AtomicTask, ...],
) -> tuple[ScoredItem, ...]:
    """Pair each task with a wrong answer of the same shape.

    The same-category task in the other split is the natural distractor:
    same type, same format, similar length. A few categories share an
    answer across splits, and those fall back to the nearest task with a
    different answer of the same coarse type.
    """

    items: list[ScoredItem] = []

    for index, task in enumerate(tasks):
        target = _shape_class(task, task.expected_answer)
        distractor: str | None = None

        for offset in range(len(paired)):
            candidate = paired[
                (index + offset) % len(paired)
            ]

            if (
                candidate.expected_answer
                == task.expected_answer
            ):
                continue

            same_shape = (
                _shape_class(
                    candidate,
                    candidate.expected_answer,
                )
                == target
            )

            if offset == 0 or same_shape:
                distractor = candidate.expected_answer
                break

        if distractor is None:
            raise ValueError(
                f"{task.task_id} has no usable distractor"
            )

        items.append(ScoredItem(task, distractor))

    return tuple(items)


MATH_ITEMS = build_items(MATH_DEV_TASKS, MATH_TEST_TASKS)
CODE_ITEMS = build_items(CODE_DEV_TASKS, CODE_TEST_TASKS)

# Used only to measure how far a perturbation moved the model.
PROBE_ITEMS = MATH_ITEMS[:4] + CODE_ITEMS[:4]


def margin(
    model: BaseModel,
    items: tuple[ScoredItem, ...],
) -> float:
    """Mean of log P(correct) - log P(wrong) over one task family."""

    total = 0.0

    for item in items:
        correct, _ = model.answer_logprob(
            item.task.prompt,
            item.task.expected_answer,
            system_prompt=SYSTEM_PROMPT,
        )
        wrong, _ = model.answer_logprob(
            item.task.prompt,
            item.distractor,
            system_prompt=SYSTEM_PROMPT,
        )
        total += correct - wrong

    return total / len(items)


def margins(model: BaseModel) -> dict[str, float]:
    return {
        "math": margin(model, MATH_ITEMS),
        "code": margin(model, CODE_ITEMS),
    }


def probe_logits(model: BaseModel) -> list[mx.array]:
    return [
        model.next_token_logits(
            item.task.prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        for item in PROBE_ITEMS
    ]


def displacement(
    model: BaseModel,
    base_logits: list[mx.array],
) -> float:
    """Mean next-token KL from the base: how far the model moved."""

    total = 0.0

    for item, base in zip(PROBE_ITEMS, base_logits):
        current = model.next_token_logits(
            item.task.prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        base_log = base - mx.logsumexp(base)
        current_log = current - mx.logsumexp(current)
        kl = mx.sum(
            mx.exp(base_log) * (base_log - current_log)
        )
        mx.eval(kl)
        total += float(kl.item())

    return total / len(PROBE_ITEMS)


# --- sampling -------------------------------------------------------------


def perturbation_for(
    model: BaseModel,
    layer_indices: tuple[int, ...],
    provider: CachedBasis | None,
    *,
    scale: float,
    seed: int,
) -> LocalPerturbation:
    return LocalPerturbation(
        model,
        PerturbationConfig(
            layer_indices=layer_indices,
            module_paths=DEFAULT_MODULE_PATHS,
            rank=RANK,
            scale=scale,
            seed=seed,
            subspace_weight=(
                0.0 if provider is None else 1.0
            ),
        ),
        basis_provider=provider,
    )


def match_scale(
    model: BaseModel,
    layer_indices: tuple[int, ...],
    provider: CachedBasis | None,
    base_logits: list[mx.array],
    *,
    target_kl: float,
    steps: int = 10,
) -> tuple[float, float]:
    """Find the scale whose displacement reaches target_kl.

    KL rises monotonically with scale, so bisecting in log space
    converges in a handful of probes. This is what makes the arms
    comparable: they end up pushing the model equally hard and differ
    only in direction.
    """

    low, high = 0.002, 0.5

    for _ in range(steps):
        middle = (low * high) ** 0.5
        perturbation = perturbation_for(
            model,
            layer_indices,
            provider,
            scale=middle,
            seed=9999,
        )

        with perturbation.active():
            measured = displacement(model, base_logits)

        if measured < target_kl:
            low = middle
        else:
            high = middle

    scale = (low * high) ** 0.5
    perturbation = perturbation_for(
        model,
        layer_indices,
        provider,
        scale=scale,
        seed=9999,
    )

    with perturbation.active():
        return scale, displacement(model, base_logits)


def build_provider(source: str | None) -> CachedBasis | None:
    if source is None:
        return None

    return CachedBasis(
        CheckpointDeltaSource(source),
        basis_rank=BASIS_RANK,
    )


def prepare(
    model: BaseModel,
    layer_indices: tuple[int, ...],
    source: str | None,
) -> CachedBasis | None:
    """Build every basis once, report it, then free the checkpoint."""

    provider = build_provider(source)

    if provider is None:
        return None

    warm = perturbation_for(
        model,
        layer_indices,
        provider,
        scale=0.01,
        seed=0,
    )

    with warm.active():
        pass

    print(f"  {provider.energy_report()}")
    release = getattr(provider.source, "release", None)

    if release is not None:
        release()

    return provider


def main() -> None:
    print("Loading the base model")
    model = BaseModel()
    layer_indices = all_layer_indices(model)

    print(
        f"{model.config.model_id}: "
        f"{len(layer_indices)} layers x "
        f"{len(DEFAULT_MODULE_PATHS)} matrices, rank {RANK}"
    )
    print(
        f"{len(MATH_ITEMS)} math and {len(CODE_ITEMS)} code "
        f"tasks; every arm matched to KL {TARGET_KL:g}"
    )

    base_logits = probe_logits(model)

    started = time.time()
    base = margins(model)
    seconds = time.time() - started

    print(
        f"\nBase margin: math {base['math']:+.3f}, "
        f"code {base['code']:+.3f} "
        f"({seconds:.0f}s per sample)"
    )
    print(
        "Estimated run: "
        f"{seconds * len(ARMS) * SAMPLES_PER_ARM / 60:.0f} min"
    )

    results: dict[str, dict[str, object]] = {}

    for name, source in ARMS:
        print(f"\n--- {name} ---")

        provider = prepare(model, layer_indices, source)
        scale, measured_kl = match_scale(
            model,
            layer_indices,
            provider,
            base_logits,
            target_kl=TARGET_KL,
        )
        drift = abs(measured_kl - TARGET_KL) / TARGET_KL
        print(
            f"  scale {scale:.4f} reaches KL "
            f"{measured_kl:.4f} ({drift:.0%} off target)"
        )

        # Arms are only comparable while they sit at the same
        # displacement. A large miss means the comparison below is
        # measuring force as well as direction.
        if drift > 0.1:
            print(
                "  WARNING: this arm is not KL-matched to the "
                "others; raise `steps` in match_scale before "
                "comparing densities"
            )

        deltas: dict[str, list[float]] = {
            "math": [],
            "code": [],
        }

        for seed in range(SAMPLES_PER_ARM):
            perturbation = perturbation_for(
                model,
                layer_indices,
                provider,
                scale=scale,
                seed=seed,
            )

            with perturbation.active():
                sample = margins(model)

            for family in ("math", "code"):
                deltas[family].append(
                    sample[family] - base[family]
                )

            print(
                f"  seed {seed:>2}: "
                f"math {deltas['math'][-1]:>+7.3f}, "
                f"code {deltas['code'][-1]:>+7.3f}"
            )

        results[name] = {
            "source": source,
            "scale": scale,
            "kl": measured_kl,
            "deltas": deltas,
        }

    print("\n" + "=" * 62)
    print("How often a random direction improved each family")
    print("=" * 62)
    print(
        f"\n  {'arm':>12}  {'math':>13}  {'code':>13}  "
        f"{'KL':>7}"
    )

    for name, _ in ARMS:
        row = results[name]
        cells = []

        for family in ("math", "code"):
            values = row["deltas"][family]
            wins = sum(value > 0.0 for value in values)
            cells.append(
                f"{wins:>2}/{len(values):<2} "
                f"({wins / len(values):>4.0%})"
            )

        print(
            f"  {name:>12}  {cells[0]:>13}  "
            f"{cells[1]:>13}  {row['kl']:>7.4f}"
        )

    print("\n  mean margin change")
    print(f"  {'arm':>12}  {'math':>13}  {'code':>13}")

    for name, _ in ARMS:
        row = results[name]
        means = [
            statistics.fmean(row["deltas"][family])
            for family in ("math", "code")
        ]
        print(
            f"  {name:>12}  {means[0]:>+13.3f}  "
            f"{means[1]:>+13.3f}"
        )

    error = 50.0 / SAMPLES_PER_ARM**0.5
    print(
        f"""
Reading it: the hypothesis predicts math-delta beats isotropic on math
and not on code, and coder-delta the reverse. Flat rows mean subspace
geometry carries no task information. Both deltas winning everywhere
means the prior is doing something generic, which is also a negative.

With {SAMPLES_PER_ARM} samples the standard error on a density near 50% is about
{error:.0f} points, so only a large gap means anything. This is a smoke test of
the idea, not evidence for it."""
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "task_suite": TASK_SUITE_VERSION,
                "base_model_id": model.config.model_id,
                "objective": (
                    "mean over tasks of log P(correct) - "
                    "log P(matched distractor), answer tag "
                    "teacher-forced"
                ),
                "target_kl": TARGET_KL,
                "rank": RANK,
                "basis_rank": BASIS_RANK,
                "samples_per_arm": SAMPLES_PER_ARM,
                "base_margin": base,
                "arms": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
