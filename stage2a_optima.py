"""Stage 2a: inside the shared region, do math and code want different addresses?

Stage 1 established the map and ruled out the easy interpretations:

    math and coder deltas overlap 12.9x chance, only 3.4-4.0x against
    generic post-training, and 11.9x of that survives projecting the
    generic direction out. The region is real and task-family specific.

    But random directions inside it are worse than isotropic at finding
    large gains: 4/100 versus 23/100 above +0.3 on math. The region is
    low-sensitivity, not high-reward.

A region can be worth knowing even when random points in it are not.
"The restaurant is in San Francisco" narrows the search enormously and
still leaves you lost if you pick a random street corner.

So this asks the next question, and it asks it without a navigator:

    Optimise an address directly for math. Optimise one directly for
    code. Are they different addresses?

That is the oracle. Both searches get the task rewards handed to them,
so whatever they find is a ceiling on what any predictor could achieve.
Three outcomes, all informative:

    distinct optima      Different tasks want different coordinates in
                         one shared map. A navigator has a target, and
                         CROWN is worth building.

    identical optima     Same region, same best point. There is nothing
                         task-specific left to predict.

    neither beats random The region is intrinsically flat and no address
                         in it is worth finding.

Held-out tasks decide it. Each family's 16 tasks split 8 for the search
and 8 never seen by it, because an address tuned on 8 items and reported
on those same 8 items is the winner's curse that already fooled this
repository once.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mlx.core as mx

from dictionary import (
    DictionaryPerturbation,
    cosine,
    random_address,
)
from model import BaseModel
from perturb import all_layer_indices
from stage1_density import (
    CODE_ITEMS,
    MATH_ITEMS,
    PROBE_ITEMS,
    SYSTEM_PROMPT,
    displacement,
    margin,
    probe_logits,
)
from subspace import CachedBasis, CheckpointDeltaSource

# The shared map. Math and coder bases overlap 12.9x, so either serves;
# using one fixed basis for both searches is what makes the comparison
# about the task rather than about the basis.
BASIS_SOURCE = "mlx-community/Qwen2.5-Math-7B-Instruct-4bit"
BASIS_RANK = 32

DICTIONARY_SIZE = 24
RANK = 4
TARGET_KL = 0.05

RANDOM_ADDRESSES = 30
SEARCH_BUDGET = 120

SPLIT = 8
RESULTS_PATH = Path("results/stage2a_optima.json")

SEARCH_ITEMS = {
    "math": MATH_ITEMS[:SPLIT],
    "code": CODE_ITEMS[:SPLIT],
}
HELD_OUT_ITEMS = {
    "math": MATH_ITEMS[SPLIT:],
    "code": CODE_ITEMS[SPLIT:],
}


# A cheap two-prompt probe used to hold every address at the same
# functional displacement during the search. Different directions at the
# same Frobenius norm move the model by wildly different amounts -- the
# first run of this script produced KLs from 0.019 to 0.32 against a
# target of 0.05 -- so without per-address calibration the search just
# rewards whichever direction happens to touch the model least.
# Balanced across families. Calibrating on math prompts alone left the
# achieved displacement varying 2.2x across addresses when measured on
# the full probe, because code prompts are far more sensitive: an
# address "held at" the target on math was nowhere near it on code.
KL_PROBE_INDICES = (0, 1, 4, 5)
KL_PROBE = [PROBE_ITEMS[i] for i in KL_PROBE_INDICES]


def probe_kl(
    model: BaseModel,
    reference: list[mx.array],
) -> float:
    total = 0.0

    for item, base in zip(KL_PROBE, reference):
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

    return total / len(KL_PROBE)


def probe_reference(model: BaseModel) -> list[mx.array]:
    return [
        model.next_token_logits(
            item.task.prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        for item in KL_PROBE
    ]


def fit_kl_exponent(
    model: BaseModel,
    perturbation: DictionaryPerturbation,
    reference: list[mx.array],
    *,
    scale: float,
) -> float:
    """Rough starting slope for the per-address secant fit.

    Only a seed. The real slope is fitted per address inside
    `place_at_kl`, because it varies far too much between addresses for
    one number to serve.
    """

    address = random_address(DICTIONARY_SIZE, seed=31337)
    # A wide bracket. Too narrow and the two KLs differ by little more
    # than probe noise, which threw the fit outside the plausible range.
    low_scale, high_scale = scale * 0.4, scale * 2.2

    perturbation.set_address(address, scale=low_scale)
    low_kl = probe_kl(model, reference)
    perturbation.set_address(address, scale=high_scale)
    high_kl = probe_kl(model, reference)

    if low_kl <= 0 or high_kl <= low_kl:
        return 2.0

    exponent = math.log(high_kl / low_kl) / math.log(
        high_scale / low_scale
    )

    # KL should grow faster than linearly with perturbation size. A fit
    # outside this range means the probe is not measuring what it thinks
    # it is, and using it would amplify every correction instead of
    # damping it.
    if not 1.2 <= exponent <= 5.0:
        print(
            f"  WARNING: fitted KL exponent {exponent:.2f} is "
            "implausible; falling back to 2.0"
        )
        return 2.0

    return exponent


def place_at_kl(
    model: BaseModel,
    perturbation: DictionaryPerturbation,
    address: list[float],
    reference: list[mx.array],
    *,
    scale: float,
    target_kl: float,
    exponent: float,
    iterations: int = 4,
    tolerance: float = 0.12,
) -> tuple[float, float]:
    """Put one address at the target displacement, not a fixed norm.

    Two correction steps rather than one. A single step leaves a ~2.5x
    spread in achieved KL when the fitted exponent is imperfect, and
    stage 1 showed a 2.9x difference in displacement is enough to change
    which arm looks better -- so the residual has to be smaller than the
    effect being measured.
    """

    current = scale
    perturbation.set_address(address, scale=current)
    measured = probe_kl(model, reference)
    local = exponent

    for _ in range(iterations):
        if measured <= 0.0:
            return current, measured

        # Stop as soon as this address is close enough. Most converge in
        # two steps; only the awkward ones pay for more.
        if abs(measured - target_kl) / target_kl <= tolerance:
            break

        previous_scale, previous_kl = current, measured

        current = current * (target_kl / measured) ** (1.0 / local)
        current = min(max(current, 1e-4), 1.0)
        perturbation.set_address(address, scale=current)
        measured = probe_kl(model, reference)

        # Refit the slope from this address's own two points. A single
        # global exponent cannot work: measured across addresses the
        # log-log slope runs from 0.87 to 3.35, and KL at one fixed
        # Frobenius norm varies 11x between addresses. The curve is
        # monotonic though, so a secant fit from the pair just observed
        # converges where a shared constant does not.
        step = math.log(current / previous_scale)

        if measured > 0.0 and abs(step) > 1e-6:
            fitted = math.log(measured / previous_kl) / step

            if 0.8 <= fitted <= 5.0:
                local = fitted

    return current, measured


def score(
    model: BaseModel,
    family: str,
    items,
) -> float:
    return margin(model, items)


def match_scale(
    model: BaseModel,
    perturbation: DictionaryPerturbation,
    base_logits: list[mx.array],
    *,
    target_kl: float,
    steps: int = 10,
) -> tuple[float, float]:
    """Calibrate distance so addresses are compared at equal effect."""

    probe = random_address(DICTIONARY_SIZE, seed=4242)
    low, high = 0.002, 0.5

    for _ in range(steps):
        middle = (low * high) ** 0.5
        perturbation.set_address(probe, scale=middle)

        if displacement(model, base_logits) < target_kl:
            low = middle
        else:
            high = middle

    scale = (low * high) ** 0.5
    perturbation.set_address(probe, scale=scale)

    return scale, displacement(model, base_logits)


def optimise_address(
    model: BaseModel,
    perturbation: DictionaryPerturbation,
    family: str,
    reference: list[mx.array],
    *,
    scale: float,
    budget: int,
    seed: int,
    exponent: float,
) -> tuple[list[float], float, list[float], list[float]]:
    """(1+1) evolution strategy over the D-dimensional address.

    A plain hill climber with the one-fifth success rule. The search
    space is a direction in R^D, which is small enough that this beats
    anything fancier at a few hundred evaluations.
    """

    items = SEARCH_ITEMS[family]
    address = random_address(DICTIONARY_SIZE, seed=seed)

    place_at_kl(
        model, perturbation, address, reference,
        scale=scale, target_kl=TARGET_KL, exponent=exponent,
    )
    best = score(model, family, items)

    sigma = 0.35
    successes = 0
    window = 10
    history = [best]
    achieved_kl: list[float] = []

    for step in range(budget):
        noise = random_address(
            DICTIONARY_SIZE,
            seed=seed * 100_003 + step,
        )
        candidate = [
            value + sigma * shift
            for value, shift in zip(address, noise)
        ]

        # Each candidate is placed at the same displacement, so the
        # search compares directions rather than perturbation strength.
        _, measured = place_at_kl(
            model, perturbation, candidate, reference,
            scale=scale, target_kl=TARGET_KL, exponent=exponent,
        )
        achieved_kl.append(measured)
        value = score(model, family, items)

        if value > best:
            address, best = candidate, value
            successes += 1

        history.append(best)

        # One-fifth rule: widen the step while it keeps paying off,
        # narrow it when it stops.
        if (step + 1) % window == 0:
            if successes > window / 5:
                sigma *= 1.5
            else:
                sigma /= 1.5

            sigma = min(max(sigma, 0.02), 1.5)
            successes = 0

        if (step + 1) % 25 == 0:
            print(
                f"    {family} step {step + 1:>3}/{budget}: "
                f"search margin {best:+.4f}, sigma {sigma:.3f}"
            )

    return address, best, history, achieved_kl


def evaluate_everywhere(
    model: BaseModel,
    perturbation: DictionaryPerturbation,
    address: list[float],
    reference: list[mx.array],
    *,
    scale: float,
    base_logits: list[mx.array],
    exponent: float,
) -> dict[str, float]:
    place_at_kl(
        model, perturbation, address, reference,
        scale=scale, target_kl=TARGET_KL, exponent=exponent,
    )

    return {
        "math_search": score(model, "math", SEARCH_ITEMS["math"]),
        "code_search": score(model, "code", SEARCH_ITEMS["code"]),
        "math_held_out": score(model, "math", HELD_OUT_ITEMS["math"]),
        "code_held_out": score(model, "code", HELD_OUT_ITEMS["code"]),
        # The broad probe is the honest check: the search only ever saw
        # the four calibration prompts, so agreement here is evidence
        # the addresses really are at equal displacement.
        "kl": displacement(model, base_logits),
        "kl_calibration_probe": probe_kl(model, reference),
    }


def main() -> None:
    print("Loading the base model")
    model = BaseModel()
    layer_indices = all_layer_indices(model)

    print(f"Building the shared basis from {BASIS_SOURCE}")
    source = CheckpointDeltaSource(BASIS_SOURCE)
    provider = CachedBasis(source, basis_rank=BASIS_RANK)

    base_logits = probe_logits(model)
    # Sliced from the unperturbed logits. Capturing this after a
    # perturbation is installed silently measures every later KL against
    # an already-moved model, which is exactly what broke the first run.
    reference = [base_logits[i] for i in KL_PROBE_INDICES]
    base_scores = {
        "math_search": score(model, "math", SEARCH_ITEMS["math"]),
        "code_search": score(model, "code", SEARCH_ITEMS["code"]),
        "math_held_out": score(model, "math", HELD_OUT_ITEMS["math"]),
        "code_held_out": score(model, "code", HELD_OUT_ITEMS["code"]),
    }
    print(
        "\nBase margins: "
        f"math held-out {base_scores['math_held_out']:+.3f}, "
        f"code held-out {base_scores['code_held_out']:+.3f}"
    )

    perturbation = DictionaryPerturbation(
        model,
        provider,
        layer_indices=layer_indices,
        dictionary_size=DICTIONARY_SIZE,
        rank=RANK,
    )

    with perturbation.active():
        source.release()
        print(f"  {provider.energy_report()}")
        print(
            f"  dictionary of {DICTIONARY_SIZE} directions, "
            f"rank {RANK} each; an address is "
            f"{DICTIONARY_SIZE} numbers"
        )

        scale, kl = match_scale(
            model,
            perturbation,
            base_logits,
            target_kl=TARGET_KL,
        )
        print(f"  scale {scale:.4f} reaches KL {kl:.4f}")

        exponent = fit_kl_exponent(
            model,
            perturbation,
            reference,
            scale=scale,
        )
        print(
            f"  KL grows as scale^{exponent:.2f}; every address is "
            "re-scaled to the target before scoring"
        )

        started = time.time()
        perturbation.set_address(
            random_address(DICTIONARY_SIZE, seed=0),
            scale=scale,
        )
        score(model, "math", SEARCH_ITEMS["math"])
        per_eval = time.time() - started
        print(
            f"  {per_eval:.1f}s per evaluation; estimated total "
            f"{per_eval * (RANDOM_ADDRESSES * 2 + SEARCH_BUDGET * 2) / 60:.0f} min"
        )

        # --- baseline: how good is the best of N random addresses? ----
        print(
            f"\nSampling {RANDOM_ADDRESSES} random addresses "
            "(the thing stage 1 did)"
        )
        random_best = {"math": None, "code": None}

        for offset, family in enumerate(("math", "code")):
            best_value = None
            best_address = None

            for index in range(RANDOM_ADDRESSES):
                address = random_address(
                    DICTIONARY_SIZE,
                    seed=90_000 + offset * 5_000 + index,
                )
                place_at_kl(
                    model, perturbation, address, reference,
                    scale=scale, target_kl=TARGET_KL,
                    exponent=exponent,
                )
                value = score(model, family, SEARCH_ITEMS[family])

                if best_value is None or value > best_value:
                    best_value, best_address = value, address

            random_best[family] = best_address
            print(
                f"  best random {family} address: "
                f"search margin {best_value:+.4f} "
                f"(base {base_scores[family + '_search']:+.4f})"
            )

        # --- the oracle searches --------------------------------------
        print(f"\nOptimising an address for each family "
              f"({SEARCH_BUDGET} evaluations each)")
        addresses: dict[str, list[float]] = {}
        histories: dict[str, list[float]] = {}

        kl_spread: dict[str, list[float]] = {}

        for family in ("math", "code"):
            address, best, history, achieved = optimise_address(
                model,
                perturbation,
                family,
                reference,
                scale=scale,
                budget=SEARCH_BUDGET,
                seed=11 if family == "math" else 23,
                exponent=exponent,
            )
            addresses[family] = address
            histories[family] = history
            kl_spread[family] = achieved
            low, high = min(achieved), max(achieved)
            print(
                f"  {family}: search margin "
                f"{base_scores[family + '_search']:+.4f} -> {best:+.4f}"
                f"   (candidate KL stayed in {low:.4f}-{high:.4f})"
            )

        # --- cross-evaluation ------------------------------------------
        report = {
            name: evaluate_everywhere(
                model, perturbation, value, reference,
                scale=scale, base_logits=base_logits,
                exponent=exponent,
            )
            for name, value in (
                ("optimised for math", addresses["math"]),
                ("optimised for code", addresses["code"]),
                ("best random (math)", random_best["math"]),
                ("best random (code)", random_best["code"]),
            )
        }

    similarity = cosine(addresses["math"], addresses["code"])

    print("\n" + "=" * 66)
    print("Held-out margin change vs the base model")
    print("=" * 66)
    print(
        f"\n  {'address':>22} {'math':>10} {'code':>10} {'KL':>8}"
    )

    for name, values in report.items():
        print(
            f"  {name:>22} "
            f"{values['math_held_out'] - base_scores['math_held_out']:>+10.4f} "
            f"{values['code_held_out'] - base_scores['code_held_out']:>+10.4f} "
            f"{values['kl']:>8.4f}"
        )

    math_row = report["optimised for math"]
    code_row = report["optimised for code"]
    math_gap = (
        math_row["math_held_out"] - code_row["math_held_out"]
    )
    code_gap = (
        code_row["code_held_out"] - math_row["code_held_out"]
    )

    print(
        f"\n  cosine(address_math, address_code) = {similarity:+.3f}"
        f"   (0 = unrelated, 1 = identical)"
    )
    print(
        f"  math address beats code address on math by {math_gap:+.4f}"
    )
    print(
        f"  code address beats math address on code by {code_gap:+.4f}"
    )

    print(
        """
Reading it:

  both gaps positive, cosine low
      Different tasks want different coordinates in one shared map.
      That is the result CROWN needs, and stage 2b is to predict those
      coordinates from task evidence instead of searching for them.

  gaps near zero, cosine high
      Both searches walked to the same place. The map has one good
      address, not a task-specific one, and there is nothing for a
      navigator to learn.

  neither address beats the best random one
      150 evaluations found nothing a handful of random draws did not.
      The region is flat and stage 2b has no foundation.

  held-out much worse than search
      The address overfitted 8 tasks. Enlarge the split before reading
      anything else in this table."""
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "basis_source": BASIS_SOURCE,
                "basis_rank": BASIS_RANK,
                "dictionary_size": DICTIONARY_SIZE,
                "rank": RANK,
                "target_kl": TARGET_KL,
                "scale": scale,
                "search_budget": SEARCH_BUDGET,
                "random_addresses": RANDOM_ADDRESSES,
                "split": SPLIT,
                "base": base_scores,
                "addresses": addresses,
                "cosine": similarity,
                "kl_exponent": exponent,
                "candidate_kl_spread": {
                    family: {
                        "min": min(values),
                        "max": max(values),
                    }
                    for family, values in kl_spread.items()
                },
                "report": report,
                "histories": histories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
