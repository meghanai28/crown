from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model import BaseModel
from perturb import (
    DEFAULT_MODULE_PATHS,
    LocalPerturbation,
    PerturbationConfig,
)
from scoring import (
    ScoreResult,
    normalize_code_value,
    score_prediction,
)
from tasks import (
    CODE_DEV_TASKS,
    CODE_TEST_TASKS,
    MATH_DEV_TASKS,
    MATH_TEST_TASKS,
    TASK_SUITE_VERSION,
    AtomicTask,
)


# One seed defines one direction. Reusing that seed at every scale
# measures several distances along the same direction.
DIRECTION_SEEDS = (0, 1, 2, 3)
SCALES = (0.001, 0.003, 0.01, 0.03)

# None asks perturb.py to choose four depth-spanning layers.
LAYER_INDICES: tuple[int, ...] | None = None
MODULE_PATHS = DEFAULT_MODULE_PATHS
RANK = 4
MAX_TOKENS = 196

# Math already has little headroom. Start by searching the weaker
# code family, then enable math after the pipeline is validated.
SEARCH_MATH = False
SEARCH_CODE = True

RESULTS_PATH = Path("results/selected_specialists.json")

EVALUATION_SYSTEM_PROMPT = (
    "Solve the task carefully. Keep any reasoning brief. "
    "Your final line must contain only the final value inside "
    "<answer> and </answer>. Do not write anything after the "
    "closing tag."
)


@dataclass(frozen=True)
class CandidateScore:
    seed: int | None
    scale: float | None
    results: tuple[ScoreResult, ...]
    predictions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError(
                "CandidateScore needs at least one result"
            )

        if len(self.results) != len(self.predictions):
            raise ValueError(
                "results and predictions must have equal length"
            )

        if self.seed is None and self.scale is not None:
            raise ValueError(
                "a base score cannot have a scale"
            )

        if self.seed is not None and self.scale is None:
            raise ValueError(
                "a perturbation must have a scale"
            )

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def semantic_correct(self) -> int:
        return sum(
            result.semantic_correct
            for result in self.results
        )

    @property
    def value_correct(self) -> int:
        return sum(
            result.value_correct
            for result in self.results
        )

    @property
    def strict_correct(self) -> int:
        return sum(
            result.strict_correct
            for result in self.results
        )

    @property
    def parsed(self) -> int:
        return sum(
            result.parsed
            for result in self.results
        )

    @property
    def format_valid(self) -> int:
        return sum(
            result.format_valid
            for result in self.results
        )

    @property
    def semantic_accuracy(self) -> float:
        return self.semantic_correct / self.total

    @property
    def value_accuracy(self) -> float:
        return self.value_correct / self.total

    @property
    def strict_accuracy(self) -> float:
        return self.strict_correct / self.total

    @property
    def label(self) -> str:
        if self.seed is None:
            return "base"

        return (
            f"scale {self.scale:g}, "
            f"direction {self.seed}"
        )


def _prediction_preview(
    prediction: str,
    limit: int = 260,
) -> str:
    one_line = " ".join(prediction.split())

    if len(one_line) <= limit:
        return one_line

    return one_line[: limit - 3] + "..."


def answer_key(
    task: AtomicTask,
    result: ScoreResult,
) -> tuple[str, str]:
    answer = result.extracted_answer

    if answer is None:
        return "missing", ""

    if task.family == "math":
        return "math", answer

    value_type, value = normalize_code_value(answer)
    return value_type, repr(value)


def count_changed_answers(
    tasks: tuple[AtomicTask, ...],
    base_score: CandidateScore,
    candidate_score: CandidateScore,
) -> int:
    return sum(
        answer_key(task, base_result)
        != answer_key(task, candidate_result)
        for task, base_result, candidate_result in zip(
            tasks,
            base_score.results,
            candidate_score.results,
        )
    )


def count_changed_predictions(
    base_score: CandidateScore,
    candidate_score: CandidateScore,
) -> int:
    return sum(
        base_prediction != candidate_prediction
        for base_prediction, candidate_prediction in zip(
            base_score.predictions,
            candidate_score.predictions,
        )
    )


def print_answer_changes(
    tasks: tuple[AtomicTask, ...],
    base_score: CandidateScore,
    candidate_score: CandidateScore,
    *,
    header: str = "Extracted answers changed by winner:",
) -> None:
    print(f"\n  {header}")
    found_change = False

    for task, base_result, candidate_result in zip(
        tasks,
        base_score.results,
        candidate_score.results,
    ):
        if (
            answer_key(task, base_result)
            == answer_key(task, candidate_result)
        ):
            continue

        found_change = True
        print(f"    {task.task_id}")
        print(
            "      base:      "
            f"{base_result.extracted_answer!r}"
        )
        print(
            "      candidate: "
            f"{candidate_result.extracted_answer!r}"
        )

    if not found_change:
        print("    none")


def print_transitions(
    tasks: tuple[AtomicTask, ...],
    base_score: CandidateScore,
    candidate_score: CandidateScore,
    *,
    header: str,
) -> None:
    """Show which items changed correctness, not just how many.

    Two runs can post the same score while disagreeing on which items
    they solve. Without this, an unchanged total looks like an
    unchanged model.
    """

    print(f"\n  {header}")
    gains = 0
    losses = 0

    for task, base_result, candidate_result in zip(
        tasks,
        base_score.results,
        candidate_score.results,
    ):
        if (
            base_result.semantic_correct
            == candidate_result.semantic_correct
        ):
            continue

        if candidate_result.semantic_correct:
            gains += 1
            transition = "FAIL -> PASS"
        else:
            losses += 1
            transition = "PASS -> FAIL"

        print(f"    {task.task_id}: {transition}")
        print(
            "      expected:  "
            f"{task.expected_answer!r}"
        )
        print(
            "      base:      "
            f"{base_result.extracted_answer!r}"
        )
        print(
            "      candidate: "
            f"{candidate_result.extracted_answer!r}"
        )

    if not gains and not losses:
        print("    no item changed correctness")

    agreed = sum(
        base_result.semantic_correct
        and candidate_result.semantic_correct
        for base_result, candidate_result in zip(
            base_score.results,
            candidate_score.results,
        )
    )

    print(
        f"    gains {gains}, losses {losses}, "
        f"net {gains - losses:+d}; "
        f"{agreed}/{base_score.total} solved by both"
    )


def evaluate_current_model(
    model: BaseModel,
    tasks: tuple[AtomicTask, ...],
    *,
    seed: int | None,
    scale: float | None,
    show_details: bool,
) -> CandidateScore:
    predictions: list[str] = []
    results: list[ScoreResult] = []

    for task in tasks:
        prediction = model.generate(
            task.prompt,
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            max_tokens=MAX_TOKENS,
        )
        result = score_prediction(
            family=task.family,
            expected=task.expected_answer,
            prediction=prediction,
        )

        predictions.append(prediction)
        results.append(result)

        if show_details:
            status = (
                "PASS"
                if result.semantic_correct
                else "FAIL"
            )
            strict_status = (
                "yes"
                if result.strict_correct
                else "no"
            )

            print(f"    {task.task_id}: {status}")
            print(
                f"      expected:  "
                f"{task.expected_answer!r}"
            )
            print(
                f"      extracted: "
                f"{result.extracted_answer!r}"
            )
            print(
                f"      source: {result.source}; "
                f"strict: {strict_status}"
            )

            if not result.semantic_correct:
                print(
                    "      raw: "
                    f"{_prediction_preview(prediction)!r}"
                )

    return CandidateScore(
        seed=seed,
        scale=scale,
        results=tuple(results),
        predictions=tuple(predictions),
    )


def evaluate_direction(
    model: BaseModel,
    tasks: tuple[AtomicTask, ...],
    *,
    seed: int,
    scale: float,
    show_details: bool = False,
) -> CandidateScore:
    config = PerturbationConfig(
        layer_indices=LAYER_INDICES,
        module_paths=MODULE_PATHS,
        rank=RANK,
        scale=scale,
        seed=seed,
    )
    perturbation = LocalPerturbation(
        model,
        config,
    )

    with perturbation.active():
        return evaluate_current_model(
            model,
            tasks,
            seed=seed,
            scale=scale,
            show_details=show_details,
        )


def print_score(
    score: CandidateScore,
    *,
    answer_changes: int | None = None,
    text_changes: int | None = None,
) -> None:
    message = (
        f"  {score.label}: "
        f"semantic {score.semantic_correct}/{score.total} "
        f"({score.semantic_accuracy:.1%}), "
        f"value {score.value_correct}/{score.total}, "
        f"strict {score.strict_correct}/{score.total}, "
        f"tags {score.format_valid}/{score.total}"
    )

    if answer_changes is not None:
        message += (
            f", answer changes "
            f"{answer_changes}/{score.total}"
        )

    if text_changes is not None:
        message += (
            f", text changes "
            f"{text_changes}/{score.total}"
        )

    print(message)


def _candidate_is_better(
    candidate: CandidateScore,
    best: CandidateScore,
) -> bool:
    # A perturbation must add semantic task capability. Better
    # formatting alone is not enough to call it a specialist.
    if candidate.semantic_correct > best.semantic_correct:
        return True

    # Once a perturbation has beaten the base, use exact value count
    # and then strict protocol count to break ties among specialists.
    if (
        best.seed is not None
        and candidate.semantic_correct
        == best.semantic_correct
    ):
        return (
            candidate.value_correct,
            candidate.strict_correct,
        ) > (
            best.value_correct,
            best.strict_correct,
        )

    return False


def search_candidates(
    model: BaseModel,
    family: str,
    dev_tasks: tuple[AtomicTask, ...],
) -> tuple[CandidateScore, CandidateScore]:
    print(f"\nSearching {family} candidates")

    base_score = evaluate_current_model(
        model,
        dev_tasks,
        seed=None,
        scale=None,
        show_details=True,
    )
    print_score(base_score)

    best_score = base_score

    for scale in SCALES:
        print(f"\n  Testing scale {scale:g}")

        for seed in DIRECTION_SEEDS:
            candidate_score = evaluate_direction(
                model,
                dev_tasks,
                seed=seed,
                scale=scale,
            )
            answer_changes = count_changed_answers(
                dev_tasks,
                base_score,
                candidate_score,
            )
            text_changes = count_changed_predictions(
                base_score,
                candidate_score,
            )

            print_score(
                candidate_score,
                answer_changes=answer_changes,
                text_changes=text_changes,
            )

            if _candidate_is_better(
                candidate_score,
                best_score,
            ):
                best_score = candidate_score
                print("    New best candidate")

    if best_score.seed is None:
        print(
            f"\n  No {family} direction "
            "beat the base semantically."
        )
    else:
        print(
            f"\n  Selected {family} "
            f"{best_score.label}"
        )
        print_answer_changes(
            dev_tasks,
            base_score,
            best_score,
        )

    return base_score, best_score


def evaluate_base_only(
    model: BaseModel,
    family: str,
    dev_tasks: tuple[AtomicTask, ...],
) -> tuple[CandidateScore, CandidateScore]:
    print(
        f"\nEvaluating {family} base only; "
        "perturbation search is disabled"
    )
    base_score = evaluate_current_model(
        model,
        dev_tasks,
        seed=None,
        scale=None,
        show_details=True,
    )
    print_score(base_score)
    return base_score, base_score


def select_or_use_base(
    model: BaseModel,
    family: str,
    dev_tasks: tuple[AtomicTask, ...],
    *,
    search_enabled: bool,
) -> tuple[CandidateScore, CandidateScore]:
    if search_enabled:
        return search_candidates(
            model,
            family,
            dev_tasks,
        )

    return evaluate_base_only(
        model,
        family,
        dev_tasks,
    )


def evaluate_selected_model(
    model: BaseModel,
    tasks: tuple[AtomicTask, ...],
    selected_dev: CandidateScore,
    base_test: CandidateScore,
) -> CandidateScore:
    if selected_dev.seed is None:
        return base_test

    if selected_dev.scale is None:
        raise RuntimeError(
            "selected perturbation has no scale"
        )

    print(
        "\n  Re-running the same held-out tasks under "
        f"{selected_dev.label}"
    )

    return evaluate_direction(
        model,
        tasks,
        seed=selected_dev.seed,
        scale=selected_dev.scale,
        show_details=True,
    )


def selection_outcome(
    selected_dev: CandidateScore,
    *,
    search_enabled: bool,
) -> str:
    if not search_enabled:
        return "search_disabled"

    if selected_dev.seed is None:
        return "no_direction_beat_base"

    return "direction_selected"


def evaluate_test_split(
    model: BaseModel,
    family: str,
    test_tasks: tuple[AtomicTask, ...],
    selected_dev: CandidateScore,
) -> tuple[CandidateScore, CandidateScore]:
    print(f"\nEvaluating {family} test tasks")

    base_test = evaluate_current_model(
        model,
        test_tasks,
        seed=None,
        scale=None,
        show_details=True,
    )
    print_score(base_test)

    selected_test = evaluate_selected_model(
        model,
        test_tasks,
        selected_dev,
        base_test,
    )

    if selected_test is not base_test:
        print_score(
            selected_test,
            answer_changes=count_changed_answers(
                test_tasks,
                base_test,
                selected_test,
            ),
            text_changes=count_changed_predictions(
                base_test,
                selected_test,
            ),
        )
        print_transitions(
            test_tasks,
            base_test,
            selected_test,
            header=(
                "Held-out correctness transitions "
                "(base -> selected):"
            ),
        )
        print_answer_changes(
            test_tasks,
            base_test,
            selected_test,
            header=(
                "Extracted answers changed on held-out tasks:"
            ),
        )

    return base_test, selected_test


def print_final_result(
    family: str,
    base_dev: CandidateScore,
    selected_dev: CandidateScore,
    base_test: CandidateScore,
    selected_test: CandidateScore,
    *,
    outcome: str,
) -> None:
    # A base-only family has no before-and-after to report. Printing
    # one anyway reads like a search that found nothing, which is a
    # different and much stronger claim.
    if outcome != "direction_selected":
        reason = (
            "no search was run"
            if outcome == "search_disabled"
            else (
                f"{len(SCALES) * len(DIRECTION_SEEDS)} "
                "directions searched, none beat the base on dev"
            )
        )
        print(f"  {family}: {reason}; base model only")
        print(
            "    semantic dev "
            f"{base_dev.semantic_accuracy:.1%}, test "
            f"{base_test.semantic_accuracy:.1%}"
        )
        print(
            "    strict   dev "
            f"{base_dev.strict_accuracy:.1%}, test "
            f"{base_test.strict_accuracy:.1%}"
        )
        return

    print(
        f"  {family} {selected_dev.label} "
        f"(best of {len(SCALES) * len(DIRECTION_SEEDS)}):"
    )
    print(
        "    semantic dev "
        f"{base_dev.semantic_accuracy:.1%} -> "
        f"{selected_dev.semantic_accuracy:.1%}, test "
        f"{base_test.semantic_accuracy:.1%} -> "
        f"{selected_test.semantic_accuracy:.1%}"
    )
    print(
        "    strict   dev "
        f"{base_dev.strict_accuracy:.1%} -> "
        f"{selected_dev.strict_accuracy:.1%}, test "
        f"{base_test.strict_accuracy:.1%} -> "
        f"{selected_test.strict_accuracy:.1%}"
    )
    print(
        "    dev is the selection set and is biased upward by "
        "the search; only the test column is unbiased."
    )


def per_task_records(
    tasks: tuple[AtomicTask, ...],
    score: CandidateScore,
) -> list[dict[str, object]]:
    """Keep the evidence behind each total for later analysis."""

    return [
        {
            "task_id": task.task_id,
            "category": task.category,
            "expected": task.expected_answer,
            "extracted": result.extracted_answer,
            "source": result.source,
            "format_valid": result.format_valid,
            "value_correct": result.value_correct,
            "semantic_correct": result.semantic_correct,
            "strict_correct": result.strict_correct,
            "prediction": prediction,
        }
        for task, result, prediction in zip(
            tasks,
            score.results,
            score.predictions,
        )
    ]


def score_summary(
    score: CandidateScore,
    tasks: tuple[AtomicTask, ...],
) -> dict[str, object]:
    return {
        "tasks": per_task_records(tasks, score),
        "semantic_correct": score.semantic_correct,
        "value_correct": score.value_correct,
        "strict_correct": score.strict_correct,
        "format_valid": score.format_valid,
        "total": score.total,
        "semantic_accuracy": score.semantic_accuracy,
        "value_accuracy": score.value_accuracy,
        "strict_accuracy": score.strict_accuracy,
    }


def main() -> None:
    print("Loading the base model")
    model = BaseModel()

    example_config = PerturbationConfig(
        layer_indices=LAYER_INDICES,
        module_paths=MODULE_PATHS,
        rank=RANK,
        scale=SCALES[0],
        seed=DIRECTION_SEEDS[0],
    )
    example_perturbation = LocalPerturbation(
        model,
        example_config,
    )
    resolved_layer_indices = (
        example_perturbation.layer_indices
    )

    print(
        "Perturbing layers:",
        resolved_layer_indices,
    )
    print(
        "Matrices per layer:",
        len(MODULE_PATHS),
    )

    math_base_dev, math_best_dev = select_or_use_base(
        model,
        "math",
        MATH_DEV_TASKS,
        search_enabled=SEARCH_MATH,
    )
    code_base_dev, code_best_dev = select_or_use_base(
        model,
        "code",
        CODE_DEV_TASKS,
        search_enabled=SEARCH_CODE,
    )

    math_base_test, math_selected_test = evaluate_test_split(
        model,
        "math",
        MATH_TEST_TASKS,
        math_best_dev,
    )
    code_base_test, code_selected_test = evaluate_test_split(
        model,
        "code",
        CODE_TEST_TASKS,
        code_best_dev,
    )

    math_outcome = selection_outcome(
        math_best_dev,
        search_enabled=SEARCH_MATH,
    )
    code_outcome = selection_outcome(
        code_best_dev,
        search_enabled=SEARCH_CODE,
    )

    print("\nFinal selection")
    print_final_result(
        "math",
        math_base_dev,
        math_best_dev,
        math_base_test,
        math_selected_test,
        outcome=math_outcome,
    )
    print_final_result(
        "code",
        code_base_dev,
        code_best_dev,
        code_base_test,
        code_selected_test,
        outcome=code_outcome,
    )

    results = {
        "task_suite": TASK_SUITE_VERSION,
        "selection_metric": "semantic_accuracy",
        "perturbation": {
            "normalization": "relative_frobenius",
            "layer_indices": list(
                resolved_layer_indices
            ),
            "module_paths": list(MODULE_PATHS),
            "rank": RANK,
            "scales_searched": list(SCALES),
            "direction_seeds": list(
                DIRECTION_SEEDS
            ),
        },
        "math_search_enabled": SEARCH_MATH,
        "code_search_enabled": SEARCH_CODE,
        "math": {
            "outcome": math_outcome,
            "selected_seed": math_best_dev.seed,
            "selected_scale": math_best_dev.scale,
            "base_dev": score_summary(
                math_base_dev,
                MATH_DEV_TASKS,
            ),
            "selected_dev": score_summary(
                math_best_dev,
                MATH_DEV_TASKS,
            ),
            "base_test": score_summary(
                math_base_test,
                MATH_TEST_TASKS,
            ),
            "selected_test": score_summary(
                math_selected_test,
                MATH_TEST_TASKS,
            ),
        },
        "code": {
            "outcome": code_outcome,
            "selected_seed": code_best_dev.seed,
            "selected_scale": code_best_dev.scale,
            "base_dev": score_summary(
                code_base_dev,
                CODE_DEV_TASKS,
            ),
            "selected_dev": score_summary(
                code_best_dev,
                CODE_DEV_TASKS,
            ),
            "base_test": score_summary(
                code_base_test,
                CODE_TEST_TASKS,
            ),
            "selected_test": score_summary(
                code_selected_test,
                CODE_TEST_TASKS,
            ),
        },
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()