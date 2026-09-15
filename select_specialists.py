from __future__ import annotations

import json
from dataclasses import dataclass

from model import BaseModel
from perturb import LocalPerturbation, PerturbationConfig
from scoring import (
    extract_explicit_answer,
    extract_math_answer,
    is_correct,
    normalize_code_value,
)
from tasks import (
    CODE_DEV_TASKS,
    CODE_TEST_TASKS,
    MATH_DEV_TASKS,
    MATH_TEST_TASKS,
    TASK_SUITE_VERSION,
    AtomicTask,
)


# Search settings
SCALES = (0.02, 0.05, 0.10, 0.20)
NUM_SEEDS_PER_SCALE = 8

LAYER_INDEX = 14
RANK = 4
MAX_TOKENS = 196

# Math is currently 94%, so there is almost no room
# for a math perturbation to improve.
SEARCH_MATH = False
SEARCH_CODE = True

EVALUATION_SYSTEM_PROMPT = (
    "Solve the task carefully using at most two short "
    "sentences of reasoning. "
    "Then put only the final value inside "
    "<answer> and </answer>."
)


@dataclass(frozen=True)
class CandidateScore:
    seed: int | None
    scale: float | None
    correct: int
    total: int
    extracted_answers: tuple[str | None, ...]
    predictions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(
                "CandidateScore total must be positive"
            )

        if len(self.extracted_answers) != self.total:
            raise ValueError(
                "extracted_answers length must equal total"
            )

        if len(self.predictions) != self.total:
            raise ValueError(
                "predictions length must equal total"
            )

        if self.seed is None and self.scale is not None:
            raise ValueError(
                "a base score cannot have a scale"
            )

        if self.seed is not None and self.scale is None:
            raise ValueError(
                "a perturbed score must have a scale"
            )

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def parsed(self) -> int:
        return sum(
            answer is not None
            for answer in self.extracted_answers
        )

    @property
    def label(self) -> str:
        if self.seed is None:
            return "base"

        return (
            f"scale {self.scale:.3f}, "
            f"seed {self.seed}"
        )


def extract_candidate_answer(
    task: AtomicTask,
    prediction: str,
) -> str | None:
    if task.family == "math":
        return extract_math_answer(prediction)

    if task.family == "code":
        return extract_explicit_answer(prediction)

    raise ValueError(
        f"unsupported task family: {task.family!r}"
    )


def answer_key(
    task: AtomicTask,
    answer: str | None,
) -> tuple[str, str]:
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
    changed = 0

    for task, base_answer, candidate_answer in zip(
        tasks,
        base_score.extracted_answers,
        candidate_score.extracted_answers,
    ):
        base_key = answer_key(
            task,
            base_answer,
        )
        candidate_key = answer_key(
            task,
            candidate_answer,
        )

        if base_key != candidate_key:
            changed += 1

    return changed


def print_answer_changes(
    tasks: tuple[AtomicTask, ...],
    base_score: CandidateScore,
    candidate_score: CandidateScore,
) -> None:
    print("\n  Answers changed by selected candidate:")

    found_change = False

    for task, base_answer, candidate_answer in zip(
        tasks,
        base_score.extracted_answers,
        candidate_score.extracted_answers,
    ):
        base_key = answer_key(
            task,
            base_answer,
        )
        candidate_key = answer_key(
            task,
            candidate_answer,
        )

        if base_key == candidate_key:
            continue

        found_change = True

        print(f"    {task.task_id}")
        print(f"      base: {base_answer!r}")
        print(
            f"      candidate: "
            f"{candidate_answer!r}"
        )

    if not found_change:
        print("    none")


def evaluate_current_model(
    model: BaseModel,
    tasks: tuple[AtomicTask, ...],
    *,
    seed: int | None,
    scale: float | None,
    show_details: bool,
) -> CandidateScore:
    correct = 0
    predictions: list[str] = []
    extracted_answers: list[str | None] = []

    for task in tasks:
        prediction = model.generate(
            task.prompt,
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            max_tokens=MAX_TOKENS,
        )

        extracted_answer = extract_candidate_answer(
            task,
            prediction,
        )

        prediction_matches = is_correct(
            family=task.family,
            expected=task.expected_answer,
            prediction=prediction,
        )

        predictions.append(prediction)
        extracted_answers.append(extracted_answer)

        if prediction_matches:
            correct += 1

        if show_details:
            status = (
                "PASS"
                if prediction_matches
                else "FAIL"
            )

            print(f"    {task.task_id}: {status}")
            print(
                f"      expected: "
                f"{task.expected_answer!r}"
            )
            print(
                f"      extracted: "
                f"{extracted_answer!r}"
            )

            if extracted_answer is None:
                print(
                    f"      raw prediction: "
                    f"{prediction!r}"
                )

    return CandidateScore(
        seed=seed,
        scale=scale,
        correct=correct,
        total=len(tasks),
        extracted_answers=tuple(extracted_answers),
        predictions=tuple(predictions),
    )


def evaluate_seed(
    model: BaseModel,
    tasks: tuple[AtomicTask, ...],
    *,
    seed: int,
    scale: float,
) -> CandidateScore:
    config = PerturbationConfig(
        layer_index=LAYER_INDEX,
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
            show_details=False,
        )


def print_score(
    score: CandidateScore,
    *,
    changed: int | None = None,
) -> None:
    message = (
        f"  {score.label}: "
        f"{score.correct}/{score.total} "
        f"({score.accuracy:.1%}), "
        f"parsed {score.parsed}/{score.total}"
    )

    if changed is not None:
        message += (
            f", changed {changed}/{score.total}"
        )

    print(message)


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

    # The base starts as the winner.
    best_score = base_score

    for scale in SCALES:
        print(f"\n  Testing scale {scale:.3f}")

        for seed in range(NUM_SEEDS_PER_SCALE):
            candidate_score = evaluate_seed(
                model,
                dev_tasks,
                seed=seed,
                scale=scale,
            )

            changed = count_changed_answers(
                dev_tasks,
                base_score,
                candidate_score,
            )

            print_score(
                candidate_score,
                changed=changed,
            )

            # A perturbation must strictly beat the
            # current winner.
            if (
                candidate_score.correct
                > best_score.correct
            ):
                best_score = candidate_score
                print("    New best candidate")

    if best_score.seed is None:
        print(
            f"\n  No {family} perturbation "
            "beat the base."
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
    selected_score: CandidateScore,
    base_test_score: CandidateScore,
) -> CandidateScore:
    if selected_score.seed is None:
        return base_test_score

    if selected_score.scale is None:
        raise RuntimeError(
            "selected perturbation has no scale"
        )

    return evaluate_seed(
        model,
        tasks,
        seed=selected_score.seed,
        scale=selected_score.scale,
    )


def print_final_result(
    family: str,
    base_dev: CandidateScore,
    selected_dev: CandidateScore,
    base_test: CandidateScore,
    selected_test: CandidateScore,
) -> None:
    print(
        f"  {family} {selected_dev.label}: "
        f"dev {base_dev.accuracy:.1%} "
        f"-> {selected_dev.accuracy:.1%}, "
        f"test {base_test.accuracy:.1%} "
        f"-> {selected_test.accuracy:.1%}"
    )


def main() -> None:
    print("Loading the base model")
    model = BaseModel()

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

    print("\nEvaluating math test tasks")

    math_base_test = evaluate_current_model(
        model,
        MATH_TEST_TASKS,
        seed=None,
        scale=None,
        show_details=True,
    )

    math_selected_test = evaluate_selected_model(
        model,
        MATH_TEST_TASKS,
        math_best_dev,
        math_base_test,
    )

    print("\nEvaluating code test tasks")

    code_base_test = evaluate_current_model(
        model,
        CODE_TEST_TASKS,
        seed=None,
        scale=None,
        show_details=True,
    )

    code_selected_test = evaluate_selected_model(
        model,
        CODE_TEST_TASKS,
        code_best_dev,
        code_base_test,
    )

    print("\nFinal selection")

    print_final_result(
        "math",
        math_base_dev,
        math_best_dev,
        math_base_test,
        math_selected_test,
    )

    print_final_result(
        "code",
        code_base_dev,
        code_best_dev,
        code_base_test,
        code_selected_test,
    )

    results = {
        "task_suite": TASK_SUITE_VERSION,
        "layer_index": LAYER_INDEX,
        "rank": RANK,
        "scales_searched": list(SCALES),
        "seeds_per_scale": NUM_SEEDS_PER_SCALE,
        "math_search_enabled": SEARCH_MATH,
        "code_search_enabled": SEARCH_CODE,
        "math": {
            "selected_seed": math_best_dev.seed,
            "selected_scale": math_best_dev.scale,
            "base_dev_accuracy": (
                math_base_dev.accuracy
            ),
            "selected_dev_accuracy": (
                math_best_dev.accuracy
            ),
            "base_test_accuracy": (
                math_base_test.accuracy
            ),
            "selected_test_accuracy": (
                math_selected_test.accuracy
            ),
        },
        "code": {
            "selected_seed": code_best_dev.seed,
            "selected_scale": code_best_dev.scale,
            "base_dev_accuracy": (
                code_base_dev.accuracy
            ),
            "selected_dev_accuracy": (
                code_best_dev.accuracy
            ),
            "base_test_accuracy": (
                code_base_test.accuracy
            ),
            "selected_test_accuracy": (
                code_selected_test.accuracy
            ),
        },
    }

    with open(
        "selected_specialists.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nSaved selected_specialists.json")


if __name__ == "__main__":
    main()