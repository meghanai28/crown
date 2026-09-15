from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


TASK_SUITE_VERSION = "crown_atomic_pilot_v2"

FINAL_ANSWER_INSTRUCTION = (
    "You may show reasoning before the final answer.\n"
    "End with exactly one final-answer field in this form:\n"
    "<answer>value</answer>\n"
    "Do not put reasoning inside the answer tags."
)


def build_math_prompt(question: str) -> str:
    return (
        "Solve the following mathematical problem exactly.\n"
        "The required answer is a single integer.\n\n"
        f"Problem:\n{question}\n\n"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )


def build_python_prompt(expression: str) -> str:
    return (
        "Evaluate the following expression using standard "
        "Python 3 semantics.\n"
        "Do not rewrite, approximate, or paraphrase the "
        "expression.\n\n"
        f"Expression:\n{expression}\n\n"
        "The result is judged exactly.\n"
        "For string results, preserve every character, "
        "including spaces, separators, hyphens, underscores, "
        "and punctuation.\n"
        "Put only the string's contents inside the answer tags, "
        "without adding quotation marks.\n"
        "For lists and numbers, use standard Python literal "
        "notation.\n\n"
        f"{FINAL_ANSWER_INSTRUCTION}"
    )


@dataclass(frozen=True)
class AtomicTask:
    task_id: str
    family: str
    category: str
    task_text: str
    expected_answer: str

    def __post_init__(self) -> None:
        if self.family not in {"math", "code"}:
            raise ValueError(
                "family must be either 'math' or 'code'"
            )

        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")

        if not self.category.strip():
            raise ValueError("category cannot be empty")

        if not self.task_text.strip():
            raise ValueError("task_text cannot be empty")

        if not self.expected_answer.strip():
            raise ValueError(
                "expected_answer cannot be empty"
            )

    @property
    def prompt(self) -> str:
        if self.family == "math":
            return build_math_prompt(self.task_text)

        return build_python_prompt(self.task_text)


@dataclass(frozen=True)
class CompositeTask:
    task_id: str
    math_question: str
    expected_intermediate: int
    code_expression_template: str
    expected_answer: str

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")

        if not self.math_question.strip():
            raise ValueError(
                "math_question cannot be empty"
            )

        if "{k}" not in self.code_expression_template:
            raise ValueError(
                "code_expression_template must contain {k}"
            )

        if not self.expected_answer.strip():
            raise ValueError(
                "expected_answer cannot be empty"
            )

    def stage_one_prompt(self) -> str:
        return build_math_prompt(self.math_question)

    def stage_two_prompt(
        self,
        intermediate: int,
    ) -> str:
        expression = self.code_expression_template.format(
            k=intermediate
        )

        return build_python_prompt(expression)

    def direct_prompt(self) -> str:
        expression = self.code_expression_template.format(
            k="k"
        )

        return (
            "Complete this two-stage task exactly.\n\n"
            "Stage 1: Solve the mathematical problem and call "
            "its integer answer k.\n"
            f"{self.math_question}\n\n"
            "Stage 2: Substitute that exact value of k into "
            "this Python expression and evaluate it:\n"
            f"{expression}\n\n"
            "The final Python result is judged exactly. "
            "For strings, preserve every character, including "
            "spaces and punctuation.\n\n"
            f"{FINAL_ANSWER_INSTRUCTION}"
        )


def math_task(
    task_id: str,
    category: str,
    question: str,
    expected_answer: str,
) -> AtomicTask:
    return AtomicTask(
        task_id=task_id,
        family="math",
        category=category,
        task_text=question,
        expected_answer=expected_answer,
    )


def code_task(
    task_id: str,
    category: str,
    expression: str,
    expected_answer: str,
) -> AtomicTask:
    return AtomicTask(
        task_id=task_id,
        family="code",
        category=category,
        task_text=expression,
        expected_answer=expected_answer,
    )


MATH_DEV_TASKS: tuple[AtomicTask, ...] = (
    math_task(
        "math_dev_01",
        "modular_arithmetic",
        "What is (23 × 17 + 5) modulo 11?",
        "0",
    ),
    math_task(
        "math_dev_02",
        "gcd",
        "What is the greatest common divisor of 96 and 144?",
        "48",
    ),
    math_task(
        "math_dev_03",
        "linear_equation",
        "Solve 7x - 9 = 40 for x.",
        "7",
    ),
    math_task(
        "math_dev_04",
        "floor_square_root",
        "What is the floor of the square root of 80?",
        "8",
    ),
    math_task(
        "math_dev_05",
        "order_operations",
        "Using standard order of operations, "
        "what is 18 + 6 × 4 - 9?",
        "33",
    ),
    math_task(
        "math_dev_06",
        "absolute_difference",
        "What is the absolute difference between 37 and 52?",
        "15",
    ),
    math_task(
        "math_dev_07",
        "integer_quotient",
        "What is the integer quotient when 103 is divided by 9?",
        "11",
    ),
    math_task(
        "math_dev_08",
        "lcm",
        "What is the least common multiple of 12 and 18?",
        "36",
    ),
    math_task(
        "math_dev_09",
        "modular_exponent",
        "What is 2 raised to the power 10, modulo 7?",
        "2",
    ),
    math_task(
        "math_dev_10",
        "integer_fraction",
        "What is three fourths of 28?",
        "21",
    ),
    math_task(
        "math_dev_11",
        "arithmetic_mean",
        "What is the arithmetic mean of 6, 8, 10, and 12?",
        "9",
    ),
    math_task(
        "math_dev_12",
        "arithmetic_series",
        "What is the sum of all integers from 1 through 20, "
        "inclusive?",
        "210",
    ),
    math_task(
        "math_dev_13",
        "percentage",
        "What is 15 percent of 240?",
        "36",
    ),
    math_task(
        "math_dev_14",
        "word_arithmetic",
        "Five boxes each contain 8 objects. "
        "After 7 objects are removed, how many remain?",
        "33",
    ),
    math_task(
        "math_dev_15",
        "combinations",
        "How many ways are there to choose 2 objects from "
        "8 distinct objects when order does not matter?",
        "28",
    ),
    math_task(
        "math_dev_16",
        "digit_sum",
        "What is the sum of the digits of 5837?",
        "23",
    ),
)


MATH_TEST_TASKS: tuple[AtomicTask, ...] = (
    math_task(
        "math_test_01",
        "modular_arithmetic",
        "What is (29 × 13 + 4) modulo 9?",
        "3",
    ),
    math_task(
        "math_test_02",
        "gcd",
        "What is the greatest common divisor of 126 and 210?",
        "42",
    ),
    math_task(
        "math_test_03",
        "linear_equation",
        "Solve 6x + 11 = 53 for x.",
        "7",
    ),
    math_task(
        "math_test_04",
        "floor_square_root",
        "What is the floor of the square root of 120?",
        "10",
    ),
    math_task(
        "math_test_05",
        "order_operations",
        "Using standard order of operations, "
        "what is 25 - 3 × 6 + 8?",
        "15",
    ),
    math_task(
        "math_test_06",
        "absolute_difference",
        "What is the absolute difference between 91 and 64?",
        "27",
    ),
    math_task(
        "math_test_07",
        "integer_quotient",
        "What is the integer quotient when 157 is divided by 12?",
        "13",
    ),
    math_task(
        "math_test_08",
        "lcm",
        "What is the least common multiple of 15 and 20?",
        "60",
    ),
    math_task(
        "math_test_09",
        "modular_exponent",
        "What is 3 raised to the power 7, modulo 10?",
        "7",
    ),
    math_task(
        "math_test_10",
        "integer_fraction",
        "What is five eighths of 48?",
        "30",
    ),
    math_task(
        "math_test_11",
        "arithmetic_mean",
        "What is the arithmetic mean of 5, 9, 13, and 17?",
        "11",
    ),
    math_task(
        "math_test_12",
        "arithmetic_series",
        "What is the sum of all integers from 1 through 25, "
        "inclusive?",
        "325",
    ),
    math_task(
        "math_test_13",
        "percentage",
        "What is 12.5 percent of 160?",
        "20",
    ),
    math_task(
        "math_test_14",
        "word_arithmetic",
        "Seven bags each contain 6 objects. "
        "After 9 objects are removed, how many remain?",
        "33",
    ),
    math_task(
        "math_test_15",
        "combinations",
        "How many ways are there to choose 2 objects from "
        "9 distinct objects when order does not matter?",
        "36",
    ),
    math_task(
        "math_test_16",
        "digit_sum",
        "What is the sum of the digits of 9468?",
        "27",
    ),
)


CODE_DEV_TASKS: tuple[AtomicTask, ...] = (
    code_task(
        "code_dev_01",
        "join",
        '"-".join(["ha"] * 3)',
        "ha-ha-ha",
    ),
    code_task(
        "code_dev_02",
        "slice",
        '"neural"[1:5]',
        "eura",
    ),
    code_task(
        "code_dev_03",
        "list_comprehension",
        "[i * i for i in range(5)]",
        "[0, 1, 4, 9, 16]",
    ),
    code_task(
        "code_dev_04",
        "sum_range",
        "sum(range(2, 8))",
        "27",
    ),
    code_task(
        "code_dev_05",
        "string_repetition",
        '"ab" * 3',
        "ababab",
    ),
    code_task(
        "code_dev_06",
        "negative_index",
        '"model"[-2]',
        "e",
    ),
    code_task(
        "code_dev_07",
        "step_slice",
        "[0, 1, 2, 3, 4, 5][::2]",
        "[0, 2, 4]",
    ),
    code_task(
        "code_dev_08",
        "split_length",
        'len("a,b,c".split(","))',
        "3",
    ),
    code_task(
        "code_dev_09",
        "sorting",
        "sorted([4, 1, 3, 2])",
        "[1, 2, 3, 4]",
    ),
    code_task(
        "code_dev_10",
        "replacement",
        '"banana".replace("a", "o")',
        "bonono",
    ),
    code_task(
        "code_dev_11",
        "tuple_index",
        "(10, 20, 30, 40)[2]",
        "30",
    ),
    code_task(
        "code_dev_12",
        "counting",
        "[1, 2, 1, 3, 1].count(1)",
        "3",
    ),
    code_task(
        "code_dev_13",
        "min_max",
        "max([7, 2, 11, 4]) - min([7, 2, 11, 4])",
        "9",
    ),
    code_task(
        "code_dev_14",
        "zip_comprehension",
        "[a + b for a, b in "
        "zip([1, 2, 3], [4, 5, 6])]",
        "[5, 7, 9]",
    ),
    code_task(
        "code_dev_15",
        "filtered_comprehension",
        "[x for x in range(8) if x % 2 == 0]",
        "[0, 2, 4, 6]",
    ),
    code_task(
        "code_dev_16",
        "string_methods",
        '"  hello  ".strip().upper()',
        "HELLO",
    ),
)


CODE_TEST_TASKS: tuple[AtomicTask, ...] = (
    code_task(
        "code_test_01",
        "join",
        '"|".join(["go"] * 4)',
        "go|go|go|go",
    ),
    code_task(
        "code_test_02",
        "slice",
        '"thicket"[2:6]',
        "icke",
    ),
    code_task(
        "code_test_03",
        "list_comprehension",
        "[i + 2 for i in range(4)]",
        "[2, 3, 4, 5]",
    ),
    code_task(
        "code_test_04",
        "sum_range",
        "sum(range(3, 9))",
        "33",
    ),
    code_task(
        "code_test_05",
        "string_repetition",
        '"xy" * 4',
        "xyxyxyxy",
    ),
    code_task(
        "code_test_06",
        "negative_index",
        '"python"[-3]',
        "h",
    ),
    code_task(
        "code_test_07",
        "step_slice",
        "[1, 2, 3, 4, 5, 6][1::2]",
        "[2, 4, 6]",
    ),
    code_task(
        "code_test_08",
        "split_length",
        'len("red blue green".split())',
        "3",
    ),
    code_task(
        "code_test_09",
        "sorting",
        "sorted([9, 5, 7, 6])",
        "[5, 6, 7, 9]",
    ),
    code_task(
        "code_test_10",
        "replacement",
        '"mississippi".replace("s", "x", 2)',
        "mixxissippi",
    ),
    code_task(
        "code_test_11",
        "tuple_index",
        "(3, 6, 9, 12)[-1]",
        "12",
    ),
    code_task(
        "code_test_12",
        "counting",
        '"abracadabra".count("a")',
        "5",
    ),
    code_task(
        "code_test_13",
        "min_max",
        "max([15, 8, 3, 12]) + min([15, 8, 3, 12])",
        "18",
    ),
    code_task(
        "code_test_14",
        "zip_comprehension",
        "[a * b for a, b in "
        "zip([2, 3, 4], [5, 6, 7])]",
        "[10, 18, 28]",
    ),
    code_task(
        "code_test_15",
        "filtered_comprehension",
        "[x for x in range(10) if x % 3 == 0]",
        "[0, 3, 6, 9]",
    ),
    code_task(
        "code_test_16",
        "string_methods",
        '"Neural Thickets".lower().replace(" ", "_")',
        "neural_thickets",
    ),
)


COMPOSITE_TEST_TASKS: tuple[CompositeTask, ...] = (
    CompositeTask(
        task_id="composite_test_01",
        math_question="What is (14 × 5) modulo 6?",
        expected_intermediate=4,
        code_expression_template=(
            '":".join(["go"] * {k})'
        ),
        expected_answer="go:go:go:go",
    ),
    CompositeTask(
        task_id="composite_test_02",
        math_question=(
            "What is the greatest common divisor of 54 and 24?"
        ),
        expected_intermediate=6,
        code_expression_template='"thicket"[:{k}]',
        expected_answer="thicke",
    ),
    CompositeTask(
        task_id="composite_test_03",
        math_question="Solve 3x + 1 = 16 for x.",
        expected_intermediate=5,
        code_expression_template=(
            "[i + 1 for i in range({k})]"
        ),
        expected_answer="[1, 2, 3, 4, 5]",
    ),
    CompositeTask(
        task_id="composite_test_04",
        math_question=(
            "What is the floor of the square root of 35?"
        ),
        expected_intermediate=5,
        code_expression_template=(
            "sum(range(1, {k} + 1))"
        ),
        expected_answer="15",
    ),
)


ALL_ATOMIC_DEV_TASKS = (
    MATH_DEV_TASKS + CODE_DEV_TASKS
)

ALL_ATOMIC_TEST_TASKS = (
    MATH_TEST_TASKS + CODE_TEST_TASKS
)

SMOKE_MATH_DEV_TASKS = MATH_DEV_TASKS[:4]
SMOKE_MATH_TEST_TASKS = MATH_TEST_TASKS[:4]
SMOKE_CODE_DEV_TASKS = CODE_DEV_TASKS[:4]
SMOKE_CODE_TEST_TASKS = CODE_TEST_TASKS[:4]


def category_counts(
    tasks: tuple[AtomicTask, ...],
) -> Counter[str]:
    return Counter(task.category for task in tasks)


def validate_split(
    tasks: tuple[AtomicTask, ...],
    *,
    expected_family: str,
    expected_id_prefix: str,
) -> None:
    if not tasks:
        raise ValueError("task split cannot be empty")

    for task in tasks:
        if task.family != expected_family:
            raise ValueError(
                f"{task.task_id} has family "
                f"{task.family!r}, expected "
                f"{expected_family!r}"
            )

        if not task.task_id.startswith(expected_id_prefix):
            raise ValueError(
                f"{task.task_id} must start with "
                f"{expected_id_prefix!r}"
            )


def validate_task_collections() -> None:
    validate_split(
        MATH_DEV_TASKS,
        expected_family="math",
        expected_id_prefix="math_dev_",
    )
    validate_split(
        MATH_TEST_TASKS,
        expected_family="math",
        expected_id_prefix="math_test_",
    )
    validate_split(
        CODE_DEV_TASKS,
        expected_family="code",
        expected_id_prefix="code_dev_",
    )
    validate_split(
        CODE_TEST_TASKS,
        expected_family="code",
        expected_id_prefix="code_test_",
    )

    all_atomic_tasks = (
        ALL_ATOMIC_DEV_TASKS
        + ALL_ATOMIC_TEST_TASKS
    )

    task_ids = [
        task.task_id
        for task in all_atomic_tasks
    ]

    if len(task_ids) != len(set(task_ids)):
        raise ValueError(
            "atomic task IDs must be unique"
        )

    composite_ids = [
        task.task_id
        for task in COMPOSITE_TEST_TASKS
    ]

    if len(composite_ids) != len(set(composite_ids)):
        raise ValueError(
            "composite task IDs must be unique"
        )

    if (
        category_counts(MATH_DEV_TASKS)
        != category_counts(MATH_TEST_TASKS)
    ):
        raise ValueError(
            "math dev/test categories are not balanced"
        )

    if (
        category_counts(CODE_DEV_TASKS)
        != category_counts(CODE_TEST_TASKS)
    ):
        raise ValueError(
            "code dev/test categories are not balanced"
        )


validate_task_collections()


def main() -> None:
    print(f"Task suite: {TASK_SUITE_VERSION}")
    print(
        f"Math development tasks: "
        f"{len(MATH_DEV_TASKS)}"
    )
    print(
        f"Math test tasks: "
        f"{len(MATH_TEST_TASKS)}"
    )
    print(
        f"Code development tasks: "
        f"{len(CODE_DEV_TASKS)}"
    )
    print(
        f"Code test tasks: "
        f"{len(CODE_TEST_TASKS)}"
    )
    print(
        f"Composite test tasks: "
        f"{len(COMPOSITE_TEST_TASKS)}"
    )

    print("\nMath categories:")
    for category in sorted(
        category_counts(MATH_DEV_TASKS)
    ):
        print(f"  {category}")

    print("\nCode categories:")
    for category in sorted(
        category_counts(CODE_DEV_TASKS)
    ):
        print(f"  {category}")

    print("\nExample math prompt:")
    print(MATH_DEV_TASKS[0].prompt)

    print("\nExample code prompt:")
    print(CODE_DEV_TASKS[0].prompt)


if __name__ == "__main__":
    main()