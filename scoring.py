from __future__ import annotations

import ast
import re


ANSWER_TAG = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)

FINAL_RESULT = re.compile(
    r"(?:final answer|final result)"
    r"\s*(?:is)?\s*(?::|=)?\s*"
    r"([^\r\n]+)",
    re.IGNORECASE,
)

EVALUATES_TO = re.compile(
    r"\bevaluates\s+to\s+([^\r\n]+)",
    re.IGNORECASE,
)

INTEGER = re.compile(r"-?\d+")


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()

    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()

    if len(lines) < 3:
        return cleaned

    if lines[-1].strip() != "```":
        return cleaned

    return "\n".join(lines[1:-1]).strip()


def strip_sentence_period(text: str) -> str:
    cleaned = text.strip()

    if not cleaned.endswith("."):
        return cleaned

    without_period = cleaned[:-1].rstrip()

    try:
        ast.literal_eval(without_period)
        return without_period
    except (ValueError, SyntaxError):
        return cleaned


def extract_explicit_answer(
    prediction: str,
) -> str | None:
    # Use the last tag if the model produced multiple tags.
    tagged_answers = ANSWER_TAG.findall(prediction)

    if tagged_answers:
        return tagged_answers[-1].strip()

    final_results = list(
        FINAL_RESULT.finditer(prediction)
    )

    if final_results:
        answer = final_results[-1].group(1)
        return strip_sentence_period(answer)

    evaluation_results = list(
        EVALUATES_TO.finditer(prediction)
    )

    if evaluation_results:
        answer = evaluation_results[-1].group(1)
        return strip_sentence_period(answer)

    # A one-line response can itself be the answer.
    cleaned = prediction.strip()

    if "\n" not in cleaned:
        return cleaned

    # A long response without a final answer is invalid.
    return None


def extract_math_answer(
    prediction: str,
) -> str | None:
    explicit_answer = extract_explicit_answer(
        prediction
    )

    text_to_search = (
        explicit_answer
        if explicit_answer is not None
        else prediction
    )

    # Allow numbers such as 1,200.
    text_to_search = text_to_search.replace(",", "")

    numbers = INTEGER.findall(text_to_search)

    if not numbers:
        return None

    # The final integer is treated as the answer.
    return numbers[-1]


def normalize_code_value(
    text: str,
) -> tuple[str, object]:
    cleaned = strip_code_fence(text)

    try:
        value = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        # Unquoted string output such as ha-ha.
        return "string", cleaned.strip("'\"")

    if isinstance(value, str):
        # Quoted and unquoted strings become comparable.
        return "string", value

    # Preserve Python type differences.
    return type(value).__name__, value


def code_answer_is_correct(
    expected: str,
    prediction: str,
) -> bool:
    extracted = extract_explicit_answer(
        prediction
    )

    if extracted is None:
        return False

    return (
        normalize_code_value(extracted)
        == normalize_code_value(expected)
    )


def math_answer_is_correct(
    expected: str,
    prediction: str,
) -> bool:
    expected_answer = extract_math_answer(expected)
    predicted_answer = extract_math_answer(prediction)

    if expected_answer is None:
        return False

    if predicted_answer is None:
        return False

    return predicted_answer == expected_answer


def is_correct(
    family: str,
    expected: str,
    prediction: str,
) -> bool:
    if family == "math":
        return math_answer_is_correct(
            expected,
            prediction,
        )

    if family == "code":
        return code_answer_is_correct(
            expected,
            prediction,
        )

    raise ValueError(
        f"unsupported task family: {family!r}"
    )


def main() -> None:
    test_cases = (
        (
            "correct tagged string",
            True,
            is_correct(
                "code",
                "ha-ha-ha",
                "<answer>ha-ha-ha</answer>",
            ),
        ),
        (
            "incorrect extra separators",
            False,
            is_correct(
                "code",
                "ha-ha-ha",
                "<answer>-ha-ha-ha-</answer>",
            ),
        ),
        (
            "correct quoted prose result",
            True,
            is_correct(
                "code",
                "pyt",
                '"python"[:3] evaluates to "pyt".',
            ),
        ),
        (
            "correct tagged list",
            True,
            is_correct(
                "code",
                "[0, 1, 4]",
                "<answer>[0, 1, 4]</answer>",
            ),
        ),
        (
            "correct verbose math",
            True,
            is_correct(
                "math",
                "6",
                "The calculation gives "
                "<answer>6</answer>",
            ),
        ),
        (
            "invalid unfinished code response",
            False,
            is_correct(
                "code",
                "eura",
                "The characters are e, u, r, and a.\n"
                "I will now determine the result.",
            ),
        ),
    )

    for name, expected_result, actual_result in test_cases:
        if actual_result != expected_result:
            raise AssertionError(
                f"{name} failed: expected "
                f"{expected_result}, got {actual_result}"
            )

        print(f"PASS: {name}")

    print("\nAll scoring tests passed.")


if __name__ == "__main__":
    main()