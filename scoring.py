from __future__ import annotations

import ast
import re
from dataclasses import dataclass


ANSWER_TAG = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)

CONCLUSION_PATTERNS = (
    re.compile(
        r"(?:the\s+)?(?:final\s+)?"
        r"(?:answer|result)"
        r"(?:\s+of\s+the\s+expression)?"
        r"\s*(?:is|=|:)\s*([^\r\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bevaluates\s+to\s+([^\r\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bresulting\s+in\s+([^\r\n]+)",
        re.IGNORECASE,
    ),
)

INTEGER = re.compile(r"(?<![\w.])-?\d+(?![\w.])")
LIST_LITERAL = re.compile(r"\[[^\[\]\r\n]*\]")
TUPLE_LITERAL = re.compile(r"\([^()\r\n]*\)")
QUOTED_LITERAL = re.compile(
    r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
)


@dataclass(frozen=True)
class ScoreResult:
    """All useful judgments for one model response.

    strict_correct means the value is exactly right and the model
    obeyed the required <answer>...</answer> protocol.

    value_correct ignores the tag protocol but preserves Python
    types. For example, a list is not the same value as a tuple.

    semantic_correct tolerates presentation-only mistakes such as
    writing ``5, 7, 9`` instead of ``[5, 7, 9]``. It does not forgive
    wrong characters, numbers, operators, or sequence elements.
    """

    family: str
    expected_answer: str
    extracted_answer: str | None
    source: str
    format_valid: bool
    value_correct: bool
    semantic_correct: bool

    @property
    def strict_correct(self) -> bool:
        return self.format_valid and self.value_correct

    @property
    def parsed(self) -> bool:
        return self.extracted_answer is not None


def strip_code_fence(text: str) -> str:
    """Remove one complete Markdown code fence."""

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
    """Remove prose punctuation after a valid Python literal."""

    cleaned = text.strip()

    if not cleaned.endswith("."):
        return cleaned

    without_period = cleaned[:-1].rstrip()

    try:
        ast.literal_eval(without_period)
        return without_period
    except (ValueError, SyntaxError):
        return cleaned


def _last_conclusion(prediction: str) -> str | None:
    matches: list[tuple[int, str]] = []

    for pattern in CONCLUSION_PATTERNS:
        for match in pattern.finditer(prediction):
            matches.append(
                (match.start(), match.group(1))
            )

    if not matches:
        return None

    _, answer = max(matches, key=lambda item: item[0])
    return strip_sentence_period(answer)


def extract_explicit_answer(
    prediction: str,
) -> str | None:
    """Extract a tag, a stated conclusion, or a one-line answer."""

    tagged_answers = ANSWER_TAG.findall(prediction)

    if tagged_answers:
        return tagged_answers[-1].strip()

    conclusion = _last_conclusion(prediction)

    if conclusion is not None:
        return conclusion

    cleaned = prediction.strip()

    if cleaned and "\n" not in cleaned:
        return strip_sentence_period(cleaned)

    return None


def _extract_last_code_literal(
    prediction: str,
) -> str | None:
    """Recover the final literal from otherwise verbose prose."""

    candidates: list[tuple[int, str]] = []

    for pattern in (
        LIST_LITERAL,
        TUPLE_LITERAL,
        QUOTED_LITERAL,
    ):
        for match in pattern.finditer(prediction):
            candidate = match.group(0)

            try:
                ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue

            candidates.append(
                (match.start(), candidate)
            )

    if candidates:
        _, answer = max(
            candidates,
            key=lambda item: item[0],
        )
        return answer

    nonempty_lines = [
        line.strip()
        for line in prediction.splitlines()
        if line.strip()
    ]

    if not nonempty_lines:
        return None

    final_line = strip_sentence_period(
        nonempty_lines[-1]
    )

    try:
        ast.literal_eval(final_line)
        return final_line
    except (ValueError, SyntaxError):
        return None


def _extract_with_source(
    family: str,
    prediction: str,
) -> tuple[str | None, str, bool]:
    tagged_answers = ANSWER_TAG.findall(prediction)

    if tagged_answers:
        return (
            tagged_answers[-1].strip(),
            "answer_tag",
            True,
        )

    conclusion = _last_conclusion(prediction)

    if conclusion is not None:
        return conclusion, "conclusion", False

    cleaned = prediction.strip()

    if cleaned and "\n" not in cleaned:
        return (
            strip_sentence_period(cleaned),
            "one_line",
            False,
        )

    if family == "code":
        literal = _extract_last_code_literal(prediction)

        if literal is not None:
            return literal, "final_literal", False

    if family == "math":
        numbers = INTEGER.findall(
            prediction.replace(",", "")
        )

        if numbers:
            return numbers[-1], "final_integer", False

    return None, "missing", False


def extract_math_answer(
    prediction: str,
) -> str | None:
    """Extract the final integer from a math response."""

    extracted, _, _ = _extract_with_source(
        "math",
        prediction,
    )

    if extracted is None:
        return None

    numbers = INTEGER.findall(
        extracted.replace(",", "")
    )

    if not numbers:
        return None

    return numbers[-1]


def normalize_code_value(
    text: str,
) -> tuple[str, object]:
    """Turn a printed Python result into a typed comparison key."""

    cleaned = strip_code_fence(text).strip()

    if (
        len(cleaned) >= 2
        and cleaned.startswith("`")
        and cleaned.endswith("`")
    ):
        cleaned = cleaned[1:-1].strip()

    cleaned = strip_sentence_period(cleaned)

    try:
        value = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        # Unquoted string output such as ha-ha.
        return "str", cleaned.strip("'\"")

    if isinstance(value, str):
        # Quoted and unquoted strings become comparable.
        return "str", value

    return type(value).__name__, value


def _code_values_semantically_equal(
    expected: str,
    predicted: str,
) -> bool:
    expected_type, expected_value = normalize_code_value(
        expected
    )
    predicted_type, predicted_value = normalize_code_value(
        predicted
    )

    if (
        expected_type,
        expected_value,
    ) == (
        predicted_type,
        predicted_value,
    ):
        return True

    # Missing brackets are a presentation error, not a failure to
    # calculate the elements of the sequence.
    if expected_type in {"list", "tuple"}:
        if predicted_type in {"list", "tuple"}:
            return list(predicted_value) == list(
                expected_value
            )

    # A singleton tuple such as (30,) still contains the computed
    # scalar, even though it is not the exact Python return value.
    if expected_type in {"int", "float"}:
        if (
            predicted_type in {"list", "tuple"}
            and len(predicted_value) == 1
        ):
            return predicted_value[0] == expected_value

    return False


def score_prediction(
    family: str,
    expected: str,
    prediction: str,
) -> ScoreResult:
    """Score one prediction without conflating format and skill."""

    if family not in {"math", "code"}:
        raise ValueError(
            f"unsupported task family: {family!r}"
        )

    extracted, source, format_valid = (
        _extract_with_source(
            family,
            prediction,
        )
    )

    if extracted is None:
        return ScoreResult(
            family=family,
            expected_answer=expected,
            extracted_answer=None,
            source=source,
            format_valid=format_valid,
            value_correct=False,
            semantic_correct=False,
        )

    if family == "math":
        expected_value = extract_math_answer(expected)
        predicted_value = extract_math_answer(extracted)
        value_correct = (
            expected_value is not None
            and predicted_value == expected_value
        )
        semantic_correct = value_correct
        extracted = predicted_value
    else:
        value_correct = (
            normalize_code_value(extracted)
            == normalize_code_value(expected)
        )
        semantic_correct = (
            _code_values_semantically_equal(
                expected,
                extracted,
            )
        )

    return ScoreResult(
        family=family,
        expected_answer=expected,
        extracted_answer=extracted,
        source=source,
        format_valid=format_valid,
        value_correct=value_correct,
        semantic_correct=semantic_correct,
    )


def is_correct(
    family: str,
    expected: str,
    prediction: str,
) -> bool:
    """Backward-compatible strict score used by older scripts."""

    return score_prediction(
        family,
        expected,
        prediction,
    ).strict_correct


def main() -> None:
    """Run small scorer tests without loading the language model."""

    cases = (
        (
            "tagged exact string",
            "code",
            "ha-ha-ha",
            "<answer>ha-ha-ha</answer>",
            True,
            True,
        ),
        (
            "wrong extra separators",
            "code",
            "ha-ha-ha",
            "<answer>-ha-ha-ha-</answer>",
            False,
            False,
        ),
        (
            "correct prose result",
            "code",
            "pyt",
            'The expression evaluates to "pyt".',
            False,
            True,
        ),
        (
            "verbose untagged list",
            "code",
            "[0, 3, 6, 9]",
            "Reasoning happened.\nTherefore, the result of "
            "the expression is [0, 3, 6, 9].",
            False,
            True,
        ),
        (
            "missing list brackets",
            "code",
            "[5, 7, 9]",
            "<answer>5, 7, 9</answer>",
            False,
            True,
        ),
        (
            "singleton tuple",
            "code",
            "30",
            "<answer>(30,)</answer>",
            False,
            True,
        ),
        (
            "tagged exact math",
            "math",
            "6",
            "The calculation gives <answer>6</answer>",
            True,
            True,
        ),
        (
            "wrong computation",
            "code",
            "mixxissippi",
            'The final result is "mixixippi".',
            False,
            False,
        ),
    )

    for (
        name,
        family,
        expected,
        prediction,
        expected_strict,
        expected_semantic,
    ) in cases:
        result = score_prediction(
            family,
            expected,
            prediction,
        )

        if result.strict_correct != expected_strict:
            raise AssertionError(
                f"{name}: strict expected "
                f"{expected_strict}, got "
                f"{result.strict_correct}"
            )

        if (
            result.semantic_correct
            != expected_semantic
        ):
            raise AssertionError(
                f"{name}: semantic expected "
                f"{expected_semantic}, got "
                f"{result.semantic_correct}"
            )

        print(f"PASS: {name}")

    print("\nAll scoring tests passed.")


if __name__ == "__main__":
    main()