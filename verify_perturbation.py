from __future__ import annotations

import mlx.core as mx

from model import BaseModel
from perturb import LocalPerturbation, PerturbationConfig
from tasks import CODE_DEV_TASKS


SYSTEM_PROMPT = (
    "Solve the task carefully. End with only the final value "
    "inside <answer> and </answer>."
)


def _scalar(value: mx.array) -> float:
    mx.eval(value)
    return float(value.item())


def main() -> None:
    """Prove that a perturbation changes the model numerically."""

    print("Loading the base model")
    model = BaseModel()
    task = CODE_DEV_TASKS[1]

    config = PerturbationConfig(
        rank=4,
        scale=0.01,
        seed=0,
    )
    perturbation = LocalPerturbation(
        model,
        config,
    )

    print(
        "Resolved layers:",
        perturbation.layer_indices,
    )
    print(
        "Target matrices:",
        len(perturbation.target_labels),
    )

    base_logits = model.next_token_logits(
        task.prompt,
        system_prompt=SYSTEM_PROMPT,
    )
    base_text = model.generate(
        task.prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=96,
    )

    with perturbation.active() as active:
        print(
            "Installed wrappers:",
            len(active.wrappers),
        )

        perturbed_logits = model.next_token_logits(
            task.prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        perturbed_text = model.generate(
            task.prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=96,
        )

    restored_logits = model.next_token_logits(
        task.prompt,
        system_prompt=SYSTEM_PROMPT,
    )

    difference = perturbed_logits - base_logits
    difference_rms = mx.sqrt(
        mx.mean(difference * difference)
    )
    base_rms = mx.sqrt(
        mx.mean(base_logits * base_logits)
    )
    relative_rms = difference_rms / mx.maximum(
        base_rms,
        1e-12,
    )
    max_absolute_change = mx.max(
        mx.abs(difference)
    )

    base_log_probs = (
        base_logits - mx.logsumexp(base_logits)
    )
    perturbed_log_probs = (
        perturbed_logits
        - mx.logsumexp(perturbed_logits)
    )
    base_probabilities = mx.exp(base_log_probs)
    kl_divergence = mx.sum(
        base_probabilities
        * (base_log_probs - perturbed_log_probs)
    )

    base_argmax = int(mx.argmax(base_logits).item())
    perturbed_argmax = int(
        mx.argmax(perturbed_logits).item()
    )
    restoration_error = mx.max(
        mx.abs(restored_logits - base_logits)
    )

    print("\nNumerical effect")
    print(
        "  logit relative RMS change: "
        f"{_scalar(relative_rms):.6e}"
    )
    print(
        "  maximum absolute logit change: "
        f"{_scalar(max_absolute_change):.6e}"
    )
    print(
        "  next-token KL divergence: "
        f"{_scalar(kl_divergence):.6e}"
    )
    print(
        "  first-token argmax changed: "
        f"{base_argmax != perturbed_argmax}"
    )
    print(
        "  restoration max error: "
        f"{_scalar(restoration_error):.6e}"
    )

    print("\nBehavioral effect")
    print(f"  base:      {base_text!r}")
    print(f"  perturbed: {perturbed_text!r}")
    print(
        "  generated text changed: "
        f"{base_text != perturbed_text}"
    )

    if _scalar(max_absolute_change) == 0.0:
        raise RuntimeError(
            "the perturbation had zero numerical effect"
        )

    if _scalar(restoration_error) != 0.0:
        raise RuntimeError(
            "removing the perturbation did not restore "
            "the exact base logits"
        )

    print(
        "\nPASS: the perturbation changes logits and "
        "restores the base model exactly."
    )


if __name__ == "__main__":
    main()