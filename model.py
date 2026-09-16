from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import generate as mlx_generate
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to load and run the frozen base model."""

    model_id: str = (
        "mlx-community/Qwen2.5-7B-Instruct-4bit"
    )
    max_tokens: int = 256

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


class BaseModel:
    """A deterministic wrapper around one frozen language model."""

    def __init__(
        self,
        config: ModelConfig | None = None,
    ) -> None:
        self.config = (
            ModelConfig()
            if config is None
            else config
        )

        self.model, self.tokenizer = load(
            self.config.model_id
        )

        # The base checkpoint is never trained by this project.
        self.model.freeze()
        self.model.eval()

        # Temperature zero gives deterministic greedy decoding.
        self.sampler = make_sampler(temp=0.0)

    def format_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Convert ordinary text into the model's chat format."""

        if not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        messages: list[dict[str, str]] = []

        if system_prompt is not None:
            if not system_prompt.strip():
                raise ValueError(
                    "system_prompt cannot be empty"
                )

            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode_formatted_prompt(
        self,
        formatted_prompt: str,
    ) -> list[int]:
        """Tokenize a chat-formatted prompt once."""

        bos_token = getattr(
            self.tokenizer,
            "bos_token",
            None,
        )

        add_special_tokens = (
            bos_token is None
            or not formatted_prompt.startswith(bos_token)
        )

        try:
            token_ids = self.tokenizer.encode(
                formatted_prompt,
                add_special_tokens=add_special_tokens,
            )
        except TypeError:
            # Compatibility with older tokenizer wrappers.
            token_ids = self.tokenizer.encode(
                formatted_prompt
            )

        if not token_ids:
            raise RuntimeError(
                "the formatted prompt produced no tokens"
            )

        return list(token_ids)

    def next_token_logits(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> mx.array:
        """Return the logits for the first generated token.

        This is useful for checking whether a perturbation has a
        numerical effect even when greedy text stays unchanged.
        """

        formatted_prompt = self.format_prompt(
            user_prompt,
            system_prompt=system_prompt,
        )
        token_ids = self._encode_formatted_prompt(
            formatted_prompt
        )
        tokens = mx.array([token_ids])

        logits = self.model(tokens)[0, -1]
        logits = logits.astype(mx.float32)
        mx.eval(logits)

        return logits

    def answer_logprob(
        self,
        user_prompt: str,
        answer: str,
        *,
        system_prompt: str | None = None,
        scaffold: str = "<answer>",
    ) -> tuple[float, int]:
        """Score the answer's tokens in a single forward pass.

        The scaffold is teacher-forced rather than generated, so this
        measures whether the model puts probability on the correct
        value, not whether it remembered to emit the tag. That keeps
        format compliance out of the objective, and it costs one
        forward pass instead of a full decode.

        Returns the mean log probability per answer token and the
        number of tokens scored. The mean is comparable across tasks
        whose answers differ in length.
        """

        if not answer.strip():
            raise ValueError("answer cannot be empty")

        formatted_prompt = self.format_prompt(
            user_prompt,
            system_prompt=system_prompt,
        )
        prefix_ids = self._encode_formatted_prompt(
            formatted_prompt + scaffold
        )

        # The answer is tokenized on its own rather than as part of
        # the joined string. Both Llama and Qwen BPE merge ">" with
        # whatever follows it ("<answer>eura" tokenizes as
        # [..., ">e", "ura"]), so joint tokenization would fold the
        # closing bracket
        # into the first scored token and quietly put the model's
        # tag-emitting habit back into the objective. Encoding
        # separately keeps the scored region exactly equal to the
        # answer's characters for every task. The cost is a slightly
        # off-distribution split, which is a constant per task and
        # cancels when a perturbation is compared against the base.
        answer_ids = list(
            self.tokenizer.encode(
                answer,
                add_special_tokens=False,
            )
        )

        if not answer_ids:
            raise RuntimeError(
                "the answer produced no tokens"
            )

        full_ids = prefix_ids + answer_ids
        tokens = mx.array([full_ids])
        logits = self.model(tokens)[0]

        # Position i predicts token i + 1, so the first answer token
        # is predicted by the last prefix position.
        start = len(prefix_ids) - 1
        answer_logits = logits[
            start : start + len(answer_ids)
        ].astype(mx.float32)

        log_probs = answer_logits - mx.logsumexp(
            answer_logits,
            axis=-1,
            keepdims=True,
        )
        targets = mx.array(answer_ids)[:, None]
        chosen = mx.take_along_axis(
            log_probs,
            targets,
            axis=-1,
        )[:, 0]

        total = mx.sum(chosen)
        mx.eval(total)

        return (
            float(total.item()) / len(answer_ids),
            len(answer_ids),
        )

    def generate(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate one deterministic response."""

        token_limit = (
            self.config.max_tokens
            if max_tokens is None
            else max_tokens
        )

        if token_limit <= 0:
            raise ValueError(
                "max_tokens must be positive"
            )

        formatted_prompt = self.format_prompt(
            user_prompt,
            system_prompt=system_prompt,
        )

        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=token_limit,
            sampler=self.sampler,
            verbose=False,
        )

        return response.strip()


def main() -> None:
    """Run a small smoke test for the base model."""

    model = BaseModel()
    response = model.generate(
        "Return only the integer answer: What is 2 + 3?",
        max_tokens=16,
    )
    print(response)


if __name__ == "__main__":
    main()

