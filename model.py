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
        "mlx-community/Llama-3.2-3B-Instruct-4bit"
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

