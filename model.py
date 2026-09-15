from __future__ import annotations
from dataclasses import dataclass
from mlx_lm import generate as mlx_generate
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    max_tokens: int = 256


class BaseModel:
    """ deterministic wrapper around one frozen language model"""
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        self.config = config
        self.model, self.tokenizer = load(config.model_id)

        #base checkpoint not changed
        self.model.freeze()
        self.model.eval()

        #argmax decoding
        self.sampler = make_sampler(temp= 0.0)

    def format_prompt(self,user_prompt: str, system_prompt: str | None=None,) -> str:
        """Convert ordinary text into Llama's expected chat format."""

        if not user_prompt.strip():
            return ValueError("user_prompt cannot be empty")

        messages: list[dict[str,str]] = []

        if system_prompt is not None:
            messages.append({"role":"system", "content": system_prompt})

        messages.append({"role":"user", "content":user_prompt})

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt =True)

    def generate(self, user_prompt: str, *, system_prompt: str | None = None, max_tokens: int | None = None ) -> str:
        """Generate one deterministic response"""

        token_limit = (self.config.max_tokens if max_tokens is None else max_tokens)

        if token_limit <= 0:
            raise ValueError("max_tokens must be positive")

        formatted_prompt = self.format_prompt(user_prompt = user_prompt, system_prompt = system_prompt)
        response = mlx_generate(self.model,self.tokenizer,prompt=formatted_prompt,max_tokens=token_limit,sampler=self.sampler,verbose=False)
        return response.strip()


def main() -> None:
    """Small smoke test for base model"""
    model = BaseModel()

    response = model.generate("Return only the integer answer: What is 2 + 3?", max_tokens = 16)

    print(response)

if __name__ == "__main__":
    main()
