"""Chat Completions only: explicit models, no retries or synthetic fallbacks."""
from __future__ import annotations

import json
import re
from typing import Literal

from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
from pydantic import BaseModel, ValidationError
from .config import Settings, load_settings

Stage = Literal["breakdown", "research", "synthesis"]


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


class LLM:
    def __init__(self, settings: Settings):
        settings.require_llm()
        self.settings = settings
        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key.get_secret_value(),
                             timeout=180, max_retries=0)

    def complete_json(self, stage: Stage, messages: list[dict[str, str]], output_type: type[BaseModel]) -> BaseModel:
        config = getattr(self.settings.models, stage)
        schema = json.dumps(output_type.model_json_schema())
        system_parts = ["Return only one JSON object matching this schema. Do not expose hidden reasoning. " + schema]
        conversation = []
        for message in messages:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                conversation.append(message)
        # Some compatible gateways reject a second system message, even up front.
        prompt = [{"role": "system", "content": "\n\n".join(system_parts)}, *conversation]
        try:
            response = self.client.chat.completions.create(model=config.model, messages=prompt, **config.options)
        except APITimeoutError as exc:
            raise StageError(stage, "Provider timed out after 180 seconds; check endpoint availability/output limits.") from exc
        except APIConnectionError as exc:
            raise StageError(stage, "Cannot connect to configured endpoint; check base URL and network.") from exc
        except APIStatusError as exc:
            raise StageError(stage, f"Provider HTTP {exc.status_code}; check credentials, model access, quota and stage options.") from exc
        except Exception as exc:
            raise StageError(stage, "Provider call failed; check the configured Chat Completions options and model.") from exc
        if not response.choices or response.choices[0].finish_reason == "length":
            raise StageError(stage, "Empty or truncated response; adjust the provider-specific output limit.")
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise StageError(stage, "Provider returned empty assistant content.")
        fenced = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            return output_type.model_validate(json.loads(text))
        except (ValueError, ValidationError) as exc:
            raise StageError(stage, "Malformed JSON or schema mismatch; check model JSON-generation capability/output limits.") from exc


_default_llm: LLM | None = None


def complete_json(stage: Stage, messages: list[dict[str, str]], output_type: type[BaseModel]) -> BaseModel:
    global _default_llm
    if _default_llm is None:
        _default_llm = LLM(load_settings())
    return _default_llm.complete_json(stage, messages, output_type)
