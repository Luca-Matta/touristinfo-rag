"""Chat models behind one protocol: OpenAI and Anthropic.

Both clients support plain text and JSON-schema-constrained output (used by the query
planner and the LLM judge), report token usage, and turn SDK exceptions into
:class:`~rag.errors.ProviderError`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

from ..config import Settings
from ..errors import ConfigError, ProviderError, provider_errors

if TYPE_CHECKING:
    from anthropic.types import MessageParam, OutputConfigParam
    from openai.types.chat import ChatCompletionMessageParam

log = logging.getLogger(__name__)


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.model} returned invalid JSON: {self.text[:200]}") from exc


class ChatModel(Protocol):
    name: str

    def complete(
        self, system: str, messages: list[Message], *, schema: dict[str, Any] | None = None, max_tokens: int = 2000
    ) -> Completion: ...


class OpenAIChat:
    def __init__(self, model: str, timeout_s: float) -> None:
        from openai import OpenAI

        self.client = OpenAI(timeout=timeout_s, max_retries=2)
        self.name = model

    def complete(
        self, system: str, messages: list[Message], *, schema: dict[str, Any] | None = None, max_tokens: int = 2000
    ) -> Completion:
        kwargs: dict[str, Any] = {}
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            }
        with provider_errors(f"OpenAI ({self.name})"):
            resp = self.client.chat.completions.create(
                model=self.name,
                messages=cast("list[ChatCompletionMessageParam]", [{"role": "system", "content": system}, *messages]),
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            model=self.name,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class AnthropicChat:
    def __init__(self, model: str, timeout_s: float) -> None:
        import anthropic

        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=2)
        self.name = model

    def complete(
        self, system: str, messages: list[Message], *, schema: dict[str, Any] | None = None, max_tokens: int = 2000
    ) -> Completion:
        # Short, grounded answers need little reasoning: low effort keeps latency and cost down.
        output_config: dict[str, Any] = {"effort": "low"}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        with provider_errors(f"Anthropic ({self.name})"):
            resp = self.client.messages.create(
                model=self.name,
                max_tokens=max_tokens,
                system=system,
                messages=cast("list[MessageParam]", messages),
                output_config=cast("OutputConfigParam", output_config),
            )
        if resp.stop_reason == "refusal":
            raise ProviderError(f"{self.name} declined to answer this request")
        return Completion(
            text="".join(b.text for b in resp.content if b.type == "text"),
            model=self.name,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class LoggedChat:
    """Decorator that logs latency and token usage of every call."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.name = inner.name

    def complete(
        self, system: str, messages: list[Message], *, schema: dict[str, Any] | None = None, max_tokens: int = 2000
    ) -> Completion:
        start = time.perf_counter()
        result = self.inner.complete(system, messages, schema=schema, max_tokens=max_tokens)
        log.info(
            "%s: %.1fs, %d in / %d out tokens%s",
            self.name,
            time.perf_counter() - start,
            result.input_tokens,
            result.output_tokens,
            " (json)" if schema else "",
        )
        return result


def make_chat_model(settings: Settings, *, planner: bool = False) -> ChatModel:
    match settings.llm_provider:
        case "openai":
            model = (planner and settings.planner_model) or settings.openai_chat_model
            return LoggedChat(OpenAIChat(model, settings.request_timeout_s))
        case "anthropic":
            model = (planner and settings.planner_model) or settings.anthropic_model
            return LoggedChat(AnthropicChat(model, settings.request_timeout_s))
    raise ConfigError("No LLM configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env (or use --retrieve-only)")
