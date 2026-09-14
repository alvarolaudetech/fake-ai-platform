"""Provider-agnostic LLM invocation layer.

Routes a chat completion request to whichever upstream provider a
`ModelConfig` points at (OpenAI, Anthropic, or an OpenAI-compatible local
server), normalizes the response shape, tracks token usage, and applies
retry/backoff for transient upstream failures.
"""
import time
from dataclasses import dataclass
from typing import AsyncIterator

import anthropic
import httpx
import openai
import tiktoken
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.database import ModelConfig, ModelProvider

settings = get_settings()

RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    httpx.TimeoutException,
)


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    finish_reason: str


class LLMServiceError(Exception):
    """Raised when the upstream provider fails after all retries."""


class LLMService:
    def __init__(self) -> None:
        self._openai_client = openai.OpenAI(
            api_key=settings.openai_api_key, timeout=settings.request_timeout_seconds
        )
        self._anthropic_client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key, timeout=settings.request_timeout_seconds
        )
        self._local_client = openai.OpenAI(
            api_key="not-needed",
            base_url=settings.local_inference_base_url,
            timeout=settings.request_timeout_seconds,
        )
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def estimate_cost_usd(self, model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1000) * float(model.input_cost_per_1k)
        output_cost = (completion_tokens / 1000) * float(model.output_cost_per_1k)
        return round(input_cost + output_cost, 6)

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        reraise=True,
    )
    def complete(
        self,
        model: ModelConfig,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Synchronous, non-streaming chat completion against `model`'s provider."""
        started_at = time.perf_counter()
        max_tokens = max_tokens or model.max_output_tokens

        try:
            if model.provider == ModelProvider.OPENAI:
                result = self._complete_openai(model, messages, temperature, max_tokens)
            elif model.provider == ModelProvider.ANTHROPIC:
                result = self._complete_anthropic(model, messages, temperature, max_tokens)
            elif model.provider == ModelProvider.LOCAL:
                result = self._complete_local(model, messages, temperature, max_tokens)
            else:
                raise LLMServiceError(f"Unsupported provider: {model.provider}")
        except RETRYABLE_EXCEPTIONS:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any SDK-specific error
            raise LLMServiceError(f"Upstream call failed for {model.model_slug}: {exc}") from exc

        result.latency_ms = int((time.perf_counter() - started_at) * 1000)
        return result

    def _complete_openai(self, model, messages, temperature, max_tokens) -> CompletionResult:
        response = self._openai_client.chat.completions.create(
            model=model.upstream_model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return CompletionResult(
            text=choice.message.content or "",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=0,
            finish_reason=choice.finish_reason or "stop",
        )

    def _complete_anthropic(self, model, messages, temperature, max_tokens) -> CompletionResult:
        system_prompt = model.system_prompt_override or next(
            (m.content for m in messages if m.role == "system"), None
        )
        conversation = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

        response = self._anthropic_client.messages.create(
            model=model.upstream_model_name,
            system=system_prompt,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return CompletionResult(
            text=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=0,
            finish_reason=response.stop_reason or "end_turn",
        )

    def _complete_local(self, model, messages, temperature, max_tokens) -> CompletionResult:
        response = self._local_client.chat.completions.create(
            model=model.upstream_model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        return CompletionResult(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else self._estimate_prompt_tokens(messages),
            completion_tokens=usage.completion_tokens if usage else self.count_tokens(choice.message.content or ""),
            latency_ms=0,
            finish_reason=choice.finish_reason or "stop",
        )

    def _estimate_prompt_tokens(self, messages: list[ChatMessage]) -> int:
        return sum(self.count_tokens(m.content) for m in messages)

    async def stream(
        self,
        model: ModelConfig,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield incremental text chunks for providers/models that support streaming."""
        if not model.supports_streaming:
            raise LLMServiceError(f"{model.model_slug} does not support streaming")

        max_tokens = max_tokens or model.max_output_tokens

        if model.provider == ModelProvider.ANTHROPIC:
            system_prompt = model.system_prompt_override or next(
                (m.content for m in messages if m.role == "system"), None
            )
            conversation = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
            with self._anthropic_client.messages.stream(
                model=model.upstream_model_name,
                system=system_prompt,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    yield text
            return

        client = self._openai_client if model.provider == ModelProvider.OPENAI else self._local_client
        completion_stream = client.chat.completions.create(
            model=model.upstream_model_name,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in completion_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


_llm_service_singleton: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service_singleton
    if _llm_service_singleton is None:
        _llm_service_singleton = LLMService()
    return _llm_service_singleton
