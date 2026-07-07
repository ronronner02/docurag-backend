import logging
import os
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None


logger = logging.getLogger(__name__)


class LlmNotConfiguredError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    model: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]
    temperature: float
    max_output_tokens: int
    timeout: float


def get_llm_settings() -> LlmSettings:
    return LlmSettings(
        provider=os.getenv("RAG_LLM_PROVIDER", "disabled").strip().lower(),
        model=(os.getenv("RAG_LLM_MODEL") or "").strip() or None,
        api_key=(
            os.getenv("RAG_LLM_API_KEY")
            or os.getenv("RAG_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        or None,
        base_url=(os.getenv("RAG_LLM_BASE_URL") or os.getenv("RAG_OPENAI_BASEURL") or "").strip()
        or None,
        temperature=float(os.getenv("RAG_LLM_TEMPERATURE", "0")),
        max_output_tokens=int(os.getenv("RAG_LLM_MAX_OUTPUT_TOKENS", "512")),
        timeout=float(os.getenv("RAG_LLM_TIMEOUT", "60")),
    )


def is_llm_enabled(settings: LlmSettings | None = None) -> bool:
    settings = settings or get_llm_settings()
    return (
        settings.provider in {"openai", "openai_compatible"}
        and bool(settings.model)
        and bool(settings.api_key)
    )


def generate_grounded_answer(prompt: str, settings: LlmSettings | None = None) -> str:
    settings = settings or get_llm_settings()
    if not is_llm_enabled(settings):
        raise LlmNotConfiguredError(
            "LLM is not configured. Set RAG_LLM_PROVIDER, RAG_LLM_MODEL, and RAG_LLM_API_KEY."
        )
    if OpenAI is None:
        raise LlmNotConfiguredError("openai package is not installed.")

    kwargs = {
        "api_key": settings.api_key,
        "timeout": settings.timeout,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url

    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严格基于检索资料回答的 RAG 助手。"
                    "你必须遵守用户提示中的 source 引用规则。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty answer.")
    return content.strip()
