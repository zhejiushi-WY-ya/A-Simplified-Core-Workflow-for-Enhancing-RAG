from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Optional

from openai import OpenAI

from .wtb_integration.tracing import record_materialization_event


def _as_bool(value: str, default: bool = False) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _should_use_wtb_cache() -> bool:
    explicit = os.getenv("RAG_USE_WTB_CACHE", "").strip()
    if explicit:
        return _as_bool(explicit, default=False)
    return bool(os.getenv("WTB_LLM_CACHE_PATH", "").strip())


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    pieces.append(str(text))
        return "\n".join(pieces)
    return str(content)


def _split_messages(messages: Iterable[Any]) -> tuple[Optional[str], str]:
    system_parts = []
    prompt_parts = []

    for message in messages:
        if isinstance(message, tuple) and len(message) == 2:
            role, content = message
        elif isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            role = "user"
            content = message

        text = _message_text(content).strip()
        if not text:
            continue

        if role == "system":
            system_parts.append(text)
        elif role == "user":
            prompt_parts.append(text)
        else:
            prompt_parts.append(f"[{role}] {text}")

    system_prompt = "\n\n".join(system_parts).strip() or None
    prompt = "\n\n".join(prompt_parts).strip()
    return system_prompt, prompt


def _request_signature(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class _EmbeddingItem:
    embedding: list[float]


class _WTBChatCompletionsAdapter:
    def __init__(self, service: Any, default_model: str):
        self._service = service
        self._default_model = default_model

    def create(
        self,
        *,
        model: Optional[str] = None,
        messages: Iterable[Any],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **_kwargs: Any,
    ) -> Any:
        system_prompt, prompt = _split_messages(messages)
        resolved_model = model or self._default_model
        result = self._service.generate_text_result(
            prompt=prompt,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        record_materialization_event(
            event_type="chat_completion",
            model=result.model,
            cache_key=result.cache_key,
            cache_hit=result.cache_hit,
            request_signature=_request_signature(
                {
                    "model": resolved_model,
                    "system_prompt": system_prompt or "",
                    "prompt": prompt,
                    "temperature": float(temperature),
                    "max_tokens": max_tokens,
                }
            ),
            metadata={"duration_ms": result.duration_ms},
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result.text))]
        )


class _WTBEmbeddingsAdapter:
    def __init__(self, service: Any, default_model: str):
        self._service = service
        self._default_model = default_model

    def create(self, *, model: Optional[str] = None, input: Any, **_kwargs: Any) -> Any:
        if isinstance(input, str):
            texts = [input]
        else:
            texts = [str(item) for item in input]
        resolved_model = model or self._default_model
        embeddings = self._service.generate_embeddings(texts, model=resolved_model)
        record_materialization_event(
            event_type="embedding_generation",
            model=resolved_model,
            request_signature=_request_signature(
                {
                    "model": resolved_model,
                    "input": texts,
                }
            ),
            metadata={"input_count": len(texts)},
        )
        return SimpleNamespace(data=[_EmbeddingItem(embedding=list(vector)) for vector in embeddings])


class _WTBOpenAICompatibleClient:
    def __init__(self, service: Any, default_text_model: str, default_embedding_model: str):
        self.chat = SimpleNamespace(
            completions=_WTBChatCompletionsAdapter(
                service=service,
                default_model=default_text_model,
            )
        )
        self.embeddings = _WTBEmbeddingsAdapter(
            service=service,
            default_model=default_embedding_model,
        )


def _build_wtb_service(config, *, base_url: str, api_key: str):
    try:
        from wtb.infrastructure.llm import LangChainOpenAIConfig, get_service
    except ImportError as exc:
        raise RuntimeError(
            "WTB cache mode is enabled but WTB is not installed in this environment."
        ) from exc

    service_config = LangChainOpenAIConfig(
        api_key=api_key,
        base_url=base_url,
        default_text_model=config.llm_model,
        default_embedding_model=config.embedding_model,
        response_cache_path=os.getenv("WTB_LLM_CACHE_PATH") or None,
        response_cache_enabled=_as_bool(
            os.getenv("WTB_LLM_RESPONSE_CACHE_ENABLED", "true"),
            default=True,
        ),
        debug=_as_bool(os.getenv("WTB_LLM_DEBUG", "false"), default=False),
    )
    return get_service(service_config)


def _build_wtb_client(config, *, base_url: str, api_key: str):
    service = _build_wtb_service(config, base_url=base_url, api_key=api_key)
    return _WTBOpenAICompatibleClient(
        service=service,
        default_text_model=config.llm_model,
        default_embedding_model=config.embedding_model,
    )

def get_llm_client(config):
    if _should_use_wtb_cache():
        return _build_wtb_client(
            config,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
    return OpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        max_retries=5,
        timeout=120.0,
    )


def get_embedding_client(config):
    if _should_use_wtb_cache():
        return _build_wtb_client(
            config,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
        )
    return OpenAI(
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        max_retries=5,
        timeout=120.0,
    )
