"""LLM providers: local-first Ollama, with strictly opt-in cloud fallback.

Auto-detection tries the usual places an Ollama server lives relative to an
add-on container - the Ollama add-on, the Docker host, localhost - so the
zero-config path works when the user already runs Ollama.

Recommended models (structured output on CPU): ``qwen3:8b`` for tool-calling and
JSON, ``qwen3:4b`` / ``gemma3:4b`` when only CPU is available.  Reasoning
("think mode") models are avoided: they burn tokens on prose we discard and are
markedly worse at emitting a bare JSON object.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

_LOGGER = logging.getLogger(__name__)

RECOMMENDED_MODELS = ("qwen3:8b", "qwen3:4b", "gemma3:4b", "llama3.1:8b", "mistral:7b")

#: Where an add-on can usually reach Ollama, best first.
OLLAMA_CANDIDATES = (
    "http://a0d7b954-ollama:11434",
    "http://addon_a0d7b954_ollama:11434",
    "http://local-ollama:11434",
    "http://homeassistant.local:11434",
    "http://172.30.32.1:11434",
    "http://host.docker.internal:11434",
    "http://localhost:11434",
)

CLOUD_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "qwen/qwen-2.5-72b-instruct"),
    "anthropic": ("https://api.anthropic.com/v1/messages", "claude-sonnet-5"),
    "google": (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "gemini-2.0-flash",
    ),
}


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a usable response."""


@dataclass
class LLMStatus:
    provider: str
    available: bool
    base_url: str | None = None
    model: str | None = None
    models: list[str] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "base_url": self.base_url,
            "model": self.model,
            "models": self.models or [],
            "error": self.error,
        }


def discover_ollama(timeout: float = 2.0, extra: str | None = None) -> tuple[str | None, list[str]]:
    """Find a reachable Ollama server; returns ``(base_url, model names)``."""
    candidates: list[str] = []
    for value in (extra, os.environ.get("OLLAMA_HOST"), os.environ.get("AMMINER_OLLAMA_URL")):
        if value:
            url = value if value.startswith("http") else f"http://{value}"
            candidates.append(url.rstrip("/"))
    candidates.extend(OLLAMA_CANDIDATES)

    for base_url in dict.fromkeys(candidates):
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=timeout, trust_env=False)
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue
        try:
            payload = response.json()
        except json.JSONDecodeError:
            continue
        models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
        _LOGGER.info("Found Ollama at %s with %d models", base_url, len(models))
        return base_url, models
    return None, []


def pick_model(available: list[str], preferred: str | None = None) -> str | None:
    """Choose the best installed model, avoiding think-mode variants."""
    if preferred:
        return preferred
    usable = [m for m in available if "think" not in m.lower() and "-r1" not in m.lower()]
    for recommended in RECOMMENDED_MODELS:
        for model in usable:
            if model.split(":")[0] == recommended.split(":")[0]:
                return model
    return usable[0] if usable else (available[0] if available else None)


class BaseProvider:
    name = "none"

    #: Whether this provider will actually be asked to generate anything.
    #: Checked by the generator instead of an isinstance test, so a provider
    #: can be swapped or subclassed without changing call sites.
    enabled = True

    def status(self) -> LLMStatus:  # pragma: no cover - trivial
        return LLMStatus(self.name, False, error="no provider configured")

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        raise LLMError("no LLM provider configured")


class NullProvider(BaseProvider):
    """The default: no LLM at all, deterministic rendering only."""

    name = "none"
    enabled = False

    def status(self) -> LLMStatus:
        return LLMStatus(
            "none",
            False,
            error="LLM disabled; suggestions are rendered deterministically from the mined schema",
        )


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 180.0):
        self.timeout = timeout
        discovered_url, models = discover_ollama(extra=base_url)
        self.base_url = base_url or discovered_url
        self.models = models
        self.model = pick_model(models, model)

    def status(self) -> LLMStatus:
        if not self.base_url:
            return LLMStatus(
                self.name,
                False,
                error="No Ollama server found. Install the Ollama add-on or set llm_base_url.",
            )
        if not self.model:
            return LLMStatus(
                self.name,
                False,
                self.base_url,
                error=f"Ollama has no models installed. Try: ollama pull {RECOMMENDED_MODELS[0]}",
            )
        return LLMStatus(self.name, True, self.base_url, self.model, self.models)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.base_url or not self.model:
            raise LLMError(self.status().error or "Ollama unavailable")
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",  # Ollama's structured-output mode
            "options": {"temperature": 0.1, "num_ctx": 8192},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, trust_env=False
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as err:
            raise LLMError(f"Ollama request failed: {err}") from err
        return _parse_json_object(content)


class CloudProvider(BaseProvider):
    """OpenAI-compatible, Anthropic and Google endpoints.  Opt-in only."""

    def __init__(self, provider: str, api_key: str, model: str | None = None,
                 base_url: str | None = None, timeout: float = 120.0):
        self.name = provider
        self.api_key = api_key
        endpoint, default_model = CLOUD_ENDPOINTS.get(provider, (None, None))
        self.base_url = base_url or endpoint
        self.model = model or default_model
        self.timeout = timeout

    def status(self) -> LLMStatus:
        if not self.api_key:
            return LLMStatus(self.name, False, self.base_url, self.model, error="no API key set")
        if not self.base_url:
            return LLMStatus(self.name, False, error=f"unknown provider '{self.name}'")
        return LLMStatus(self.name, True, self.base_url, self.model)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.api_key or not self.base_url:
            raise LLMError(self.status().error or "cloud provider not configured")
        headers = {"Content-Type": "application/json"}
        url = self.base_url

        if self.name == "anthropic":
            headers |= {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
            payload: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 2048,
                "temperature": 0.1,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            extract = lambda data: data["content"][0]["text"]  # noqa: E731
        elif self.name == "google":
            url = str(self.base_url).format(model=self.model) + f"?key={self.api_key}"
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
            }
            extract = lambda data: data["candidates"][0]["content"]["parts"][0]["text"]  # noqa: E731
        else:  # openai / openrouter - OpenAI chat-completions shape
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            extract = lambda data: data["choices"][0]["message"]["content"]  # noqa: E731

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            content = extract(response.json())
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as err:
            raise LLMError(f"{self.name} request failed: {err}") from err
        return _parse_json_object(content)


def _parse_json_object(content: str) -> dict[str, Any]:
    """Parse a model's reply, tolerating fenced code blocks and stray prose."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"model did not return JSON: {content[:200]!r}") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as err:
            raise LLMError(f"model returned invalid JSON: {err}") from err
    if not isinstance(data, dict):
        raise LLMError("model returned JSON that is not an object")
    return data


def build_provider(options) -> BaseProvider:
    """Instantiate the configured provider (``none`` by default)."""
    provider = (options.llm_provider or "none").lower()
    if provider in ("", "none", "off", "disabled"):
        return NullProvider()
    if provider == "ollama":
        return OllamaProvider(
            options.llm_base_url or None,
            options.llm_model or None,
            float(options.llm_timeout_seconds),
        )
    if provider in CLOUD_ENDPOINTS:
        if not options.llm_api_key:
            _LOGGER.warning(
                "llm_provider is '%s' but no llm_api_key is set; falling back to deterministic "
                "rendering",
                provider,
            )
            return NullProvider()
        return CloudProvider(
            provider,
            options.llm_api_key,
            options.llm_model or None,
            options.llm_base_url or None,
            float(options.llm_timeout_seconds),
        )
    _LOGGER.warning("Unknown llm_provider '%s'; using deterministic rendering", provider)
    return NullProvider()
