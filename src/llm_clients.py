"""Unified clients for querying free-tier LLM providers."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

import requests

LOGGER = logging.getLogger(__name__)
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 60
DEFAULT_MAX_TOKENS = 150


class LLMClientError(RuntimeError):
    """Raised when a provider call cannot be completed successfully."""


@dataclass
class BaseClient:
    """Base provider client with shared retry and delay behavior."""

    config: dict

    def _sleep_between_requests(self) -> None:
        """Sleep using configured inter-request delay."""
        delay = float(self.config.get("request_delay", 0.5) or 0.5)
        if delay > 0:
            time.sleep(delay)

    def _run_with_retries(self, func: Callable[[], str]) -> str:
        """Execute an API call with rate-limit aware retries."""
        for attempt in range(1, MAX_RETRIES + 1):
            self._sleep_between_requests()
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit_error(exc) and attempt < MAX_RETRIES:
                    LOGGER.warning(
                        "Rate limit encountered on attempt %s/%s; retrying in %ss.",
                        attempt,
                        MAX_RETRIES,
                        RETRY_SLEEP_SECONDS,
                    )
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue
                raise
        raise LLMClientError("Retry loop exhausted unexpectedly.")

    def generate(self, prompt: str, model_id: str) -> str:
        """Generate a text completion for the supplied prompt."""
        raise NotImplementedError


@dataclass
class GroqClient(BaseClient):
    """Groq chat-completions wrapper."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call the Groq chat completion API."""
        from groq import Groq

        api_key = self.config.get("groq_api_key", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("groq_api_key is missing in config.yaml")

        client = Groq(api_key=api_key)

        def _call() -> str:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()

        return self._run_with_retries(_call)


@dataclass
class GeminiClient(BaseClient):
    """Google Gemini wrapper using google-genai SDK."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call Google AI Studio Gemini inference using google-genai SDK."""
        from google import genai
        from google.genai import types

        api_key = self.config.get("gemini_api_key", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("gemini_api_key is missing in config.yaml")

        client = genai.Client(api_key=api_key)

        def _call() -> str:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=DEFAULT_MAX_TOKENS,
                    temperature=0,
                ),
            )
            return str(response.text or "").strip()

        return self._run_with_retries(_call)


@dataclass
class GitHubModelsClient(BaseClient):
    """GitHub Models wrapper via the OpenAI SDK."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call the GitHub Models Azure-hosted chat API."""
        from openai import OpenAI

        api_key = self.config.get("github_token", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("github_token is missing in config.yaml")

        client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=api_key)

        def _call() -> str:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()

        return self._run_with_retries(_call)


@dataclass
class OpenRouterClient(BaseClient):
    """OpenRouter wrapper using direct HTTP requests."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call OpenRouter free-tier chat completions."""
        api_key = self.config.get("openrouter_api_key", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("openrouter_api_key is missing in config.yaml")
        if not str(model_id).endswith(":free"):
            raise ValueError(f"OpenRouter model must be a free-tier model: {model_id}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/simhadripraveena2-bit/LinguisticRedline",
            "X-Title": "LinguisticRedline Research",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0,
        }

        def _call() -> str:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            if response.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests", response=response)
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()

        return self._run_with_retries(_call)

@dataclass
class CerebrasClient(BaseClient):
    """Cerebras inference wrapper using OpenAI-compatible SDK."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call Cerebras cloud inference API."""
        from openai import OpenAI

        api_key = self.config.get("cerebras_api_key", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("cerebras_api_key is missing in config.yaml")

        client = OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
        )

        def _call() -> str:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()

        return self._run_with_retries(_call)


@dataclass
class HuggingFaceClient(BaseClient):
    """Hugging Face Inference API wrapper."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call Hugging Face serverless chat completion via cerebras provider."""
        from huggingface_hub import InferenceClient

        api_key = self.config.get("huggingface_token", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("huggingface_token is missing in config.yaml")

        client = InferenceClient(
            provider="cerebras",
            api_key=api_key,
        )

        def _call() -> str:
            response = client.chat_completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            return str(response.choices[0].message.content or "").strip()

        return self._run_with_retries(_call)


@dataclass
class SambanovaClient(BaseClient):
    """SambaNova Cloud inference wrapper using OpenAI-compatible SDK."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call SambaNova Cloud inference API."""
        from openai import OpenAI

        api_key = self.config.get("sambanova_api_key", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("sambanova_api_key is missing in config.yaml")

        client = OpenAI(
            base_url="https://api.sambanova.ai/v1",
            api_key=api_key,
        )

        def _call() -> str:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()

        return self._run_with_retries(_call)

@dataclass
class MistralClient(BaseClient):
    """Mistral AI inference wrapper using OpenAI-compatible SDK."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call Mistral AI La Plateforme inference API."""
        from openai import OpenAI

        api_key = self.config.get("mistral_api_key", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("mistral_api_key is missing in config.yaml")

        client = OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=api_key,
        )

        def _call() -> str:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()

        return self._run_with_retries(_call)


@dataclass
class NvidiaClient(BaseClient):
    """NVIDIA NIM inference wrapper using OpenAI-compatible SDK."""

    def generate(self, prompt: str, model_id: str) -> str:
        """Call NVIDIA NIM hosted inference API."""
        from openai import OpenAI

        api_key = self.config.get("nvidia_api_key", "")
        if not api_key or str(api_key).startswith("YOUR_"):
            raise ValueError("nvidia_api_key is missing in config.yaml")

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )

        def _call() -> str:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()

        return self._run_with_retries(_call)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an exception likely represents provider throttling."""
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    response_code = getattr(response, "status_code", None)
    message = str(exc).lower()
    return status_code == 429 or response_code == 429 or "rate limit" in message or "429" in message


def extract_score(text: str) -> float:
    """Extract the first integer score from 1-10 from model output text.

    Handles formats: bare number, X/10, X out of 10, Score: X, Risk: X,
    Rating: X, Level: X, and common markdown/punctuation wrappers.
    """
    cleaned = re.sub(r"[`*_#>]", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Try "X/10" or "X out of 10" patterns first
    pattern_out_of = re.search(r"\b([1-9]|10)\s*(?:/|out of)\s*10\b", cleaned, re.IGNORECASE)
    if pattern_out_of:
        return float(pattern_out_of.group(1))

    # Try "Score: X" or "Risk: X" or "Rating: X" or "Level: X" label patterns
    pattern_label = re.search(
        r"(?:score|risk|rating|level)\s*[:\-]?\s*\b([1-9]|10)\b",
        cleaned,
        re.IGNORECASE,
    )
    if pattern_label:
        return float(pattern_label.group(1))

    # Fallback: first standalone number 1-10
    match = re.search(r"\b([1-9]|10)\b", cleaned)
    if not match:
        return -1.0
    return float(match.group(1))


def get_all_model_configs(config: dict) -> list[dict[str, Any]]:
    """Flatten configured provider model lists into one list."""
    models = config.get("models", {}) or {}
    flattened: list[dict[str, Any]] = []
    for provider, provider_models in models.items():
        for item in provider_models or []:
            normalized = dict(item)
            normalized.setdefault("provider", provider)
            flattened.append(normalized)
    return flattened


def get_model_client(provider: str, config: dict) -> BaseClient:
    """Instantiate a provider-specific client implementation."""
    registry: dict[str, type[BaseClient]] = {
        "groq": GroqClient,
        "cerebras": CerebrasClient,
        "sambanova": SambanovaClient,
        "mistral": MistralClient,
    }
    if provider not in registry:
        raise ValueError(f"Unsupported provider: {provider}")
    return registry[provider](config=config)


def query_model(prompt: str, model_config: dict, config: dict) -> dict:
    """Query a configured model and return a standardized response payload."""
    model_id = str(model_config.get("id", ""))
    display_name = str(model_config.get("display_name", model_id))
    provider = str(model_config.get("provider", ""))

    try:
        client = get_model_client(provider, config)
        raw_response = client.generate(prompt=prompt, model_id=model_id)
        score = extract_score(raw_response)
        return {
            "model_id": model_id,
            "display_name": display_name,
            "provider": provider,
            "score": score,
            "raw_response": raw_response,
            "success": score != -1.0,
            "error": None if score != -1.0 else "Could not extract a 1-10 score from the response.",
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Model query failed for %s (%s)", display_name, provider)
        return {
            "model_id": model_id,
            "display_name": display_name,
            "provider": provider,
            "score": -1.0,
            "raw_response": "",
            "success": False,
            "error": str(exc),
        }