"""LLM client with BYOK (Bring Your Own Key) support for OpenRouter and Sarvam AI.

Supports retry, exponential backoff, model fallback, and streaming for both providers.
Provider is selected via LLM_PROVIDER in config (auto-detected from available API keys).
"""

import json
import time
import logging
import urllib3
from typing import Generator, Optional

import requests

# Suppress InsecureRequestWarning for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_CHAIN,
    SARVAM_API_KEY, SARVAM_BASE_URL, SARVAM_MODEL_CHAIN,
    LLM_PROVIDER, LLM_TIMEOUT_SECONDS,
)
from guardrails import CircuitBreaker

logger = logging.getLogger("llm_client")


class LLMClient:
    """Calls chat/completions with automatic retry, fallback, and BYOK provider routing."""

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.circuit_breaker = CircuitBreaker()

        if self.provider == "sarvam":
            self.api_key = SARVAM_API_KEY
            self.base_url = SARVAM_BASE_URL
            self.model_chain = SARVAM_MODEL_CHAIN
        else:
            self.api_key = OPENROUTER_API_KEY
            self.base_url = OPENROUTER_BASE_URL
            self.model_chain = MODEL_CHAIN

        if not self.api_key:
            raise ValueError(
                f"No API key found for provider '{self.provider}'. "
                f"Set {'SARVAM_API_KEY' if self.provider == 'sarvam' else 'OPENROUTER_API_KEY'} in .env"
            )

        logger.info(f"LLM provider: {self.provider} | models: {self.model_chain}")

    # ── single request ──

    def _build_headers(self) -> dict:
        """Build auth headers appropriate for the active provider."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/pratiksinha312511/Tabular_RAG_pipeline"
        return headers

    def _request(self, model: str, messages: list, tools: Optional[list] = None) -> dict:
        headers = self._build_headers()

        payload: dict = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
            verify=False,  # Corporate proxy / self-signed cert workaround
        )
        resp.raise_for_status()
        return resp.json()

    # ── public interface ──

    def chat(
        self,
        messages: list,
        tools: Optional[list] = None,
        max_retries: int = 1,
    ) -> dict:
        """Chat completion with retry + model fallback.  Returns a normalised dict."""
        if not self.circuit_breaker.can_proceed():
            return self._error("Service temporarily unavailable (circuit breaker open).")

        last_err = None

        for model in self.model_chain:
            for attempt in range(max_retries + 1):
                try:
                    raw = self._request(model, messages, tools)
                    self.circuit_breaker.record_success()

                    msg = raw.get("choices", [{}])[0].get("message", {})
                    return {
                        "error": False,
                        "content": msg.get("content", "") or "",
                        "tool_calls": msg.get("tool_calls") or [],
                        "model": model,
                    }

                except requests.exceptions.Timeout:
                    last_err = f"Timeout with {model}"
                    logger.warning(f"Timeout attempt {attempt+1} on {model}")

                except requests.exceptions.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    last_err = f"HTTP {status} from {model}"
                    logger.warning(last_err)
                    if status == 429:
                        break  # rate-limited → skip retries, try next model immediately
                    if status == 503:
                        time.sleep(2 ** attempt)  # exponential backoff
                        continue
                    break  # non-retryable → next model

                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    logger.warning(f"Error attempt {attempt+1} on {model}: {last_err}")

                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s...

        self.circuit_breaker.record_failure()
        return self._error(f"All models failed. Last error: {last_err}")

    @staticmethod
    def _error(msg: str) -> dict:
        return {"error": True, "message": msg, "content": None, "tool_calls": []}

    # ── streaming interface ──

    def _request_stream(self, model: str, messages: list, tools: Optional[list] = None):
        """Open a streaming connection and return the raw response object."""
        headers = self._build_headers()
        payload: dict = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
            verify=False,
            stream=True,
        )
        resp.raise_for_status()
        return resp

    def chat_stream(
        self,
        messages: list,
        tools: Optional[list] = None,
        max_retries: int = 1,
    ) -> Generator[dict, None, None]:
        """Streaming chat — yields dicts:
        {"type": "chunk", "content": "..."}          – text delta
        {"type": "tool_calls", "tool_calls": [...]}  – tool call objects (at end)
        {"type": "done", "model": "..."}             – stream finished
        {"type": "error", "message": "..."}          – all models failed
        """
        if not self.circuit_breaker.can_proceed():
            yield {"type": "error", "message": "Service temporarily unavailable (circuit breaker open)."}
            return

        last_err = None

        for model in self.model_chain:
            for attempt in range(max_retries + 1):
                try:
                    resp = self._request_stream(model, messages, tools)
                    self.circuit_breaker.record_success()

                    accumulated_tool_calls: dict[int, dict] = {}
                    model_used = model

                    for line in resp.iter_lines():
                        if not line:
                            continue
                        decoded = line.decode("utf-8", errors="replace")
                        if not decoded.startswith("data: "):
                            continue
                        payload = decoded[6:].strip()
                        if payload == "[DONE]":
                            break

                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        # Text content
                        content = delta.get("content")
                        if content:
                            yield {"type": "chunk", "content": content}

                        # Tool calls (accumulated across chunks)
                        for tc in (delta.get("tool_calls") or []):
                            idx = tc.get("index", 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc.get("id", f"call_{idx}"),
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            atc = accumulated_tool_calls[idx]
                            if tc.get("id"):
                                atc["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                atc["function"]["name"] += fn["name"]
                            if fn.get("arguments"):
                                atc["function"]["arguments"] += fn["arguments"]

                    resp.close()

                    # Emit tool calls if any
                    if accumulated_tool_calls:
                        yield {
                            "type": "tool_calls",
                            "tool_calls": list(accumulated_tool_calls.values()),
                        }

                    yield {"type": "done", "model": model_used}
                    return  # success

                except requests.exceptions.Timeout:
                    last_err = f"Timeout with {model}"
                    logger.warning(f"Timeout attempt {attempt+1} on {model}")

                except requests.exceptions.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    last_err = f"HTTP {status} from {model}"
                    logger.warning(last_err)
                    if status == 429:
                        break
                    if status == 503:
                        time.sleep(2 ** attempt)  # exponential backoff
                        continue
                    break

                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    logger.warning(f"Stream error attempt {attempt+1} on {model}: {last_err}")

                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s...

        self.circuit_breaker.record_failure()
        yield {"type": "error", "message": f"All models failed. Last error: {last_err}"}
