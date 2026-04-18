"""OpenRouter LLM client with retry, exponential backoff, and model fallback."""

import json
import time
import logging
import urllib3
from typing import Optional

import requests

# Suppress InsecureRequestWarning for corporate proxy environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_CHAIN, LLM_TIMEOUT_SECONDS
from guardrails import CircuitBreaker

logger = logging.getLogger("llm_client")


class LLMClient:
    """Calls OpenRouter chat/completions with automatic retry and fallback."""

    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.model_chain = MODEL_CHAIN
        self.circuit_breaker = CircuitBreaker()

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    # ── single request ──

    def _request(self, model: str, messages: list, tools: Optional[list] = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pratiksinha312511/Tabular_RAG_pipeline",
        }

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
        max_retries: int = 2,
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
                    last_err = f"HTTP {exc.response.status_code if exc.response else '?'} from {model}"
                    logger.warning(last_err)
                    if exc.response is not None and exc.response.status_code in (429, 503):
                        time.sleep(2 ** attempt)
                        continue
                    break  # non-retryable → next model

                except Exception as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    logger.warning(f"Error attempt {attempt+1} on {model}: {last_err}")

                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        self.circuit_breaker.record_failure()
        return self._error(f"All models failed. Last error: {last_err}")

    @staticmethod
    def _error(msg: str) -> dict:
        return {"error": True, "message": msg, "content": None, "tool_calls": []}
