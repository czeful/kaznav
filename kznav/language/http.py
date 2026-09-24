"""Minimal OpenAI-compatible chat-completions client with explicit timeouts and retries.

Built on httpx rather than the ``openai`` SDK so that (1) the request body is
exactly what we log, (2) the raw response JSON is kept verbatim, and (3) the
SDK's own silent retries cannot distort latency or retry counts.

Retried: timeouts, connection errors, HTTP 429 and 5xx. Not retried: other
4xx (a malformed request will not heal by repetition).
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import httpx

from .base import Attempt, CallRecord, TransportError, utc_now


class ChatClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def chat(self, body: Dict[str, Any], rec: CallRecord) -> Dict[str, Any]:
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                rec.http_retries += 1
                time.sleep(self.backoff_s * 2 ** (attempt - 2))
            started = utc_now()
            t0 = time.perf_counter()
            try:
                resp = self._client.post("chat/completions", json=body)
            except httpx.TimeoutException as e:
                last = f"timeout: {e}"
                rec.attempts.append(Attempt("http", started, time.perf_counter() - t0, False, error=last))
                continue
            except httpx.TransportError as e:
                last = f"transport: {e}"
                rec.attempts.append(Attempt("http", started, time.perf_counter() - t0, False, error=last))
                continue
            latency = time.perf_counter() - t0
            try:
                data = resp.json()
            except ValueError:
                data = {"_non_json_body": resp.text[:2000]}
            if resp.status_code == 200 and "_non_json_body" not in data:
                rec.attempts.append(Attempt("http", started, latency, True, http_status=200, response=data))
                return data
            last = f"HTTP {resp.status_code}"
            rec.attempts.append(Attempt("http", started, latency, False, error=last,
                                        http_status=resp.status_code, response=data))
            if resp.status_code != 429 and resp.status_code < 500 and resp.status_code != 200:
                raise TransportError(f"non-retryable {last}: {str(data)[:300]}")
        raise TransportError(f"request failed after {self.max_attempts} attempts; last error: {last}")


def first_message_text(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise TransportError(f"response has no choices[0].message.content: {e}") from None


def first_token_top_logprobs(data: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """``{token: logprob}`` for the first generated token, or None if the server sent none."""
    try:
        content = data["choices"][0].get("logprobs") or {}
        items = content.get("content") or []
    except (KeyError, IndexError, AttributeError):
        return None
    if not items:
        return None
    out: Dict[str, float] = {}
    for cand in items[0].get("top_logprobs") or []:
        tok, lp = cand.get("token"), cand.get("logprob")
        if tok is None or lp is None:
            continue
        # Two ids can decode to the same string; keep total probability mass.
        if tok in out:
            out[tok] = math.log(math.exp(out[tok]) + math.exp(lp))
        else:
            out[tok] = float(lp)
    return out or None
