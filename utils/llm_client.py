# -*- coding: utf-8 -*-
"""
LLMClient — OpenAI-compatible Chat Completions client (requests-based)
---------------------------------------------------------------------

Features
- Works with OpenAI, vLLM OpenAI-compatible servers, Azure OpenAI (auth header switch).
- Reads endpoint/model/params from a dict (e.g., config["execute_model"]).
- Supports chat_complete() and chat_complete_n() (single request with n, fallback to loop).
- Exposes make_messages(system_prompt, user_prompt, history) helper.
- Retries with exponential backoff; pluggable headers/proxies/timeout.

Required config shape (example):
{
  "name": "exec-llama-3.1-8b-instruct",        # required: model/deployment name
  "endpoint": "http://127.0.0.1:8000",         # required: base url without trailing /v1
  "api_key": "token-abc123",                   # optional (vLLM可随便填/忽略)
  "auth_type": "bearer",                       # "bearer" (OpenAI/vLLM默认) | "azure"
  "organization": null,                        # optional (OpenAI)
  "headers": {"X-Extra": "1"},                 # optional
  "proxies": {"http": "...", "https": "..."},  # optional
  "verify_ssl": true,                          # optional, default True
  "request_timeout": 120,                      # seconds
  "max_retries": 3,                            # retries on 5xx/timeouts
  "backoff_base": 0.5,                         # seconds
  "params": {                                  # default generation params
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512
  }
}

References:
- vLLM OpenAI-compatible server (/v1/chat/completions) docs. 
- OpenAI Chat Completions API (messages format, Bearer auth). 
- Azure OpenAI uses 'api-key' header (not Bearer).
"""
from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg or {})
        self.name: str = self.cfg.get("name") or self.cfg.get("model") or "unknown-model"
        self.endpoint: str = (self.cfg.get("endpoint") or "http://127.0.0.1:8002").rstrip("/")
        self.params: Dict[str, Any] = dict(self.cfg.get("params") or {})
        self.api_key: Optional[str] = self.cfg.get("api_key")
        self.organization: Optional[str] = self.cfg.get("organization")
        self.auth_type: str = str(self.cfg.get("auth_type") or "bearer").lower().strip()
        self.timeout: float = float(self.cfg.get("request_timeout", 120))
        self.max_retries: int = int(self.cfg.get("max_retries", 3))
        self.backoff_base: float = float(self.cfg.get("backoff_base", 0.5))
        self.verify_ssl: bool = bool(self.cfg.get("verify_ssl", True))
        self.session = requests.Session()
        # optional proxies and headers
        proxies = self.cfg.get("proxies")
        if proxies:
            self.session.proxies.update(proxies)
        self.extra_headers: Dict[str, str] = dict(self.cfg.get("headers") or {})

    # -----------------------
    # Public helpers
    # -----------------------
    def make_messages(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Compose a standard Chat Completions 'messages' list.
        history: list of {"role": "user"/"assistant", "content": "..."}
        """
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": str(system_prompt)})
        if history:
            for m in history:
                r = m.get("role")
                c = m.get("content", "")
                if r in ("system", "user", "assistant") and isinstance(c, str):
                    msgs.append({"role": r, "content": c})
        if user_prompt:
            msgs.append({"role": "user", "content": str(user_prompt)})
        return msgs

    def chat_complete(self, messages: List[Dict[str, str]], **overrides) -> str:
        """
        Return the first choice text of a chat completion.
        """
        payload = self._build_payload(messages, **overrides)
        data = self._post_json("/v1/chat/completions", payload)
        if not data:
            return ""
        try:
            content = data["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else str(content)
        except Exception:
            return ""

    def chat_complete_n(self, messages: List[Dict[str, str]], n: int = 2, **overrides) -> List[str]:
        """
        Request multiple choices with one API call (if supported).
        Fallback: loop n times.
        """
        if n <= 1:
            return [self.chat_complete(messages, **overrides)]

        try:
            payload = self._build_payload(messages, **overrides)
            payload["n"] = int(n)
            data = self._post_json("/v1/chat/completions", payload)
            outs: List[str] = []
            if data and isinstance(data.get("choices"), list):
                for ch in data["choices"]:
                    txt = (ch.get("message", {}) or {}).get("content", "")
                    outs.append(txt if isinstance(txt, str) else str(txt))
            return outs or [self.chat_complete(messages, **overrides) for _ in range(n)]
        except Exception:
            # fallback loop
            return [self.chat_complete(messages, **overrides) for _ in range(n)]

    # -----------------------
    # Internal
    # -----------------------
    def _build_payload(self, messages: List[Dict[str, str]], **overrides) -> Dict[str, Any]:
        payload = {
            "model": self.name,
            "messages": messages,
        }
        # merge defaults (temperature/top_p/max_tokens/stop/seed/response_format etc.)
        payload.update(self.params or {})
        # then apply call-time overrides
        for k, v in (overrides or {}).items():
            if v is not None:
                payload[k] = v
        return payload

    def _auth_headers(self) -> Dict[str, str]:
        hdrs = {
            "Content-Type": "application/json",
        }
        # OpenAI / vLLM: Bearer
        if self.auth_type == "bearer":
            if self.api_key:
                hdrs["Authorization"] = f"Bearer {self.api_key}"
            if self.organization:
                hdrs["OpenAI-Organization"] = self.organization
        # Azure OpenAI: api-key header
        elif self.auth_type == "azure":
            if self.api_key:
                hdrs["api-key"] = self.api_key
        # merge user headers last
        hdrs.update(self.extra_headers or {})
        return hdrs

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url = f"{self.endpoint}{path}"
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    url, data=json.dumps(payload),
                    headers=self._auth_headers(),
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                if resp.status_code // 100 == 2:
                    return resp.json()
                # server errors / rate limits -> retry
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    self._sleep_backoff(attempt)
                    continue
                # other client errors -> no retry
                log.warning("LLMClient (%s) non-retryable status=%s: %s",
                            self.name, resp.status_code, resp.text[:300])
                return None
            except requests.RequestException as e:
                last_err = str(e)
                self._sleep_backoff(attempt)
        if last_err:
            log.error("LLMClient (%s) request failed after retries: %s", self.name, last_err)
        return None

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2 ** (attempt - 1))
        try:
            time.sleep(min(delay, 8.0))
        except Exception:
            pass
