from __future__ import annotations

import json
from typing import Any
from urllib import request, error

import config


class LLMClient:
    def __init__(self) -> None:
        self.provider = config.LLM_PROVIDER
        self.api_key = config.LLM_API_KEY
        self.base_url = config.LLM_BASE_URL.rstrip("/")
        self.model_name = config.LLM_MODEL_NAME

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_configured():
            raise RuntimeError("LLM API key is not configured.")

        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }).encode("utf-8"),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc
        return payload["choices"][0]["message"]["content"].strip()
