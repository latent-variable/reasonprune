"""Client for the local oMLX server (synthetic-data paraphrase + judging)."""

from __future__ import annotations

import json
import time

import requests

from .config import WORKER_MODEL, omlx_config


class Worker:
    def __init__(self, model: str = WORKER_MODEL, timeout: float = 300.0):
        cfg = omlx_config()
        self.base_url = cfg["base_url"]
        self.headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.timeout = timeout

    def chat(self, prompt: str, system: str | None = None, max_tokens: int = 512,
             temperature: float = 0.7, retries: int = 3) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions",
                                  headers=self.headers, json=payload,
                                  timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 - retry then surface
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"oMLX worker failed after {retries} tries: {last_err}")

    def alive(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", headers=self.headers,
                             timeout=5)
            return r.ok
        except requests.RequestException:
            return False
