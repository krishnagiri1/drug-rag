"""Thin, swappable LLM clients plus an on-disk response cache.

Shared with the agent-judge project (same clients, caching, and retry logic).

Providers
---------
openai     Any OpenAI-compatible endpoint (OpenAI, Groq, Together, OpenRouter, vLLM...).
           Uses OPENAI_API_KEY and optional --base-url / OPENAI_BASE_URL.
anthropic  Anthropic Messages API. Uses ANTHROPIC_API_KEY.
ollama     Local models via Ollama's native API (lets us raise the context window,
           which Ollama's OpenAI-compatible endpoint cannot). Free, runs on your GPU.
mock       Deterministic heuristic "judge" for tests and dry runs; no network, no cost.

Every call is cached by (provider, model, system, user), so re-running an experiment
or re-using a judgment in another experiment costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


def _with_retries(fn, attempts: int = 5, base_delay: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # rate limits, timeouts, transient 5xx
            last = e
            msg = str(e).lower()
            if any(k in msg for k in ("invalid api key", "authentication", "not found", "does not exist")):
                raise
            time.sleep(base_delay * (2**i))
    raise last  # type: ignore[misc]


class OpenAICompatClient:
    def __init__(self, model: str, base_url: Optional[str] = None, max_tokens: int = 1500):
        from openai import OpenAI

        self.model = model
        self.name = f"openai:{model}"
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)
        self._kwargs = {"temperature": 0, "max_tokens": max_tokens}

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def call():
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, **self._kwargs)
            except Exception as e:
                msg = str(e)
                # Some newer models reject these parameters; adapt once and retry.
                if "max_tokens" in msg and "max_tokens" in self._kwargs:
                    self._kwargs.pop("max_tokens")
                    self._kwargs["max_completion_tokens"] = self.max_tokens
                    r = self.client.chat.completions.create(model=self.model, messages=messages, **self._kwargs)
                elif "temperature" in msg and "temperature" in self._kwargs:
                    self._kwargs.pop("temperature")
                    r = self.client.chat.completions.create(model=self.model, messages=messages, **self._kwargs)
                else:
                    raise
            return r.choices[0].message.content or ""

        return _with_retries(call)


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1500):
        import anthropic

        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self._temperature: Optional[float] = 0.0

    def complete(self, system: str, user: str) -> str:
        def call():
            kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                          messages=[{"role": "user", "content": user}])
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            try:
                r = self.client.messages.create(**kwargs)
            except Exception as e:
                if "temperature" in str(e) and self._temperature is not None:
                    self._temperature = None
                    kwargs.pop("temperature")
                    r = self.client.messages.create(**kwargs)
                else:
                    raise
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

        return _with_retries(call)


class OllamaClient:
    def __init__(self, model: str, host: Optional[str] = None, num_ctx: int = 8192, json_mode: bool = False):
        self.model = model
        self.json_mode = json_mode
        self.name = f"ollama:{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = num_ctx

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            **({"format": "json"} if self.json_mode else {}),
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }

        def call():
            req = urllib.request.Request(
                f"{self.host}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())["message"]["content"]

        return _with_retries(call, attempts=3)


class MockClient:
    """Deterministic stand-in for tests and dry runs; no network, no cost.

    It recognizes the project's prompt types by their markers and returns well-formed
    replies: extractive answers citing passage [1], template questions, and judge verdicts
    computed from word overlap.
    """

    def __init__(self, model: str = "extractive"):
        self.model = model
        self.name = f"mock:{model}"

    @staticmethod
    def _words(t: str) -> set:
        return set(re.findall(r"[a-z]{4,}", t.lower()))

    def complete(self, system: str, user: str) -> str:
        if "WRITE_QUESTION" in user:
            passage = user.split("PASSAGE:", 1)[-1]
            title = passage.strip().splitlines()[0].split("|")[0].strip()
            body = " ".join(passage.strip().splitlines()[1:])
            first = re.split(r"(?<=[.;])\s", body.strip())[0][:300]
            return json.dumps({"question": f"According to its label, what does the label say about {title}?",
                               "answer": first})
        if "JUDGE_FAITHFULNESS" in user:
            ctx = user.split("PASSAGES:", 1)[-1].split("ANSWER:", 1)[0]
            ans = user.split("ANSWER:", 1)[-1]
            sents = [x for x in re.split(r"(?<=[.!?])\s+", ans.strip()) if len(x) > 10]
            cw = self._words(ctx)
            claims = [{"claim": x[:200], "supported": len(self._words(x) - cw) <= max(1, len(self._words(x)) // 3)}
                      for x in sents] or [{"claim": ans[:200], "supported": False}]
            return json.dumps({"claims": claims, "refusal": "could not find" in ans.lower()})
        if "JUDGE_CORRECTNESS" in user:
            ref = user.split("REFERENCE:", 1)[-1].split("ANSWER:", 1)[0]
            ans = user.split("ANSWER:", 1)[-1]
            r, a = self._words(ref), self._words(ans)
            overlap = len(r & a) / max(len(r), 1)
            verdict = "correct" if overlap > 0.5 else "partial" if overlap > 0.2 else "incorrect"
            return json.dumps({"verdict": verdict, "reason": f"word overlap {overlap:.2f}"})
        if "EXCERPTS:" in user:  # answer generation: extract the first sentence of passage 1
            p1 = user.split("EXCERPTS:", 1)[1].split("[1]", 1)[-1].split("\n[2]", 1)[0].split("QUESTION:", 1)[0]
            body = " ".join(p1.strip().splitlines()[1:]) or p1
            first = re.split(r"(?<=[.;])\s", body.strip())[0][:400]
            return f"{first} [1]"
        return "I could not find this in the provided drug labels."


class CachedClient:
    """Wraps any client with a thread-safe SQLite cache keyed on the full request."""

    def __init__(self, inner: LLMClient, path: str | Path = ".cache/llm_cache.sqlite"):
        self.inner = inner
        self.name = inner.name
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\x00{system}\x00{user}".encode()).hexdigest()

    def complete(self, system: str, user: str) -> str:
        k = self._key(system, user)
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            return row[0]
        v = self.inner.complete(system, user)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, v))
            self._db.commit()
        return v

    def forget(self, system: str, user: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM cache WHERE k=?", (self._key(system, user),))
            self._db.commit()


def make_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    cache: bool = True,
    cache_path: str | Path = ".cache/llm_cache.sqlite",
    **kwargs,
) -> LLMClient:
    provider = provider.lower()
    if provider == "openai":
        inner: LLMClient = OpenAICompatClient(model, base_url=base_url)
    elif provider == "anthropic":
        inner = AnthropicClient(model)
    elif provider == "ollama":
        inner = OllamaClient(model, host=base_url, **{k: v for k, v in kwargs.items() if k in ("num_ctx", "json_mode")})
    elif provider == "mock":
        inner = MockClient(model)
    else:
        raise ValueError(f"Unknown provider {provider!r}; use openai, anthropic, ollama or mock.")
    return CachedClient(inner, cache_path) if cache else inner
