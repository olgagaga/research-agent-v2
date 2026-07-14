"""
LLM client — provider-aware wrapper used by the whole agent.

Supports two providers, auto-detected from environment:

* **openai**     — direct OpenAI API. Uses the SDK's native structured-output
  ``.parse()`` (strict JSON schema handled for you), OpenAI's automatic prompt
  caching, and computes cost from token counts × a pricing table (OpenAI
  responses carry no cost field).
* **openrouter** — OpenRouter. Uses ``response_format=json_schema``, pins
  Anthropic for prompt caching, and reads cost straight from the response /
  generation API.

Selection (first match wins):
  1. ``LLM_PROVIDER`` env = ``openai`` | ``openrouter``
  2. auto: ``OPENAI_API_KEY`` set and no OpenRouter key → openai
  3. auto: OpenRouter key (``OPENROUTER_API_KEY`` or ``API_KEY``) → openrouter

Every call records normalized usage on ``self.last_usage`` (prompt_tokens,
completion_tokens, cached_tokens, cost) for the orchestrator's cost accounting.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_OPENAI_EFFORTS = {"minimal", "low", "medium", "high"}
_OR_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}

# Reasoning-model families that accept `reasoning_effort` / `max_completion_tokens`.
_REASONING_RE = re.compile(r"^(o[1345]\b|o[1345]-|gpt-5)", re.IGNORECASE)

# Approximate OpenAI prices, USD per 1M tokens: (input, cached_input, output).
# Cached input is billed at the discounted middle rate. Prices change — verify
# and edit, or override per-run with LLM_PRICE_IN / LLM_PRICE_CACHED /
# LLM_PRICE_OUT (USD per 1M). Unknown model → cost left at 0.
_OPENAI_PRICING: Dict[str, Tuple[float, float, float]] = {
    "gpt-5":        (1.25, 0.125, 10.00),
    "gpt-5-mini":   (0.25, 0.025, 2.00),
    "gpt-5-nano":   (0.05, 0.005, 0.40),
    "gpt-4o":       (2.50, 1.25, 10.00),
    "gpt-4o-mini":  (0.15, 0.075, 0.60),
    "gpt-4.1":      (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "o3":           (2.00, 0.50, 8.00),
    "o4-mini":      (1.10, 0.275, 4.40),
}


def normalize_reasoning_effort(value: str, default: str = "medium",
                               allowed: Optional[set] = None) -> str:
    v = str(value or "").strip().lower()
    allowed = allowed or _OR_EFFORTS
    return v if v in allowed else default


class LLMClient:
    """Provider-aware LLM wrapper. All LLM calls go through this class."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 provider: str | None = None):
        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        generic_key = os.environ.get("API_KEY")  # provider-agnostic fallback
        model = os.environ.get("MAIN_MODEL", "")

        # Provider selection: explicit env wins; else infer from the model id
        # ("vendor/model" ⇒ openrouter, bare "gpt-4o"/"o3" ⇒ openai); else keys.
        self._provider = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
        if self._provider not in ("openai", "openrouter"):
            if model.startswith("openai/"):
                # "openai/gpt-5-mini" is a direct-OpenAI id written OpenRouter-style.
                self._provider = "openai"
            elif "/" in model:  # e.g. "anthropic/…", "google/…"
                self._provider = "openrouter"
            elif model:  # bare "gpt-5-mini" / "o3"
                self._provider = "openai"
            elif openrouter_key and not openai_key:
                self._provider = "openrouter"
            else:
                self._provider = "openai"

        if self._provider == "openai":
            self._api_key = api_key or openai_key or generic_key or ""
            self._base_url = base_url or os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            )
        else:
            self._api_key = api_key or openrouter_key or generic_key or ""
            self._base_url = base_url or "https://openrouter.ai/api/v1"

        self._client = None
        self.last_usage: Dict[str, Any] = {}
        log.debug("LLMClient provider=%s base_url=%s", self._provider, self._base_url)

    # ------------------------------------------------------------------ client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    @property
    def provider(self) -> str:
        return self._provider

    def _clean_model(self, model: str) -> str:
        """Strip an OpenRouter-style 'openai/' prefix for the direct OpenAI API."""
        if self._provider == "openai" and model.startswith("openai/"):
            return model[len("openai/"):]
        return model

    def _is_reasoning_model(self, model: str) -> bool:
        return bool(_REASONING_RE.match((model or "").replace("openai/", "")))

    # ------------------------------------------------------------------ pricing

    def _openai_cost(self, model: str, prompt_tokens: int,
                     completion_tokens: int, cached_tokens: int = 0) -> float:
        """Precise cost: cached input tokens billed at the discounted rate.

        prompt_tokens is the FULL input count (OpenAI includes cached tokens in
        it), so the uncached portion is prompt_tokens - cached_tokens.
        """
        env_in = os.environ.get("LLM_PRICE_IN")
        env_out = os.environ.get("LLM_PRICE_OUT")
        if env_in is not None and env_out is not None:
            price_in, price_out = float(env_in), float(env_out)
            env_cached = os.environ.get("LLM_PRICE_CACHED")
            price_cached = float(env_cached) if env_cached is not None else price_in
        else:
            key = model.split(":")[0]
            price = _OPENAI_PRICING.get(key)
            if price is None:
                # Longest-prefix match (e.g. "gpt-4o-2024-08-06" → "gpt-4o").
                cands = [k for k in _OPENAI_PRICING if key.startswith(k)]
                price = _OPENAI_PRICING[max(cands, key=len)] if cands else None
            if price is None:
                return 0.0
            price_in, price_cached, price_out = price
        uncached = max(prompt_tokens - cached_tokens, 0)
        return (uncached * price_in
                + cached_tokens * price_cached
                + completion_tokens * price_out) / 1e6

    # ------------------------------------------------------------------ usage

    def _usage_openrouter(self, resp_dict: Dict[str, Any]) -> Dict[str, Any]:
        usage: Dict[str, Any] = dict(resp_dict.get("usage", {}) or {})
        pd = usage.get("prompt_tokens_details") or {}
        if not usage.get("cached_tokens") and isinstance(pd, dict) and pd.get("cached_tokens"):
            usage["cached_tokens"] = int(pd["cached_tokens"])
        if not usage.get("cache_write_tokens") and isinstance(pd, dict):
            cw = (pd.get("cache_write_tokens") or pd.get("cache_creation_tokens")
                  or pd.get("cache_creation_input_tokens"))
            if cw:
                usage["cache_write_tokens"] = int(cw)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost
        usage["prompt_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
        usage["completion_tokens"] = int(usage.get("completion_tokens", 0) or 0)
        usage["cached_tokens"] = int(usage.get("cached_tokens", 0) or 0)
        usage["cost"] = float(usage.get("cost", 0.0) or 0.0)
        return usage

    def _usage_openai(self, resp_dict: Dict[str, Any], model: str) -> Dict[str, Any]:
        u = resp_dict.get("usage", {}) or {}
        pt = int(u.get("prompt_tokens", 0) or 0)
        ct = int(u.get("completion_tokens", 0) or 0)
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        return {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cached_tokens": cached,
            "cost": self._openai_cost(model, pt, ct, cached),
        }

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """OpenRouter-only: fetch cost from the Generation API as a fallback."""
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            for attempt in range(2):
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json().get("data") or {}
                    cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                    if cost is not None:
                        return float(cost)
                time.sleep(0.5)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
        return None

    # ------------------------------------------------------------- structured

    def chat_structured(self,
                        messages: List[Dict[str, Any]],
                        response_format: Any,  # pydantic model class
                        model: str | None = None,
                        reasoning_effort: str = "medium",
                        max_tokens: int = 16384) -> Any:
        """Return a parsed Pydantic instance from a structured-output call."""
        model_name = model or self.default_model()
        if self._provider == "openai":
            return self._parse_openai(messages, response_format, model_name,
                                      reasoning_effort, max_tokens)
        return self._parse_openrouter(messages, response_format, model_name,
                                       reasoning_effort, max_tokens)

    def _parse_openai(self, messages, response_format, model_name,
                      reasoning_effort, max_tokens):
        client = self._get_client()
        model_name = self._clean_model(model_name)
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "response_format": response_format,  # pydantic class; SDK builds strict schema
        }
        if self._is_reasoning_model(model_name):
            kwargs["reasoning_effort"] = normalize_reasoning_effort(
                reasoning_effort, allowed=_OPENAI_EFFORTS
            )
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        # SDK moved parse out of beta in 2.x; support both.
        parse = getattr(getattr(client.chat.completions, "parse", None), "__call__", None)
        completion = (client.chat.completions.parse(**kwargs)
                      if parse else client.beta.chat.completions.parse(**kwargs))

        msg = completion.choices[0].message
        self.last_usage = self._usage_openai(completion.model_dump(), model_name)
        if getattr(msg, "refusal", None):
            raise RuntimeError(f"Model refused: {msg.refusal}")
        if msg.parsed is None:
            raise ValueError("OpenAI returned no parsed object (possibly truncated).")
        return msg.parsed

    def _parse_openrouter(self, messages, response_format, model_name,
                          reasoning_effort, max_tokens):
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)
        schema = response_format.model_json_schema()
        extra_body: Dict[str, Any] = {"reasoning": {"effort": effort, "exclude": True}}
        if model_name.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"], "allow_fallbacks": False,
                "require_parameters": True,
            }
        resp = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=max_tokens,
            extra_body=extra_body,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": response_format.__name__,
                                "schema": schema, "strict": True},
            },
        )
        resp_dict = resp.model_dump()
        self.last_usage = self._usage_openrouter(resp_dict)
        raw = resp_dict.get("choices", [{}])[0].get("message", {}).get("content", "")
        import json as _json
        return response_format.model_validate(_json.loads(raw))

    # ------------------------------------------------------------------- chat

    def chat(self,
             messages: List[Dict[str, Any]],
             model: str,
             tools: List[Dict[str, Any]] | None = None,
             reasoning_effort: str = "medium",
             max_tokens: int = 16384,
             tool_choice: str = "auto") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single free-form / tool-calling LLM call. Returns (message, usage)."""
        client = self._get_client()
        model = self._clean_model(model)
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}

        if self._provider == "openai":
            if self._is_reasoning_model(model):
                kwargs["reasoning_effort"] = normalize_reasoning_effort(
                    reasoning_effort, allowed=_OPENAI_EFFORTS)
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            resp = client.chat.completions.create(**kwargs)
            resp_dict = resp.model_dump()
            self.last_usage = self._usage_openai(resp_dict, model)
        else:
            effort = normalize_reasoning_effort(reasoning_effort)
            extra_body: Dict[str, Any] = {"reasoning": {"effort": effort, "exclude": True}}
            if model.startswith("anthropic/"):
                extra_body["provider"] = {
                    "order": ["Anthropic"], "allow_fallbacks": False,
                    "require_parameters": True,
                }
            kwargs.update(max_tokens=max_tokens, extra_body=extra_body)
            if tools:
                tools_c = list(tools)
                tools_c[-1] = {**tools_c[-1], "cache_control": {"type": "ephemeral", "ttl": "1h"}}
                kwargs["tools"] = tools_c
                kwargs["tool_choice"] = tool_choice
            resp = client.chat.completions.create(**kwargs)
            resp_dict = resp.model_dump()
            self.last_usage = self._usage_openrouter(resp_dict)

        msg = resp_dict.get("choices", [{}])[0].get("message", {})
        return msg, self.last_usage

    # ------------------------------------------------------------------ models

    def default_model(self) -> str:
        model = os.environ.get("MAIN_MODEL", "anthropic/claude-sonnet-4.6")
        if self._provider == "openai" and model.startswith("anthropic/"):
            fallback = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            log.warning("MAIN_MODEL=%s is not an OpenAI model; using %r. "
                        "Set MAIN_MODEL to an OpenAI model id.", model, fallback)
            return fallback
        return model

    def available_models(self) -> List[str]:
        main = self.default_model()
        code = os.environ.get("CODE_MODEL", "")
        light = os.environ.get("LIGHT_MODEL", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light not in (main, code):
            models.append(light)
        return models
