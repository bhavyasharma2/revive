"""Client for Nebius Token Factory (OpenAI-compatible) serving NVIDIA Nemotron models.

Model IDs are never hard-coded. The client asks the API which models exist
(``GET /v1/models``) and picks one per *role* by matching on the model name,
so a renamed or re-versioned Nemotron model does not break the app. Any role
can be pinned with an environment variable, e.g. ``REVIVE_MODEL_DEEP``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .textutil import strip_reasoning

TOKEN_FACTORY_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
API_KEY_ENV = "NEBIUS_API_KEY"


class MissingAPIKey(RuntimeError):
    """No Token Factory API key was provided."""


class ModelNotAvailable(RuntimeError):
    """The requested model (or role) is not served by Token Factory."""


class EmptyCompletion(RuntimeError):
    """The model returned no usable text."""


@dataclass(frozen=True)
class RoleRule:
    """How to recognise a role's model: all ``must`` words in the id, ``avoid`` words ranked last."""

    must: tuple[str, ...]
    avoid: tuple[str, ...] = ()


# deep     -> hardest reasoning: diagnosing a failure, writing the patch
# balanced -> mid-size model for most agent steps
# fast     -> cheap and quick: log triage, summaries, retries
ROLE_RULES: dict[str, RoleRule] = {
    "deep": RoleRule(must=("nemotron", "ultra")),
    "balanced": RoleRule(must=("nemotron", "super"), avoid=("vl", "omni")),
    "fast": RoleRule(must=("nemotron", "nano"), avoid=("omni", "vl")),
}


def pick_model(
    available: Iterable[str],
    role: str,
    override: Optional[str] = None,
) -> str:
    """Choose a model id for ``role`` from the ids the API reports as available.

    ``override`` pins an exact id (matched case-insensitively) and must be
    available. Without it, ids containing every ``must`` word are candidates;
    ids containing an ``avoid`` word rank last, then shorter ids first.
    """
    ids = sorted(set(available))
    lowered = {model_id.lower(): model_id for model_id in ids}

    if override:
        found = lowered.get(override.strip().lower())
        if found is None:
            raise ModelNotAvailable(
                f"model {override!r} is not available on Token Factory; "
                f"{len(ids)} models are served (run scripts/smoke_test.py to list them)"
            )
        return found

    rule = ROLE_RULES.get(role)
    if rule is None:
        raise ModelNotAvailable(f"unknown model role {role!r}; expected one of {sorted(ROLE_RULES)}")

    candidates = [
        model_id
        for model_id in ids
        if all(word in model_id.lower() for word in rule.must)
    ]
    if not candidates:
        nvidia = [m for m in ids if "nvidia" in m.lower() or "nemotron" in m.lower()]
        raise ModelNotAvailable(
            f"no available model matches role {role!r} (needs {'+'.join(rule.must)} in its id). "
            f"NVIDIA models served: {nvidia or 'none found'}"
        )

    def rank(model_id: str) -> tuple[int, int, str]:
        penalised = any(word in model_id.lower() for word in rule.avoid)
        return (1 if penalised else 0, len(model_id), model_id)

    return sorted(candidates, key=rank)[0]


class TokenFactoryClient:
    """Thin wrapper over the OpenAI SDK pointed at Token Factory."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = TOKEN_FACTORY_BASE_URL,
        *,
        client: Any = None,
        timeout: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        if client is None:
            key = api_key or os.environ.get(API_KEY_ENV)
            if not key:
                raise MissingAPIKey(
                    f"set the {API_KEY_ENV} environment variable (create a key at https://tokenfactory.nebius.com)"
                )
            from openai import OpenAI  # imported lazily so tests can run without network setup

            client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self._client = client
        self._models: Optional[list[str]] = None
        # model id -> {"calls", "prompt_tokens", "completion_tokens"}
        self.usage: dict[str, dict[str, int]] = {}

    def reset_usage(self) -> None:
        self.usage = {}

    def _record_usage(self, model: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        entry = self.usage.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        entry["calls"] += 1
        if usage is not None:
            entry["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            entry["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)

    def available_models(self, refresh: bool = False) -> list[str]:
        """Ids of every model the account can call (cached after the first call)."""
        if self._models is None or refresh:
            ids: list[str] = []
            for model in self._client.models.list():
                model_id = getattr(model, "id", None)
                if model_id:
                    ids.append(str(model_id))
            self._models = sorted(set(ids))
        return list(self._models)

    def model_for(self, role: str) -> str:
        """Resolve a role ('deep', 'balanced', 'fast') to a concrete model id."""
        override = os.environ.get(f"REVIVE_MODEL_{role.upper()}")
        return pick_model(self.available_models(), role, override)

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        role: str = "deep",
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> str:
        """Run one chat completion and return the answer text with reasoning removed."""
        chosen = model or self.model_for(role)
        request: dict[str, Any] = {
            "model": chosen,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            request["extra_body"] = extra_body

        response = self._client.chat.completions.create(**request)
        self._record_usage(chosen, response)
        if not response.choices:
            raise EmptyCompletion(f"{chosen} returned no choices")
        choice = response.choices[0]
        text = strip_reasoning(choice.message.content or "")
        if not text:
            reason = getattr(choice, "finish_reason", None)
            hint = " (hit max_tokens while reasoning; raise max_tokens)" if reason == "length" else ""
            raise EmptyCompletion(f"{chosen} returned no answer text, finish_reason={reason}{hint}")
        return text
