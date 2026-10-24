"""Check your keys and the live APIs before building on them.

Run:  NEBIUS_API_KEY=... TAVILY_API_KEY=... python scripts/smoke_test.py

Tavily is optional. Exit code is 0 only if every check that ran passed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from revive.llm import (  # noqa: E402
    ROLE_RULES,
    MissingAPIKey,
    ModelNotAvailable,
    TokenFactoryClient,
)
from revive.search import API_KEY_ENV as TAVILY_ENV  # noqa: E402
from revive.search import TavilyClient, TavilyError  # noqa: E402


def main() -> int:
    failures = 0

    print("== Token Factory ==")
    try:
        client = TokenFactoryClient()
    except MissingAPIKey as exc:
        print(f"FAIL  {exc}")
        return 1

    try:
        models = client.available_models()
    except Exception as exc:  # noqa: BLE001 - this script reports every failure kind
        print(f"FAIL  could not list models: {type(exc).__name__}: {exc}")
        return 1
    print(f"PASS  {len(models)} models available")
    nvidia = [m for m in models if "nvidia" in m.lower() or "nemotron" in m.lower()]
    print("      NVIDIA models:")
    for model_id in nvidia:
        print(f"        {model_id}")

    for role in ROLE_RULES:
        try:
            model_id = client.model_for(role)
        except ModelNotAvailable as exc:
            print(f"FAIL  role '{role}': {exc}")
            failures += 1
            continue
        try:
            reply = client.chat(
                [{"role": "user", "content": "Reply with exactly the single word: pong"}],
                model=model_id,
                max_tokens=1024,
                temperature=0.0,
            )
            print(f"PASS  role '{role}' -> {model_id}: {reply[:60]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  role '{role}' -> {model_id}: {type(exc).__name__}: {exc}")
            failures += 1

    print("== Tavily ==")
    if not os.environ.get(TAVILY_ENV):
        print(f"SKIP  {TAVILY_ENV} is not set")
    else:
        try:
            hits = TavilyClient().search("numpy 1.24 release notes np.float removed", max_results=3)
            print(f"PASS  {len(hits)} results; first: {hits[0].url if hits else 'none'}")
        except TavilyError as exc:
            print(f"FAIL  {exc}")
            failures += 1

    print("== Result ==")
    print("all checks passed" if failures == 0 else f"{failures} check(s) failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
