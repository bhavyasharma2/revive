"""Tavily web search: finds the changelog or migration note behind a deprecation.

API reference: POST https://api.tavily.com/search with a Bearer token.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import httpx

TAVILY_URL = "https://api.tavily.com/search"
API_KEY_ENV = "TAVILY_API_KEY"
SEARCH_DEPTHS = ("basic", "advanced", "fast", "ultra-fast")


class TavilyError(RuntimeError):
    """A Tavily request failed."""


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    content: str
    score: float = 0.0


class TavilyClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        http: Optional[httpx.Client] = None,
        timeout: float = 30.0,
        url: str = TAVILY_URL,
    ) -> None:
        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise TavilyError(f"set the {API_KEY_ENV} environment variable (https://app.tavily.com)")
        self._key = key
        self._url = url
        self._http = http or httpx.Client(timeout=timeout)

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[Sequence[str]] = None,
    ) -> list[SearchHit]:
        """Search the web and return ranked hits (empty list if nothing was found)."""
        if not query or not query.strip():
            raise TavilyError("query must not be empty")
        if not 1 <= max_results <= 20:
            raise TavilyError("max_results must be between 1 and 20")
        if search_depth not in SEARCH_DEPTHS:
            raise TavilyError(f"search_depth must be one of {SEARCH_DEPTHS}")

        body: dict[str, Any] = {
            "query": query.strip(),
            "max_results": max_results,
            "search_depth": search_depth,
        }
        if include_domains:
            body["include_domains"] = list(include_domains)

        try:
            response = self._http.post(
                self._url,
                json=body,
                headers={"Authorization": f"Bearer {self._key}"},
            )
        except httpx.HTTPError as exc:
            raise TavilyError(f"could not reach Tavily: {exc}") from exc

        if response.status_code == 401:
            raise TavilyError("Tavily rejected the API key (401)")
        if response.status_code == 429:
            wait = response.headers.get("Retry-After", "unknown")
            raise TavilyError(f"Tavily rate limit hit (429); retry after {wait}s")
        if response.status_code in (432, 433):
            raise TavilyError(f"Tavily usage limit exceeded ({response.status_code})")
        if response.status_code >= 400:
            raise TavilyError(f"Tavily error {response.status_code}: {response.text[:200]}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise TavilyError("Tavily returned a non-JSON response") from exc

        hits: list[SearchHit] = []
        for item in payload.get("results", []) or []:
            url = item.get("url")
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title") or ""),
                    url=str(url),
                    content=str(item.get("content") or ""),
                    score=float(item.get("score") or 0.0),
                )
            )
        return hits
