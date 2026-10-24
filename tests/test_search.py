from __future__ import annotations

import json

import httpx
import pytest

from revive.search import TavilyClient, TavilyError


def make_client(handler, key="tvly-test"):
    transport = httpx.MockTransport(handler)
    return TavilyClient(api_key=key, http=httpx.Client(transport=transport))


def test_search_sends_bearer_auth_and_parses_results():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "query": "q",
                "results": [
                    {"title": "NumPy 1.24 notes", "url": "https://numpy.org/doc/1.24", "content": "np.float removed", "score": 0.9},
                    {"title": "no url", "content": "dropped"},
                ],
            },
        )

    hits = make_client(handler).search("  numpy np.float removed ", max_results=3, include_domains=["numpy.org"])
    assert seen["auth"] == "Bearer tvly-test"
    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["body"] == {
        "query": "numpy np.float removed",
        "max_results": 3,
        "search_depth": "basic",
        "include_domains": ["numpy.org"],
    }
    assert len(hits) == 1
    assert hits[0].url == "https://numpy.org/doc/1.24" and hits[0].score == 0.9


def test_empty_results_return_empty_list():
    client = make_client(lambda request: httpx.Response(200, json={"results": []}))
    assert client.search("anything") == []


@pytest.mark.parametrize(
    "status,fragment",
    [(401, "API key"), (429, "rate limit"), (432, "usage limit"), (433, "usage limit"), (500, "500")],
)
def test_http_errors_become_tavily_errors(status, fragment):
    client = make_client(lambda request: httpx.Response(status, headers={"Retry-After": "7"}, text="boom"))
    with pytest.raises(TavilyError) as err:
        client.search("x")
    assert fragment in str(err.value)


def test_rate_limit_message_includes_retry_after():
    client = make_client(lambda request: httpx.Response(429, headers={"Retry-After": "7"}))
    with pytest.raises(TavilyError, match="7"):
        client.search("x")


def test_non_json_body_is_an_error():
    client = make_client(lambda request: httpx.Response(200, text="<html>"))
    with pytest.raises(TavilyError):
        client.search("x")


def test_network_failure_is_an_error():
    def handler(request):
        raise httpx.ConnectError("down")

    with pytest.raises(TavilyError, match="could not reach"):
        make_client(handler).search("x")


@pytest.mark.parametrize(
    "kwargs",
    [{"query": ""}, {"query": "   "}, {"query": "x", "max_results": 0}, {"query": "x", "max_results": 21}, {"query": "x", "search_depth": "deep"}],
)
def test_invalid_arguments_are_rejected_before_any_request(kwargs):
    def handler(request):
        raise AssertionError("no request should be sent")

    with pytest.raises(TavilyError):
        make_client(handler).search(**kwargs)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(TavilyError):
        TavilyClient()
