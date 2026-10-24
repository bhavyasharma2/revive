"""Web API with a fake repair function (no model, no network)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from revive.agent import Outcome
from revive.notebook import dumps_notebook
from revive.web.app import create_app, load_recordings
from revive.web.jobs import CachedRun, JobManager, LimitExceeded, RateLimiter

from conftest import make_notebook, stdout_output


def fake_result(nb, outcome=Outcome.VERIFIED):
    return SimpleNamespace(
        outcome=outcome,
        runs=outcome != Outcome.FAILED,
        notebook=nb,
        to_dict=lambda: {"outcome": outcome.value, "attempts": []},
    )


def fake_repair(nb, emit, options=None):
    emit({"type": "run_start", "round": 0})
    emit({"type": "run_result", "round": 0, "ok": True, "failed_cell": None, "error": "ok", "duration_s": 0.1})
    emit({"type": "done", "outcome": "verified", "attempts": 0, "last_error": ""})
    return fake_result(nb), {"fake": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5}}


def failing_repair(nb, emit, options=None):
    raise RuntimeError("token factory is down")


def test_time_machine_option_reaches_the_repair_and_has_its_own_cache(tmp_path):
    seen = []

    def recording_repair(nb, emit, options=None):
        seen.append(options)
        return fake_repair(nb, emit, options)

    app = create_app(recording_repair, recordings=tmp_path)
    with TestClient(app) as c:
        plain = c.post("/api/jobs", json={"sample": "numpy_asscalar"}).json()
        read_events(c, plain["id"])
        forced = c.post("/api/jobs", json={"sample": "numpy_asscalar", "time_machine": True}).json()
        read_events(c, forced["id"])
        assert forced["live"] is True          # the plain run's recording is not reused for a Time Machine run
        assert seen == [{"time_machine": "auto", "as_of": None}, {"time_machine": "always", "as_of": None}]


def test_time_machine_recordings_load_under_their_own_key(tmp_path):
    rec = {"events": [{"type": "notebook", "cells": []}], "notebook_json": None, "report": None,
           "outcome": "verified", "recorded_at": 1761200000.0}
    (tmp_path / "pandas_append.tm.json").write_text(json.dumps(rec))
    app = create_app(fake_repair, recordings=tmp_path)
    with TestClient(app) as c:
        sample = {s["name"]: s for s in c.get("/api/samples").json()}["pandas_append"]
        assert sample["recorded_at"] is None and sample["recorded_tm_at"] == 1761200000.0
        assert c.post("/api/jobs", json={"sample": "pandas_append", "time_machine": True}).json()["live"] is False


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIVE_LIVE_PER_CLIENT", "3")
    monkeypatch.delenv("REVIVE_ALLOW_UPLOADS", raising=False)
    monkeypatch.delenv("REVIVE_LIVE", raising=False)
    app = create_app(fake_repair, recordings=tmp_path)
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


def read_events(client, job_id):
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        kind = "message"
        for line in response.iter_lines():
            if line.startswith("event: "):
                kind = line[len("event: "):]
            elif line.startswith("data: "):
                events.append((kind, json.loads(line[len("data: "):])))
                if kind == "end":
                    break
                kind = "message"
    return events


def test_index_and_config(client):
    page = client.get("/")
    assert page.status_code == 200 and "Revive" in page.text
    assert client.get("/api/config").json() == {"allow_uploads": False, "live_enabled": True}
    assert client.get("/healthz").json() == {"ok": True}


def test_samples_listed_from_benchmark(client):
    samples = client.get("/api/samples").json()
    names = {s["name"] for s in samples}
    assert "legacy_churn_model" in names and "sklearn_kmeans_silent" in names
    assert all(s["recorded_at"] is None for s in samples)


def test_live_sample_job_streams_events_and_offers_download(client):
    job = client.post("/api/jobs", json={"sample": "pandas_append"}).json()
    assert job["live"] is True
    events = read_events(client, job["id"])
    kinds = [e[1]["type"] for e in events if e[0] == "message"]
    assert kinds[0] == "notebook" and kinds[-1] == "done"
    notebook_event = events[0][1]
    assert any(c["type"] == "code" and "append" in c["source"] for c in notebook_event["cells"])
    assert events[-1][0] == "end" and events[-1][1]["outcome"] == "verified"

    download = client.get(f"/api/jobs/{job['id']}/notebook")
    assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
    assert json.loads(download.text)["nbformat"] == 4
    report = client.get(f"/api/jobs/{job['id']}/report").json()
    assert report["usage"]["fake"]["calls"] == 1


def test_verified_live_run_is_cached_and_replayed(client):
    first = client.post("/api/jobs", json={"sample": "numpy_asscalar"}).json()
    read_events(client, first["id"])
    assert client.get("/api/samples").json()  # still fine
    second = client.post("/api/jobs", json={"sample": "numpy_asscalar"}).json()
    assert second["live"] is False and second["recorded_at"] is not None
    replay = read_events(client, second["id"])
    assert [e[1]["type"] for e in replay if e[0] == "message"][-1] == "done"
    assert client.get(f"/api/jobs/{second['id']}/notebook").status_code == 200
    # asking for live explicitly bypasses the cache
    third = client.post("/api/jobs", json={"sample": "numpy_asscalar", "live": True}).json()
    assert third["live"] is True


def test_per_client_live_limit(client):
    for _ in range(3):
        assert client.post("/api/jobs", json={"sample": "pandas_iteritems", "live": True}).status_code == 201
    blocked = client.post("/api/jobs", json={"sample": "pandas_iteritems", "live": True})
    assert blocked.status_code == 429 and "live repairs" in blocked.json()["detail"]


def test_uploads_are_off_by_default(client):
    nb = dumps_notebook(make_notebook("print(1)"))
    response = client.post("/api/jobs", json={"notebook": nb})
    assert response.status_code == 403


def test_uploads_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIVE_ALLOW_UPLOADS", "1")
    app = create_app(fake_repair, recordings=tmp_path)
    with TestClient(app) as c:
        nb = dumps_notebook(make_notebook(("print(1)", [stdout_output("1\n")])))
        job = c.post("/api/jobs", json={"notebook": nb}).json()
        assert job["live"] is True and job["label"] == "Uploaded notebook"
        assert c.post("/api/jobs", json={"notebook": "not json"}).status_code == 400
        assert c.post("/api/jobs", json={"notebook": "x" * 2_000_001}).status_code == 413


def test_bad_requests(client):
    assert client.post("/api/jobs", json={}).status_code == 400
    assert client.post("/api/jobs", json={"sample": "nope"}).status_code == 404
    assert client.post("/api/jobs", json={"sample": "pandas_append", "notebook": "{}"}).status_code == 400
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/api/jobs/missing/events").status_code == 404
    assert client.get("/api/jobs/missing/notebook").status_code == 404


def test_live_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIVE_LIVE", "0")
    app = create_app(fake_repair, recordings=tmp_path)
    with TestClient(app) as c:
        response = c.post("/api/jobs", json={"sample": "pandas_append"})
        assert response.status_code == 403 and "replays" in response.json()["detail"]


def test_repair_crash_is_reported_to_the_browser(tmp_path):
    app = create_app(failing_repair, recordings=tmp_path)
    with TestClient(app) as c:
        job = c.post("/api/jobs", json={"sample": "pandas_append"}).json()
        events = read_events(c, job["id"])
        errors = [e[1] for e in events if e[0] == "message" and e[1]["type"] == "error"]
        assert errors and "token factory is down" in errors[0]["message"]
        assert events[-1][1]["error"].startswith("RuntimeError")


def test_recordings_are_loaded_for_replay(tmp_path):
    recording = {
        "events": [{"type": "notebook", "cells": []}, {"type": "done", "outcome": "verified", "attempts": 1, "last_error": ""}],
        "notebook_json": dumps_notebook(make_notebook("print(1)")),
        "report": {"outcome": "verified"},
        "outcome": "verified",
        "recorded_at": 1761200000.0,
    }
    (tmp_path / "pandas_append.json").write_text(json.dumps(recording))
    (tmp_path / "broken.json").write_text("{not json")
    app = create_app(fake_repair, recordings=tmp_path)
    with TestClient(app) as c:
        samples = {s["name"]: s for s in c.get("/api/samples").json()}
        assert samples["pandas_append"]["recorded_at"] == 1761200000.0
        job = c.post("/api/jobs", json={"sample": "pandas_append"}).json()
        assert job["live"] is False


def test_rate_limiter_resets_daily():
    now = [0.0]
    limiter = RateLimiter(per_client=1, per_day=2, clock=lambda: now[0])
    limiter.check_and_count("a")
    with pytest.raises(LimitExceeded):
        limiter.check_and_count("a")
    limiter.check_and_count("b")
    with pytest.raises(LimitExceeded):
        limiter.check_and_count("c")  # global cap
    now[0] = 86400.0
    limiter.check_and_count("a")


def test_busy_server_rejects_new_live_jobs():
    import threading

    gate = threading.Event()

    def slow(nb, emit, options=None):
        gate.wait(5)
        return fake_result(nb), {}

    manager = JobManager(slow, max_concurrent=1, max_queued=0)
    nb = make_notebook("print(1)")
    manager.start(nb, label="a", client="x")
    with pytest.raises(LimitExceeded, match="busy"):
        manager.start(nb, label="b", client="y")
    gate.set()
    manager.shutdown()


def test_replay_paces_events_and_copies_results():
    manager = JobManager(fake_repair, replay_step_s=0.0)
    manager.seed_cache("k", CachedRun(events=[{"type": "notebook", "cells": []}, {"type": "done", "outcome": "verified"}],
                                      notebook_json="{}", report={"r": 1}, outcome="verified", recorded_at=1.0))
    job = manager.start(make_notebook("1"), label="x", client="c", cache_key="k", live=False)
    for _ in range(100):
        if job.done:
            break
        time.sleep(0.01)
    assert job.done and job.outcome == "verified" and job.report == {"r": 1} and len(job.events) == 2
    manager.shutdown()


def test_sample_dates_from_the_manifest_reach_the_repair(tmp_path):
    seen = []

    def recording_repair(nb, emit, options=None):
        seen.append(options)
        return fake_repair(nb, emit, options)

    app = create_app(recording_repair, recordings=tmp_path)
    with TestClient(app) as c:
        samples = {s["name"]: s for s in c.get("/api/samples").json()}
        assert samples["sklearn_kmeans_silent_no_outputs"]["as_of"] == "2023-02-01"
        job = c.post("/api/jobs", json={"sample": "sklearn_kmeans_silent_no_outputs"}).json()
        read_events(c, job["id"])
        assert seen[-1]["as_of"] == "2023-02-01"
        job = c.post("/api/jobs", json={"sample": "pandas_append"}).json()
        read_events(c, job["id"])
        assert seen[-1]["as_of"] is None


def test_head_requests_for_uptime_monitors():
    from fastapi.testclient import TestClient
    from revive.web.app import create_app

    client = TestClient(create_app())
    for path in ("/", "/healthz"):
        assert client.head(path).status_code == 200
        assert client.get(path).status_code == 200
