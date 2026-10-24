"""Background repair jobs, their event logs, a replay cache and usage limits."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nbformat import NotebookNode

from ..compare import comparable_text
from ..notebook import dumps_notebook

# Builds and runs one repair. Receives the notebook, an event callback and an
# options dict (e.g. {"time_machine": "always"}) and returns (RepairResult,
# usage dict). The default lives in app.py; tests inject fakes.
RepairFn = Callable[[NotebookNode, Callable[[dict], None], dict], tuple[Any, dict]]

REPLAY_STEP_S = 0.35


class LimitExceeded(RuntimeError):
    """A usage limit was hit; the message says which and when it resets."""


def notebook_event(nb: NotebookNode) -> dict:
    """First event of every job: the cells as the author saved them."""
    cells = []
    for index, cell in enumerate(nb.cells):
        entry = {"index": index, "type": cell.cell_type, "source": cell.source}
        if cell.cell_type == "code":
            entry["saved_output"] = comparable_text(cell.get("outputs", []))
        cells.append(entry)
    return {"type": "notebook", "cells": cells}


@dataclass
class Job:
    id: str
    label: str
    live: bool
    created: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    done: bool = False
    outcome: Optional[str] = None
    error: Optional[str] = None
    notebook_json: Optional[str] = None
    report: Optional[dict] = None
    recorded_at: Optional[float] = None

    def emit(self, event: dict) -> None:
        self.events.append(event)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "live": self.live,
            "done": self.done,
            "outcome": self.outcome,
            "error": self.error,
            "recorded_at": self.recorded_at,
            "has_notebook": self.notebook_json is not None,
        }


@dataclass
class CachedRun:
    events: list[dict]
    notebook_json: Optional[str]
    report: Optional[dict]
    outcome: str
    recorded_at: float


class RateLimiter:
    """Fixed daily windows: per client and global."""

    def __init__(self, per_client: int, per_day: int, clock: Callable[[], float] = time.time) -> None:
        self.per_client = per_client
        self.per_day = per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._day: Optional[int] = None
        self._by_client: dict[str, int] = {}
        self._total = 0

    def _roll(self) -> None:
        day = int(self._clock() // 86400)
        if day != self._day:
            self._day, self._by_client, self._total = day, {}, 0

    def check_and_count(self, client: str) -> None:
        with self._lock:
            self._roll()
            if self._total >= self.per_day:
                raise LimitExceeded("the daily limit for live repairs has been reached; replays still work, and it resets at 00:00 UTC")
            used = self._by_client.get(client, 0)
            if used >= self.per_client:
                raise LimitExceeded(f"you have used your {self.per_client} live repairs for today; replays still work, and it resets at 00:00 UTC")
            self._by_client[client] = used + 1
            self._total += 1


class JobManager:
    def __init__(
        self,
        repair: RepairFn,
        *,
        max_concurrent: int = 2,
        max_queued: int = 6,
        limiter: Optional[RateLimiter] = None,
        replay_step_s: float = REPLAY_STEP_S,
        max_jobs_kept: int = 200,
    ) -> None:
        self._repair = repair
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="revive-job")
        self._max_active = max_concurrent + max_queued
        self._limiter = limiter
        self._replay_step = replay_step_s
        self._max_kept = max_jobs_kept
        self._lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.cache: dict[str, CachedRun] = {}

    # -- bookkeeping -------------------------------------------------------------

    def _active(self) -> int:
        return sum(1 for job in self.jobs.values() if not job.done)

    def _register(self, job: Job) -> None:
        with self._lock:
            self.jobs[job.id] = job
            if len(self.jobs) > self._max_kept:
                finished = sorted((j for j in self.jobs.values() if j.done), key=lambda j: j.created)
                for old in finished[: len(self.jobs) - self._max_kept]:
                    self.jobs.pop(old.id, None)

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    # -- starting jobs -----------------------------------------------------------

    def start(
        self,
        nb: NotebookNode,
        *,
        label: str,
        client: str,
        cache_key: Optional[str] = None,
        live: bool = True,
        options: Optional[dict] = None,
    ) -> Job:
        """Start a live repair, or a replay when ``live`` is False and a cached run exists."""
        cached = self.cache.get(cache_key) if cache_key else None
        if not live and cached is not None:
            job = Job(id=uuid.uuid4().hex[:12], label=label, live=False, recorded_at=cached.recorded_at)
            self._register(job)
            threading.Thread(target=self._replay, args=(job, cached), daemon=True).start()
            return job

        with self._lock:
            if self._active() >= self._max_active:
                raise LimitExceeded("the server is busy with other repairs; try again in a minute")
        if self._limiter is not None:
            self._limiter.check_and_count(client)

        job = Job(id=uuid.uuid4().hex[:12], label=label, live=True)
        self._register(job)
        self._pool.submit(self._run, job, copy.deepcopy(nb), cache_key, dict(options or {}))
        return job

    def _run(self, job: Job, nb: NotebookNode, cache_key: Optional[str], options: dict) -> None:
        job.emit(notebook_event(nb))
        try:
            result, usage = self._repair(nb, job.emit, options)
            job.outcome = result.outcome.value
            job.report = {**result.to_dict(), "usage": usage}
            if result.runs:
                job.notebook_json = dumps_notebook(result.notebook)
            if cache_key and job.outcome == "verified":
                self.cache[cache_key] = CachedRun(
                    events=list(job.events),
                    notebook_json=job.notebook_json,
                    report=job.report,
                    outcome=job.outcome,
                    recorded_at=time.time(),
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser, never swallowed silently
            job.error = f"{type(exc).__name__}: {exc}"
            job.emit({"type": "error", "message": job.error})
        finally:
            job.done = True

    def _replay(self, job: Job, cached: CachedRun) -> None:
        for event in cached.events:
            job.emit(event)
            if event.get("type") not in ("notebook", "run_start"):
                time.sleep(self._replay_step)
        job.notebook_json = cached.notebook_json
        job.report = cached.report
        job.outcome = cached.outcome
        job.done = True

    def seed_cache(self, key: str, run: CachedRun) -> None:
        self.cache[key] = run

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
