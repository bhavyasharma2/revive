"""Run Revive on every benchmark notebook and write a results table.

Run:  python scripts/run_benchmark.py            (all notebooks)
      python scripts/run_benchmark.py --only pandas_append sklearn_ovr_trap
      python scripts/run_benchmark.py --set real     (real public notebooks, benchmark/real/)

Needs NEBIUS_API_KEY (and TAVILY_API_KEY for web search). Writes
benchmark/results/results.json, benchmark/results/RESULTS.md and the repaired
notebooks in benchmark/results/revived/.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from revive.agent import Outcome, ReviveAgent  # noqa: E402
from revive.compare import Status, compare_notebooks  # noqa: E402
from revive.env import NotebookEnv  # noqa: E402
from revive.llm import TokenFactoryClient  # noqa: E402
from revive.notebook import dumps_notebook, load_notebook, save_notebook  # noqa: E402
from revive.search import TavilyClient, TavilyError  # noqa: E402
from revive.web.jobs import notebook_event  # noqa: E402

BENCH = ROOT / "benchmark"
RECORDINGS = BENCH / "recordings"
SETS = {
    # name: (folder with manifest.json and notebooks/, results folder, record failed runs too)
    "benchmark": (BENCH, BENCH / "results", False),
    "real": (BENCH / "real", BENCH / "real" / "results", True),
}
RESULTS = BENCH / "results"
RECORD_ALL = False
MAX_ATTEMPTS = 4


def reconstruction_vs_real(result, entry: Optional[dict]) -> Optional[str]:
    """For notebooks saved without outputs we still know the real outputs (the base notebook's):
    does the Time Machine's rebuilt original reproduce them?"""
    if not entry or entry.get("kind") != "no_outputs" or not entry.get("base"):
        return None
    rec = result.reconstruction
    if rec is None or not rec.ok or rec.reference is None:
        return "no reconstruction"
    real = load_notebook(BENCH / "notebooks" / f"{entry['base']}.ipynb")
    return compare_notebooks(real, rec.reference).summary()


def run_one(path: Path, llm: TokenFactoryClient, search, args, entry: Optional[dict] = None) -> dict:
    nb = load_notebook(path)
    llm.reset_usage()
    events: list[dict] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="revive-bench-") as tmp:
        env = None if args.no_env else NotebookEnv.create(Path(tmp) / "venv")
        work = Path(tmp) / "work"  # files the notebook writes (figures, CSVs) go here, not into the repo
        work.mkdir()
        agent = ReviveAgent(
            llm,
            search=search,
            env=env,
            max_attempts=args.max_attempts,
            max_total_attempts=getattr(args, "max_total_attempts", None),
            cell_timeout=args.timeout,
            cwd=str(work),
            on_event=events.append,
            as_of=(entry or {}).get("as_of"),
        )
        try:
            result = agent.repair(nb)
        except Exception as exc:  # noqa: BLE001 - one broken run must not stop the benchmark
            return {"name": path.stem, "outcome": "error", "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.monotonic() - started, 1), "usage": llm.usage}

    if result.runs:
        (RESULTS / "revived").mkdir(parents=True, exist_ok=True)
        save_notebook(result.notebook, RESULTS / "revived" / path.name)

    if result.outcome == Outcome.VERIFIED or RECORD_ALL:
        # A recording of this live run, replayed by the web app without spending credits.
        RECORDINGS.mkdir(parents=True, exist_ok=True)
        recording = {
            "events": [notebook_event(nb)] + events,
            "notebook_json": dumps_notebook(result.notebook),
            "report": {**result.to_dict(), "usage": llm.usage},
            "outcome": result.outcome.value,
            "recorded_at": time.time(),
        }
        (RECORDINGS / f"{path.stem}.json").write_text(json.dumps(recording), encoding="utf-8")

    ran_at_some_point = any(e["type"] == "run_result" and e["ok"] for e in events)
    comparison = result.comparison
    return {
        "name": path.stem,
        "outcome": result.outcome.value,
        "ran_after_repair": ran_at_some_point,
        "verified": result.outcome == Outcome.VERIFIED,
        "drift_caught": any(e["type"] == "compare" and not e["verified"] and not e["unverifiable"] for e in events),
        "attempts": len(result.attempts),
        "cells_matched": f"{comparison.count(Status.MATCH)}/{comparison.compared}" if comparison else "",
        "reference": result.reference,
        "time_machine": result.reconstruction.to_dict() if result.reconstruction else None,
        "reconstruction_vs_real_outputs": reconstruction_vs_real(result, entry),
        "seconds": round(time.monotonic() - started, 1),
        "installed": result.installed,
        "usage": llm.usage,
        "explanations": [a.explanation for a in result.attempts if a.explanation],
        "attempt_results": [f"{a.number}: {a.error} -> {a.result}" for a in result.attempts],
        "last_error": result.last_error,
    }


def write_markdown(rows: list[dict], manifest: dict[str, dict]) -> str:
    total = len(rows)
    verified = sum(1 for r in rows if r.get("verified"))
    ran = sum(1 for r in rows if r.get("ran_after_repair"))
    caught = sum(1 for r in rows if r.get("drift_caught"))
    tokens = sum(u["prompt_tokens"] + u["completion_tokens"] for r in rows for u in r.get("usage", {}).values())
    no_out = [r for r in rows if manifest.get(r["name"], {}).get("kind") == "no_outputs"]
    no_out_verified = sum(1 for r in no_out if r.get("verified"))
    lines = [
        "# Revive benchmark results" if not RECORD_ALL else "# Revive on real public notebooks",
        "",
        f"- Notebooks: **{total}**, all broken against current NumPy/pandas/scikit-learn. Saved outputs are authentic: produced by the old code on old libraries."
        if not RECORD_ALL else
        f"- Notebooks: **{total}**, written by other people (see benchmark/real/README.md). Saved outputs are the authors' own.",
        f"- Verified repairs: **{verified}/{total}**",
    ]
    if no_out:
        faithful = sum(1 for r in no_out if str(r.get("reconstruction_vs_real_outputs", "")).startswith("verified"))
        lines.append(f"- Saved without outputs, verified against the Time Machine's rebuilt original environment: **{no_out_verified}/{len(no_out)}**")
        lines.append(f"- Time Machine faithfulness: the rebuilt original reproduced the author's real (hidden) outputs in **{faithful}/{len(no_out)}**")
    lines += [
        f"- Ran at some point during repair: {ran}/{total}",
        f"- Times a fix ran but the output check caught different numbers: {caught}",
        f"- Total tokens: {tokens:,}",
        f"- Attempts allowed per notebook: {MAX_ATTEMPTS}",
        "",
        "| Notebook | What broke | Checked against | Outcome | Attempts | Cells matched | Time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    reference_names = {"saved": "saved outputs", "reconstructed": "rebuilt original", "none": "nothing"}
    for r in rows:
        breaks = manifest.get(r["name"], {}).get("breaks", "")
        reference = reference_names.get(r.get("reference", ""), r.get("reference", ""))
        lines.append(
            f"| {r['name']} | {breaks} | {reference} | {r['outcome']} | {r.get('attempts', '')} | {r.get('cells_matched', '')} | {r.get('seconds', '')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="notebook names to run (default: all)")
    parser.add_argument("--max-attempts", type=int, default=4, help="attempts in a row without progress before giving up")
    parser.add_argument("--max-total-attempts", type=int, default=12, help="hard cap while attempts keep making progress")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-search", action="store_true")
    parser.add_argument("--no-env", action="store_true")
    parser.add_argument("--set", choices=sorted(SETS), default="benchmark", help="which notebooks: the benchmark (default) or real public notebooks")
    args = parser.parse_args()

    global RESULTS, RECORD_ALL, MAX_ATTEMPTS
    folder, RESULTS, RECORD_ALL = SETS[args.set]
    MAX_ATTEMPTS = f"up to {args.max_total_attempts}, stopping after {args.max_attempts} in a row without progress"
    manifest = {m["name"]: m for m in json.loads((folder / "manifest.json").read_text(encoding="utf-8"))}
    names = args.only or list(manifest)
    unknown = [n for n in names if n not in manifest]
    if unknown:
        print(f"unknown notebook(s): {unknown}. Available: {list(manifest)}")
        return 2

    llm = TokenFactoryClient()
    search = None
    if not args.no_search:
        try:
            search = TavilyClient()
        except TavilyError as exc:
            print(f"web search disabled: {exc}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        print(f"{name:<32} ...", end="", flush=True)
        row = run_one(folder / "notebooks" / f"{name}.ipynb", llm, search, args, manifest[name])
        rows.append(row)
        print(f" {row['outcome']:<16} attempts={row.get('attempts', '-')}  {row['seconds']}s")

    (RESULTS / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    markdown = write_markdown(rows, manifest)
    (RESULTS / "RESULTS.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
