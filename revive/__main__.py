"""Command line: python -m revive path/to/notebook.ipynb"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .agent import Outcome, ReviveAgent
from .env import EnvError, NotebookEnv
from .llm import MissingAPIKey, TokenFactoryClient
from .notebook import has_stored_outputs, load_notebook, save_notebook
from .search import TavilyClient, TavilyError
from .timemachine import git_last_changed, validate_date


def _printer(event: dict) -> None:
    kind = event["type"]
    if kind == "run_start":
        print(f"\n[run {event['round']}] executing notebook...")
    elif kind == "run_result":
        state = "ran cleanly" if event["ok"] else f"failed: {event['error']}"
        print(f"[run {event['round']}] {state} ({event['duration_s']}s)")
    elif kind == "search":
        print(f"[search] {event['query']}")
        for source in event["sources"][:3]:
            print(f"         - {source['url']}")
    elif kind == "thinking":
        print(f"[attempt {event['attempt']}] asking Nemotron for a fix...")
    elif kind == "patch":
        print(f"[attempt {event['attempt']}] {event['explanation']}")
        for line in event["diff"].splitlines():
            print(f"    {line}")
    elif kind == "patch_rejected":
        print(f"[attempt {event['attempt']}] rejected: {event['reason']}")
    elif kind == "install":
        print(f"[install] {'ok' if event['ok'] else 'FAILED'}: {' '.join(event['specs'])}")
    elif kind == "compare":
        print(f"[verify] {event['summary']}")
        for cell in event["cells"]:
            if cell["status"] in ("drift", "changed", "missing"):
                print(f"         cell {cell['cell_index']} {cell['status']}: {cell['detail']}")
    elif kind == "warning":
        print(f"[warning] {event['message']}")
    elif kind == "reconstruct_env":
        print(f"[time machine] attempt {event['attempt']}: rebuilding {event['description']}")
        if event.get("reasoning"):
            print(f"               {event['reasoning']}")
    elif kind == "reconstruct_adjust":
        print(f"[time machine] {event['message']}")
    elif kind == "reconstruct_result":
        if event["ok"]:
            line = f"[time machine] the untouched original runs there ({event.get('duration_s', '?')}s)"
            if event.get("vs_saved"):
                line += f"; against its saved outputs: {event['vs_saved']}"
            print(line)
        else:
            print(f"[time machine] attempt {event['attempt']} failed: {event['error'].splitlines()[0][:200]}")


VERDICTS = {
    Outcome.VERIFIED: "VERIFIED: the notebook runs and every reference output matches.",
    Outcome.ALREADY_WORKING: "Already working: it ran and matched without any change.",
    Outcome.RUNS_UNVERIFIED: "Runs, but the original had no saved outputs to verify against.",
    Outcome.DRIFT: "Runs, but NOT verified: some outputs differ from the saved ones (see above).",
    Outcome.FAILED: "Could not repair the notebook.",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="revive", description="Repair a broken notebook and verify the fix.")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--out", type=Path, help="where to write the repaired notebook (default: <name>.revived.ipynb)")
    parser.add_argument("--max-attempts", type=int, default=4, help="attempts in a row without progress before giving up")
    parser.add_argument("--max-total-attempts", type=int, default=12, help="hard cap while attempts keep making progress")
    parser.add_argument("--timeout", type=int, default=300, help="per-cell timeout in seconds")
    parser.add_argument("--no-search", action="store_true", help="do not use Tavily web search")
    parser.add_argument("--no-env", action="store_true", help="run in the current environment (no pip installs)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance for numeric outputs")
    parser.add_argument(
        "--time-machine",
        choices=("auto", "always", "off"),
        default="auto",
        help="rebuild the notebook's original environment to get reference outputs "
             "(auto: only when the notebook has no saved outputs)",
    )
    parser.add_argument(
        "--written-before",
        metavar="YYYY-MM-DD",
        help="when the notebook was last changed; the Time Machine only uses packages released before it "
             "(default: the date of the notebook's last git commit, if it is in a git repository)",
    )
    args = parser.parse_args(argv)

    if not args.notebook.is_file():
        print(f"error: {args.notebook} not found", file=sys.stderr)
        return 2

    try:
        llm = TokenFactoryClient()
    except MissingAPIKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    search = None
    if not args.no_search:
        try:
            search = TavilyClient()
        except TavilyError as exc:
            print(f"[warning] web search disabled: {exc}")

    try:
        as_of = validate_date(args.written_before)
    except ValueError as exc:
        print(f"error: --written-before: {exc}", file=sys.stderr)
        return 2
    if as_of is None:
        as_of = git_last_changed(args.notebook)
        if as_of:
            print(f"[time machine] last git commit of this notebook: {as_of}")

    nb = load_notebook(args.notebook)
    if not has_stored_outputs(nb):
        if args.time_machine == "off":
            print("[warning] this notebook has no saved outputs, so a fix can be run but not verified")
        else:
            print("[time machine] no saved outputs: rebuilding the original environment to get reference outputs")

    env = None
    tmp = None
    if not args.no_env:
        tmp = tempfile.TemporaryDirectory(prefix="revive-env-")
        print("[env] creating an isolated environment...")
        try:
            env = NotebookEnv.create(Path(tmp.name) / "venv")
        except EnvError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    agent = ReviveAgent(
        llm,
        search=search,
        env=env,
        max_attempts=args.max_attempts,
        max_total_attempts=args.max_total_attempts,
        cell_timeout=args.timeout,
        cwd=str(args.notebook.resolve().parent),
        on_event=_printer,
        rtol=args.rtol,
        time_machine=args.time_machine,
        as_of=as_of,
    )
    try:
        result = agent.repair(nb)
    finally:
        if tmp is not None:
            tmp.cleanup()

    out = args.out or args.notebook.with_name(args.notebook.stem + ".revived.ipynb")
    report_path = out.with_suffix(".report.json")
    if result.runs:
        save_notebook(result.notebook, out)
        print(f"\nrepaired notebook: {out}")
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    print(VERDICTS[result.outcome])
    if result.reference == "reconstructed":
        print("(reference outputs came from running the original notebook in its rebuilt environment)")
    return 0 if result.outcome in (Outcome.VERIFIED, Outcome.ALREADY_WORKING) else 1


if __name__ == "__main__":
    raise SystemExit(main())
