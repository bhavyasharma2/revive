# Revive

**Revive repairs ML notebooks that broke when their libraries changed, then proves the fix:** every cell is re-run and checked against what the notebook printed before it broke. When a notebook was shared without its outputs, Revive's **Time Machine** rebuilds the environment it was written for, runs the untouched original there, and uses that as the reference.

Built for the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com/) (Coding and Agentic Engineering track) with **NVIDIA Nemotron** models on **Nebius Token Factory** and **Tavily** web search.

- **Live demo:** _(added after deployment)_
- **Demo video:** _(added after upload)_

![The Time Machine verifying a KMeans notebook that was saved without outputs](docs/time-machine.png)

## The problem

Notebooks are how ML work is shared, and they rot. NumPy 2 removed `np.float` and `np.NaN`, pandas 2 and 3 removed `DataFrame.append`, `.ix` and `fillna(method=...)`, scikit-learn renamed parameters and changed defaults. One recent study found that only 35.1% of ML engineering notebooks it tested were reproducible ([MLEModernizer, 2026](https://arxiv.org/abs/2602.07195)).

Getting an old notebook to *run* again is the easy half. A fix can execute cleanly and still compute something different. Delete the removed `multi_class='ovr'` argument from a `LogisticRegression` and the code runs, but silently switches algorithm. Upgrade past scikit-learn 1.4 and `KMeans` gives worse clusters with no error at all, because its `n_init` default changed. Nothing crashes; the numbers are just wrong.

And many notebooks are shared with outputs cleared, so there is nothing to compare against. A notebook like the KMeans one above runs fine today: every tool that only checks "does it run" calls it working.

## What Revive does

1. **Runs the notebook in an isolated environment** and finds the first cell that fails.
2. **Researches the failure.** Nemotron Nano writes a search query; Tavily finds the changelog, migration guide or issue behind the change. Deprecation warnings the notebook printed in its own era are passed along too: they name what is about to break.
3. **Writes a fix.** Nemotron Ultra reads the code, the error, the library versions and that evidence, and returns a patch: exact snippet replacements in the broken cells and, only when needed, pinned package versions. Patches are checked before anything runs: invalid Python is rejected, and cells that already reproduce their reference output are locked against edits.
4. **Re-runs and repeats** until the notebook runs.
5. **Verifies every cell** against the reference, with a numeric tolerance. If a fix runs but changes the numbers, Revive shows the model exactly what the reference says and what it printed, searches for the behaviour change, and tries again. A notebook is only called **verified** when every reference output matches.

### The Time Machine

The reference is the outputs the author saved. When there are none, Revive recovers them:

1. **Nemotron infers the original environment** from the evidence in the notebook: the Python version in its metadata, warnings it printed, the APIs it calls (for example: `np.float` means NumPy before 1.24, `DataFrame.append` means pandas before 2.0) and, when known, **when it was last changed**. For a notebook in a git repository that date is its last commit; it can also be given with `--written-before`.
2. **uv builds that environment** (an old Python plus exact old library versions) in seconds. With a date, uv only allows packages released before it (`--exclude-newer`), so even libraries Nemotron leaves unpinned resolve to what existed back then.
3. **The untouched original runs there.** If it fails, the error goes back to Nemotron for a better guess.
4. **Its outputs become the reference** the modernised notebook must reproduce.

Why the date matters: some changes leave no trace in the code. The KMeans notebook calls the same API before and after scikit-learn 1.4 changed its default; only the era tells the two apart. In our first benchmark run without dates, Nemotron reasonably chose a recent scikit-learn for it, which already has the new behaviour, so the rebuilt "original" could not expose the change. With the date, it does.

When a notebook does have saved outputs, the Time Machine can also be switched on to check them: the original should reproduce its own saved outputs in the rebuilt environment. Doing exactly that is how we found a mistake in an early version of our own benchmark (see below).

```
broken notebook ──► run in sandbox ──► error? ──► Nemotron Nano: search query ──► Tavily: changelog
       │                  ▲                                                          │
       │                  │                                                          ▼
       │                  └──── check + apply patch ◄──── Nemotron Ultra: patch (replacements, pins)
       │                  │
       │                  ▼ runs
       │        compare every cell with the reference ──► drift? ──► back to Nemotron: reference vs. now
       │                  │ all match
       │                  ▼
       │              VERIFIED  (+ repaired notebook and a JSON report of every attempt and source)
       │
       └─ no saved outputs? ──► Time Machine: Nemotron infers Python + library versions ──► uv builds them
                                ──► untouched original runs there ──► its outputs are the reference
```

Harmless differences are ignored: timings, progress bars, memory addresses, whitespace, warnings on stderr, NumPy 2's `np.float64(0.5)` way of printing `0.5`, and a difference of one unit in the last printed digit of numbers printed with three or more decimals (two numerically equivalent solvers can print `0.182` and `0.181`).

## How NVIDIA Nemotron and Nebius Token Factory are used

| Step | Model (served by Token Factory) | Why this model |
|---|---|---|
| Infer the original environment (Time Machine) | `nvidia/Nemotron-3-Ultra-550b-a55b` | Needs detailed knowledge of when APIs were added, changed and removed |
| Diagnose the failure and write the patch | `nvidia/Nemotron-3-Ultra-550b-a55b` | The hard reasoning step: a library change and its effect on results |
| Write the web search query | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | A fast, cheap call that runs on every attempt |
| Optional cheaper patching | `nvidia/nemotron-3-super-120b-a12b` | Set `REVIVE_MODEL_DEEP` to it for development; roughly 3x cheaper than Ultra |

- All calls go through Token Factory's OpenAI-compatible API (`https://api.tokenfactory.nebius.com/v1/`).
- **Model IDs are never hard-coded.** Revive lists the available models at startup (`GET /v1/models`) and picks one per role by name, so a renamed or newer Nemotron version does not break it. Any role can be pinned with `REVIVE_MODEL_DEEP`, `REVIVE_MODEL_BALANCED` or `REVIVE_MODEL_FAST`.
- Reasoning output (`<think>...</think>`) is stripped before parsing, temporary server errors (HTTP 429/5xx) are retried without costing an attempt, and token usage is tracked per model, so the benchmark reports exactly what each repair cost.
- Token Factory made it practical to use a 550B-parameter model for the hard steps and a 30B model for the frequent one from the same API, with nothing to host.

## Benchmark

`benchmark/` holds 20 small ML notebooks, each broken against current NumPy 2.5, pandas 3.1 and scikit-learn 1.9. `scripts/make_benchmark.py` builds them reproducibly, and its saved outputs are **authentic**:

1. The old code is executed in a real old environment (Python 3.10 with NumPy 1.22, pandas 1.5 and scikit-learn 1.1, or Python 3.8 with pandas 0.25 for `.ix`). Those outputs, including the deprecation warnings it printed, are saved.
2. A correct modern version must reproduce them today, which proves each notebook can be repaired.
3. The old code must be broken today: it fails, or it runs and prints different numbers.

Three notebooks are also included **saved without outputs**, as people often share them. Only the Time Machine can verify a repair of those. Each comes with the date version control would give it (`as_of` in `benchmark/manifest.json`: 2023-02-01). That date is deliberately not a hint at the exact versions: on it, the newest NumPy had already removed `np.float`, so the Time Machine still has to work the versions out. Because we know their real outputs, the benchmark also reports **faithfulness**: whether the rebuilt original reproduces the author's hidden outputs.

Traps where only output verification can catch the problem:

- `sklearn_kmeans_silent` runs without any error but prints a different inertia today.
- `sklearn_ovr_trap` breaks on a removed argument, and the obvious fix of deleting it runs but changes the accuracy and the predicted probabilities.
- `sklearn_kmeans_silent_no_outputs` is the same silent change with no saved outputs: it "works" unless you rebuild the original environment.

### Results (final run)

| | |
|---|---|
| Verified repairs | **20/20** |
| Notebooks saved without outputs, verified through the Time Machine | **3/3** |
| Time Machine faithfulness: rebuilt original reproduced the author's real (hidden) outputs | **3/3** |
| Fixes that ran but printed different numbers, caught by output verification and then corrected | **3** |
| Average attempts per notebook | 1.5 (max 3) |
| Median time per notebook | 44 s (whole benchmark: about 17 minutes) |
| Total tokens, all 20 notebooks | 162,423 |

| Notebook | What broke | Checked against | Outcome | Attempts | Cells matched | Time (s) |
|---|---|---|---|---|---|---|
| numpy_type_aliases | np.float / np.int / np.bool (removed in NumPy 1.24) | saved outputs | verified | 1 | 3/3 | 26.1 |
| numpy2_constants | np.NaN, np.Inf, np.round_ (removed in NumPy 2.0) | saved outputs | verified | 1 | 2/2 | 33.7 |
| numpy2_functions | np.product, np.cumproduct, np.alltrue, np.in1d, np.trapz, np.row_stack | saved outputs | verified | 2 | 3/3 | 76.6 |
| numpy_asscalar | np.asscalar (removed in NumPy 1.23) | saved outputs | verified | 1 | 1/1 | 21.2 |
| pandas_append | DataFrame.append and Series.append (removed in pandas 2.0) | saved outputs | verified | 1 | 2/2 | 32.1 |
| pandas_iteritems | DataFrame.iteritems (removed in pandas 2.0) | saved outputs | verified | 1 | 1/1 | 23.0 |
| pandas_ix_lookup | .ix (removed in pandas 1.0), DataFrame.lookup (removed in 2.0) | saved outputs | verified | 1 | 2/2 | 60.8 |
| pandas_mad_applymap | DataFrame.mad (removed in pandas 2.0), DataFrame.applymap (removed in 3.0) | saved outputs | verified | 2 | 2/2 | 44.3 |
| pandas_io_fill | read_csv(delim_whitespace=...) and fillna(method=...) (removed in pandas 3.0) | saved outputs | verified | 1 | 2/2 | 25.1 |
| pandas_groupby_numeric | groupby().mean() no longer drops non-numeric columns silently | saved outputs | verified | 1 | 1/1 | 31.0 |
| pandas_dates | Series.dt.weekofyear and Series.first (removed in pandas 2.x/3.0) | saved outputs | verified | 2 | 2/2 | 50.1 |
| pandas_dates_no_outputs | same, saved without outputs | rebuilt original | verified | 2 | 2/2 | 87.2 |
| sklearn_encoders | OneHotEncoder(sparse=...) (removed in 1.4), get_feature_names (removed in 1.2) | saved outputs | verified | 2 | 2/2 | 47.5 |
| sklearn_rmse | mean_squared_error(squared=False) (removed in 1.6) | saved outputs | verified | 1 | 1/1 | 28.7 |
| sklearn_renamed_params | base_estimator, algorithm='full', affinity (renamed/removed) | saved outputs | verified | 2 | 3/3 | 40.4 |
| sklearn_ovr_trap | LogisticRegression(multi_class='ovr') removed; deleting the argument silently switches to multinomial | saved outputs | verified | 1 | 1/1 | 30.1 |
| sklearn_kmeans_silent | KMeans default n_init changed from 10 to 'auto' in 1.4: no error, different clusters | saved outputs | verified | 3 | 1/1 | 90.3 |
| sklearn_kmeans_silent_no_outputs | same, saved without outputs | rebuilt original | verified | 2 | 1/1 | 137.1 |
| legacy_churn_model | DataFrame.append, np.float/np.int, OneHotEncoder(sparse=), iteritems | saved outputs | verified | 2 | 6/6 | 43.7 |
| legacy_churn_model_no_outputs | same, saved without outputs | rebuilt original | verified | 1 | 6/6 | 71.9 |

The three cases where output verification mattered, from `benchmark/results/results.json`:

- **sklearn_kmeans_silent** never raised an error. Nemotron's first fix ran but printed a different inertia; the second fix used a removed option; the third restored the old initialisation (`init='random', n_init=10`) and matched.
- **sklearn_kmeans_silent_no_outputs**: the Time Machine rebuilt Python 3.10 with scikit-learn 1.2.1 (it corrected two version guesses Nemotron made that were released after the notebook's date), ran the original, and only then could the drift in today's version be seen and fixed with `n_init=10`.
- **sklearn_encoders**: the first fix ran but printed a NumPy array where the original printed a list; the second added `.tolist()`.

![A fix that ran but printed a different inertia, caught and corrected](docs/drift-caught.png)

Honest history: on the current benchmark, two earlier full runs verified 17/20 and 19/20 before the Time Machine fixes described above (dated rebuilds; correcting versions released after the date). Before that, a first version of the benchmark, whose expected outputs came from modern rewrites of the old code, verified 17/17, 15/17 and 16/17; running the original code in rebuilt old environments then showed that one of those expected outputs (`sklearn_ovr_trap`) did not exactly match what old scikit-learn printed. That is why the benchmark now uses authentic outputs. One model call in the final run kept returning a temporary 503 after its retries and cost one attempt; the next attempt succeeded.

Re-run it yourself: `python scripts/run_benchmark.py` (writes `benchmark/results/RESULTS.md`).

## Quick start

Requirements: Python 3.10+, a [Token Factory API key](https://tokenfactory.nebius.com), optionally a [Tavily API key](https://app.tavily.com). The Time Machine downloads old Python builds and packages, so it needs internet access.

```bash
git clone https://github.com/<you>/revive.git && cd revive
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
export NEBIUS_API_KEY=...            # Windows PowerShell: $env:NEBIUS_API_KEY="..."
export TAVILY_API_KEY=...            # optional
python scripts/smoke_test.py         # checks keys, lists Nemotron models, calls each role
```

**Command line**

```bash
python -m revive examples/legacy_churn_model.ipynb
python -m revive benchmark/notebooks/sklearn_kmeans_silent_no_outputs.ipynb    # uses the Time Machine
```

Revive writes `<name>.revived.ipynb` and a JSON report with every attempt, the sources it found, the per-cell verification and, if used, the rebuilt environment. Flags: `--time-machine auto|always|off` (default `auto`: only when there are no saved outputs), `--written-before YYYY-MM-DD` (default: the notebook's last git commit date, if any), `--max-attempts N`, `--timeout SECONDS` (per cell), `--no-search`, `--no-env`, `--rtol`.

**Web app**

```bash
uvicorn revive.web.app:app --port 8000
```

Open http://localhost:8000, pick a notebook and press **Repair notebook**. The notebook view marks each cell as it runs, fails, gets edited and is checked; the repair log shows every rebuild, run, search and patch. Tick **Time Machine** to also rebuild the original environment of a notebook that has saved outputs. Runs recorded by the benchmark replay instantly at no cost; **Run live** calls Nemotron now.

| Setting | Default | Meaning |
|---|---|---|
| `REVIVE_ALLOW_UPLOADS` | `0` | Let visitors repair their own notebooks. Uploaded notebooks run code on the server, so keep this off on public hosts. |
| `REVIVE_LIVE` | `1` | `0` allows recorded replays only. |
| `REVIVE_LIVE_PER_CLIENT` / `REVIVE_LIVE_PER_DAY` | `3` / `20` | Caps on live repairs, to protect API credits. |
| `REVIVE_MAX_CONCURRENT` | `2` | Repairs running at the same time. |

**Tests**

```bash
pytest       # about 250 tests, no API keys needed
```

The tests start real Jupyter kernels, create real isolated environments and (when uv is available) build real old Python environments; the model is replaced by scripted replies. They run on every push through GitHub Actions.

## Deploy

- **Hugging Face Spaces (free):** `pip install huggingface_hub`, set `HF_TOKEN`, `NEBIUS_API_KEY` and `TAVILY_API_KEY`, then run `python scripts/deploy_huggingface.py --github <repo url>`. Keys are stored as encrypted Space secrets.
- **Any container host, including Nebius Serverless Endpoints:** `docker build -t revive .` then run it with `NEBIUS_API_KEY` and `TAVILY_API_KEY` set; it listens on `$PORT` (default 7860). The image ships with Python 3.8 and 3.10 for the Time Machine.

## Project layout

| Path | What it is |
|---|---|
| `revive/runner.py` | Executes a notebook in a fresh kernel and reports the exact failing cell, timeouts and crashed kernels |
| `revive/compare.py` | Per-cell output verification with numeric tolerance and normalisation |
| `revive/agent.py` | The repair loop |
| `revive/timemachine.py` | Infers, builds (with uv) and runs the notebook's original environment |
| `revive/patches.py` | The patch format and its checks: strict allowlist for pip requirements, exact unique snippets, syntax check |
| `revive/env.py` | Isolated per-notebook environments layered on the host environment |
| `revive/llm.py`, `revive/search.py` | Token Factory and Tavily clients |
| `revive/web/` | FastAPI app with live event streaming, replay cache and usage limits |
| `benchmark/`, `scripts/` | Benchmark notebooks, generator, runner, deploy script |

## Limitations

- **The Time Machine's reference is only as good as the rebuilt environment.** Revive validates the guess by running the untouched original there (and, when saved outputs exist, by comparing with them), but a notebook that runs under a slightly different version than the author's could print slightly different numbers. Without a date, silent behaviour changes (like a changed default) cannot be dated from the code at all. The report always says which reference a verification used.
- **Old environments need wheels.** The Time Machine supports Python 3.8 to 3.12; notebooks that need libraries without wheels for those Pythons cannot be rebuilt.
- **Randomness without a seed** looks like drift, because the numbers really do change between runs. Revive reports it rather than hiding it.
- **Plots are not compared.** Only text output is checked; images vary byte-for-byte between library versions.
- **Isolation is per package, not per machine.** Executing a notebook runs its code. The hosted demo only runs the bundled notebooks; run untrusted notebooks inside a container.
- **The benchmark is synthetic**, written for this project. Its outputs are authentic and its breaks are real, documented API changes, but real-world notebooks are messier.

## Prior art

Revive builds on known ideas; these projects came first and are worth knowing:

- [nbval](https://pypi.org/project/nbval/) validates notebooks by comparing re-run outputs with saved ones, but does not repair anything.
- [SnifferDog](https://arxiv.org/abs/2103.02959) infers the library versions a notebook needs from the APIs it uses and writes a requirements file. It restores environments; it does not modernise code or compare outputs.
- [Containing the Reproducibility Gap](https://arxiv.org/abs/2604.01072) (2026) containerises notebook repositories and compares re-executed cells with saved outputs, in a current environment and without repairing code.
- [repro-check](https://pypi.org/project/repro-check/) and [ReproRun](https://github.com/WUBING2023/ReproRun) make old research code run again.
- [MLEModernizer](https://arxiv.org/abs/2602.07195) repairs Kaggle notebooks with an LLM and checks the final competition score with a grader.

What Revive adds: a repair loop that must pass **per-cell output verification**, and the **Time Machine**, which uses the untouched original running in its rebuilt environment as the reference for verifying the modernised version, so even notebooks saved without outputs can be verified. In our search of the projects above we did not find this combination; if you know of one, please open an issue.

## Credits

Built with NVIDIA Nemotron, Nebius Token Factory, Tavily, uv, nbclient and FastAPI. Developed with help from Claude (Anthropic).

## License

[MIT](LICENSE)
