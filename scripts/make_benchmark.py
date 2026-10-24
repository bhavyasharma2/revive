"""Build the Revive benchmark: small ML notebooks written against old library APIs.

Each notebook is written twice: with the old API it was "written" with, and
with a correct modern equivalent. Then:
1. The old version is executed in a real old environment (an old Python and
   2019-2022 library versions, built with uv). Those outputs, including any
   warnings it printed, are what gets saved: authentic, not simulated.
2. The modern version is executed today and must reproduce those outputs, which
   proves the notebook can be repaired.
3. The old version must really be broken today: it fails ("error"), or it runs
   and prints different numbers ("silent").
A few notebooks are also saved with their outputs cleared ("no_outputs"), as
people often share them; only the Time Machine can verify a repair of those.

Run:  python scripts/make_benchmark.py      (needs uv and network for the old environments)
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook  # noqa: E402

from revive.compare import compare_notebooks  # noqa: E402
from revive.notebook import save_notebook, strip_outputs  # noqa: E402
from revive.runner import execute_notebook  # noqa: E402
from revive.timemachine import EnvSpec, LegacyEnv  # noqa: E402

OUT_DIR = ROOT / "benchmark" / "notebooks"

# The environments the notebooks were "written" in.
ENV_2022 = EnvSpec("3.10", {"numpy": "1.22.4", "pandas": "1.5.3", "scikit-learn": "1.1.3"})
ENV_2019 = EnvSpec("3.8", {"numpy": "1.18.5", "pandas": "0.25.3"})   # .ix needs pandas < 1.0

# Saved again with outputs cleared, to test verification without saved outputs.
NO_OUTPUT_VARIANTS = ("legacy_churn_model", "sklearn_kmeans_silent", "pandas_dates")

# The date version control would give for those notebooks (their "last commit"). It is
# consistent with ENV_2022 being the newest environment at the time, but deliberately not
# identical to it: on that date the newest NumPy (1.24) had already removed np.float, so
# the Time Machine still has to work out the right versions from the evidence.
NO_OUTPUT_AS_OF = "2023-02-01"

IMPORTS = "import numpy as np\nimport pandas as pd"

# Each cell is ("md", text), a plain source string (same in both versions),
# or (modern_source, legacy_source).
SPECS: list[dict] = [
    {
        "name": "numpy_type_aliases",
        "title": "Feature scaling with NumPy type aliases",
        "breaks": "np.float / np.int / np.bool (removed in NumPy 1.24)",
        "kind": "error",
        "cells": [
            IMPORTS,
            "rng = np.random.RandomState(1)\nages = rng.randint(18, 80, 50)\nincome = rng.normal(52000, 9000, 50)",
            ("scaled = (income - income.mean()).astype(np.float64) / income.std()\nprint(round(float(scaled[:5].sum()), 5))",
             "scaled = (income - income.mean()).astype(np.float) / income.std()\nprint(round(float(scaled[:5].sum()), 5))"),
            ("buckets = (ages // 10).astype(int)\nprint(np.bincount(buckets).tolist())",
             "buckets = (ages // 10).astype(np.int)\nprint(np.bincount(buckets).tolist())"),
            ("is_senior = (ages > 60).astype(bool)\nprint(int(is_senior.sum()))",
             "is_senior = (ages > 60).astype(np.bool)\nprint(int(is_senior.sum()))"),
        ],
    },
    {
        "name": "numpy2_constants",
        "title": "Cleaning sensor readings with NumPy constants",
        "breaks": "np.NaN, np.Inf, np.round_ (removed in NumPy 2.0)",
        "kind": "error",
        "cells": [
            IMPORTS,
            ("readings = np.array([3.2, np.nan, 4.8, np.inf, 5.1, -np.inf, 2.9])\nfinite = readings[np.isfinite(readings)]\nprint(finite.tolist())",
             "readings = np.array([3.2, np.NaN, 4.8, np.Inf, 5.1, -np.Inf, 2.9])\nfinite = readings[np.isfinite(readings)]\nprint(finite.tolist())"),
            ("print(np.round(finite.mean(), 3))",
             "print(np.round_(finite.mean(), 3))"),
        ],
    },
    {
        "name": "numpy2_functions",
        "title": "Array utilities renamed in NumPy 2",
        "breaks": "np.product, np.cumproduct, np.alltrue, np.in1d, np.trapz, np.row_stack",
        "kind": "error",
        "cells": [
            IMPORTS,
            ("growth = np.array([1.02, 1.05, 0.98, 1.10])\nprint(round(float(np.prod(growth)), 6))\nprint(np.round(np.cumprod(growth), 4).tolist())",
             "growth = np.array([1.02, 1.05, 0.98, 1.10])\nprint(round(float(np.product(growth)), 6))\nprint(np.round(np.cumproduct(growth), 4).tolist())"),
            ("print(bool(np.all(growth > 0.9)))\nprint(np.isin([1, 5, 9], [5, 9, 12]).tolist())",
             "print(bool(np.alltrue(growth > 0.9)))\nprint(np.in1d([1, 5, 9], [5, 9, 12]).tolist())"),
            ("x = np.linspace(0, 1, 11)\nprint(round(float(np.trapezoid(x ** 2, x)), 6))\nprint(np.vstack([x[:2], x[-2:]]).shape)",
             "x = np.linspace(0, 1, 11)\nprint(round(float(np.trapz(x ** 2, x)), 6))\nprint(np.row_stack([x[:2], x[-2:]]).shape)"),
        ],
    },
    {
        "name": "numpy_asscalar",
        "title": "Extracting Python scalars from arrays",
        "breaks": "np.asscalar (removed in NumPy 1.23)",
        "kind": "error",
        "cells": [
            IMPORTS,
            ("best = np.array([0.8731])\nprint(best.item() * 100)",
             "best = np.array([0.8731])\nprint(np.asscalar(best) * 100)"),
        ],
    },
    {
        "name": "pandas_append",
        "title": "Collecting experiment results row by row",
        "breaks": "DataFrame.append and Series.append (removed in pandas 2.0)",
        "kind": "error",
        "cells": [
            IMPORTS,
            ("results = pd.DataFrame(columns=['run', 'lr', 'loss'])\nfor i, lr in enumerate([0.1, 0.01, 0.001]):\n    row = pd.DataFrame([{'run': i, 'lr': lr, 'loss': round(1 / (1 + 50 * lr), 4)}])\n    results = pd.concat([results, row], ignore_index=True)\nprint(results.to_string())",
             "results = pd.DataFrame(columns=['run', 'lr', 'loss'])\nfor i, lr in enumerate([0.1, 0.01, 0.001]):\n    row = {'run': i, 'lr': lr, 'loss': round(1 / (1 + 50 * lr), 4)}\n    results = results.append(row, ignore_index=True)\nprint(results.to_string())"),
            ("tags = pd.Series(['baseline', 'tuned'])\ntags = pd.concat([tags, pd.Series(['final'])], ignore_index=True)\nprint(tags.tolist())",
             "tags = pd.Series(['baseline', 'tuned'])\ntags = tags.append(pd.Series(['final']), ignore_index=True)\nprint(tags.tolist())"),
        ],
    },
    {
        "name": "pandas_iteritems",
        "title": "Column summaries with iteritems",
        "breaks": "DataFrame.iteritems (removed in pandas 2.0)",
        "kind": "error",
        "cells": [
            IMPORTS,
            "df = pd.DataFrame({'height': [170, 182, 165, 190], 'weight': [65, 80, 58, 92]})",
            ("for name, col in df.items():\n    print(name, col.min(), col.max(), round(col.mean(), 2))",
             "for name, col in df.iteritems():\n    print(name, col.min(), col.max(), round(col.mean(), 2))"),
        ],
    },
    {
        "name": "pandas_ix_lookup",
        "title": "Label lookups with .ix and DataFrame.lookup",
        "breaks": ".ix (removed in pandas 1.0), DataFrame.lookup (removed in 2.0)",
        "kind": "error",
        "cells": [
            IMPORTS,
            "scores = pd.DataFrame({'math': [91, 78, 85], 'physics': [88, 95, 70], 'chem': [72, 81, 90]}, index=['ana', 'ben', 'chloe'])",
            ("print(scores.loc['ben', 'physics'])\nprint(scores.iloc[0].tolist())",
             "print(scores.ix['ben', 'physics'])\nprint(scores.ix[0].tolist())"),
            ("favourite = ['physics', 'math', 'chem']\nidx = scores.columns.get_indexer(favourite)\nprint(scores.to_numpy()[np.arange(len(scores)), idx].tolist())",
             "favourite = ['physics', 'math', 'chem']\nprint(scores.lookup(scores.index, favourite).tolist())"),
        ],
    },
    {
        "name": "pandas_mad_applymap",
        "title": "Robust statistics with mad and applymap",
        "breaks": "DataFrame.mad (removed in pandas 2.0), DataFrame.applymap (removed in 3.0)",
        "kind": "error",
        "cells": [
            IMPORTS,
            "df = pd.DataFrame({'a': [1.0, 2.5, 3.0, 10.0], 'b': [4.0, 4.5, 5.5, 6.0]})",
            ("print((df - df.mean()).abs().mean().round(4).to_dict())",
             "print(df.mad().round(4).to_dict())"),
            ("print(df.map(lambda v: f'{v:.1f}').iloc[0].tolist())",
             "print(df.applymap(lambda v: f'{v:.1f}').iloc[0].tolist())"),
        ],
    },
    {
        "name": "pandas_io_fill",
        "title": "Reading whitespace data and forward-filling gaps",
        "breaks": "read_csv(delim_whitespace=...) and fillna(method=...) (removed in pandas 3.0)",
        "kind": "error",
        "cells": [
            IMPORTS + "\nimport io",
            ("raw = 'day temp\\n1 21.5\\n2 NaN\\n3 23.0\\n4 NaN\\n5 24.5'\ndf = pd.read_csv(io.StringIO(raw), sep=r'\\s+')\nprint(df.shape)",
             "raw = 'day temp\\n1 21.5\\n2 NaN\\n3 23.0\\n4 NaN\\n5 24.5'\ndf = pd.read_csv(io.StringIO(raw), delim_whitespace=True)\nprint(df.shape)"),
            ("print(df['temp'].ffill().tolist())",
             "print(df['temp'].fillna(method='ffill').tolist())"),
        ],
    },
    {
        "name": "pandas_groupby_numeric",
        "title": "Group means on a mixed-type table",
        "breaks": "groupby().mean() no longer drops non-numeric columns silently",
        "kind": "error",
        "cells": [
            IMPORTS,
            "sales = pd.DataFrame({'region': ['n', 'n', 's', 's', 's'], 'rep': ['ali', 'bo', 'cy', 'di', 'ed'], 'units': [10, 14, 7, 9, 11], 'price': [2.5, 2.0, 3.0, 3.5, 2.5]})",
            ("print(sales.groupby('region').mean(numeric_only=True).round(3).to_string())",
             "print(sales.groupby('region').mean().round(3).to_string())"),
        ],
    },
    {
        "name": "pandas_dates",
        "title": "Weekly features from timestamps",
        "breaks": "Series.dt.weekofyear and Series.first (removed in pandas 2.x/3.0)",
        "kind": "error",
        "cells": [
            IMPORTS,
            "dates = pd.Series(pd.date_range('2021-03-01', periods=6, freq='5D'))",
            ("print(dates.dt.isocalendar().week.astype('int64').tolist())",
             "print(dates.dt.weekofyear.tolist())"),
            ("ts = pd.Series(range(6), index=dates.values)\nprint(ts[ts.index < ts.index[0] + pd.Timedelta('11D')].tolist())",
             "ts = pd.Series(range(6), index=dates.values)\nprint(ts.first('11D').tolist())"),
        ],
    },
    {
        "name": "sklearn_encoders",
        "title": "Encoding categorical and text features",
        "breaks": "OneHotEncoder(sparse=...) (removed in 1.4), get_feature_names (removed in 1.2)",
        "kind": "error",
        "cells": [
            "from sklearn.preprocessing import OneHotEncoder\nfrom sklearn.feature_extraction.text import CountVectorizer",
            ("enc = OneHotEncoder(sparse_output=False)\nprint(enc.fit_transform([['red'], ['blue'], ['red']]).tolist())",
             "enc = OneHotEncoder(sparse=False)\nprint(enc.fit_transform([['red'], ['blue'], ['red']]).tolist())"),
            ("vec = CountVectorizer().fit(['the cat sat', 'the dog ran'])\nprint([str(f) for f in vec.get_feature_names_out()])",
             "vec = CountVectorizer().fit(['the cat sat', 'the dog ran'])\nprint(vec.get_feature_names())"),
        ],
    },
    {
        "name": "sklearn_rmse",
        "title": "Regression error report",
        "breaks": "mean_squared_error(squared=False) (removed in 1.6)",
        "kind": "error",
        "cells": [
            "import numpy as np\nfrom sklearn.linear_model import Ridge\nfrom sklearn.datasets import make_regression",
            "X, y = make_regression(n_samples=120, n_features=4, noise=12.0, random_state=3)\nmodel = Ridge(alpha=1.0).fit(X[:90], y[:90])\npred = model.predict(X[90:])",
            ("from sklearn.metrics import root_mean_squared_error\nprint(round(root_mean_squared_error(y[90:], pred), 4))",
             "from sklearn.metrics import mean_squared_error\nprint(round(mean_squared_error(y[90:], pred, squared=False), 4))"),
        ],
    },
    {
        "name": "sklearn_renamed_params",
        "title": "Ensembles and clustering with renamed parameters",
        "breaks": "base_estimator, algorithm='full', affinity (renamed/removed)",
        "kind": "error",
        "cells": [
            "from sklearn.datasets import make_classification, make_blobs\nfrom sklearn.ensemble import BaggingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.cluster import KMeans, AgglomerativeClustering",
            ("X, y = make_classification(n_samples=200, n_features=6, random_state=7)\nbag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=10, random_state=7).fit(X, y)\nprint(round(bag.score(X, y), 4))",
             "X, y = make_classification(n_samples=200, n_features=6, random_state=7)\nbag = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=10, random_state=7).fit(X, y)\nprint(round(bag.score(X, y), 4))"),
            ("P, _ = make_blobs(n_samples=90, centers=3, random_state=2)\nkm = KMeans(n_clusters=3, algorithm='lloyd', n_init=10, random_state=0).fit(P)\nprint(round(km.inertia_, 3))",
             "P, _ = make_blobs(n_samples=90, centers=3, random_state=2)\nkm = KMeans(n_clusters=3, algorithm='full', n_init=10, random_state=0).fit(P)\nprint(round(km.inertia_, 3))"),
            ("agg = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='average').fit(P)\nprint(sorted(__import__('collections').Counter(agg.labels_).values()))",
             "agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average').fit(P)\nprint(sorted(__import__('collections').Counter(agg.labels_).values()))"),
        ],
    },
    {
        "name": "sklearn_ovr_trap",
        "title": "One-vs-rest classifier (dropping the old flag changes results)",
        "breaks": "LogisticRegression(multi_class='ovr') removed; deleting the argument silently switches to multinomial",
        "kind": "error",
        "cells": [
            "from sklearn.datasets import make_classification\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.multiclass import OneVsRestClassifier",
            "X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_classes=4, n_clusters_per_class=1, random_state=11)",
            ("clf = OneVsRestClassifier(LogisticRegression(max_iter=2000)).fit(X[:200], y[:200])\nprint(round(clf.score(X[200:], y[200:]), 4))\nprint(clf.predict_proba(X[200:203]).round(3).tolist())",
             "clf = LogisticRegression(multi_class='ovr', max_iter=2000).fit(X[:200], y[:200])\nprint(round(clf.score(X[200:], y[200:]), 4))\nprint(clf.predict_proba(X[200:203]).round(3).tolist())"),
        ],
    },
    {
        "name": "sklearn_kmeans_silent",
        "title": "Customer segments with KMeans (runs fine, wrong numbers)",
        "breaks": "KMeans default n_init changed from 10 to 'auto' in 1.4: no error, different clusters",
        "kind": "silent",
        "cells": [
            "from sklearn.cluster import KMeans\nfrom sklearn.datasets import make_blobs",
            "X, _ = make_blobs(n_samples=300, centers=6, cluster_std=2.2, random_state=4)",
            ("km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(X)\nprint(round(km.inertia_, 2))",
             "km = KMeans(n_clusters=6, random_state=0).fit(X)\nprint(round(km.inertia_, 2))"),
        ],
    },
    {
        "name": "legacy_churn_model",
        "title": "Customer churn model (written in 2021)",
        "intro": "A small end-to-end example: build a dataset, encode categories, train a model, report accuracy.",
        "breaks": "DataFrame.append, np.float/np.int, OneHotEncoder(sparse=), iteritems",
        "kind": "error",
        "cells": [
            "import numpy as np\nimport pandas as pd\nfrom sklearn.preprocessing import OneHotEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split",
            "rng = np.random.RandomState(0)\nn = 400\ndf = pd.DataFrame({\n    'tenure': rng.randint(1, 72, n),\n    'monthly': rng.uniform(20, 120, n).round(2),\n    'plan': rng.choice(['basic', 'plus', 'pro'], n),\n})\ndf['churn'] = ((df['monthly'] > 80) & (df['tenure'] < 24)).astype(int)\nprint(df.shape)\nprint(df['churn'].mean())",
            ("late_signups = pd.DataFrame({'tenure': [2, 3], 'monthly': [99.0, 110.0], 'plan': ['pro', 'pro'], 'churn': [1, 1]})\ndf = pd.concat([df, late_signups], ignore_index=True)\nprint(len(df))",
             "late_signups = pd.DataFrame({'tenure': [2, 3], 'monthly': [99.0, 110.0], 'plan': ['pro', 'pro'], 'churn': [1, 1]})\ndf = df.append(late_signups, ignore_index=True)\nprint(len(df))"),
            ("ratio = df['monthly'].to_numpy().astype(np.float64) / df['tenure'].to_numpy().astype(np.int64)\nprint(round(float(ratio.mean()), 4))",
             "ratio = df['monthly'].values.astype(np.float) / df['tenure'].values.astype(np.int)\nprint(round(float(ratio.mean()), 4))"),
            ("enc = OneHotEncoder(sparse_output=False)\nplan_onehot = enc.fit_transform(df[['plan']])\nprint(plan_onehot.shape)\nprint(list(enc.categories_[0]))",
             "enc = OneHotEncoder(sparse=False)\nplan_onehot = enc.fit_transform(df[['plan']])\nprint(plan_onehot.shape)\nprint(list(enc.categories_[0]))"),
            "X = np.hstack([df[['tenure', 'monthly']].to_numpy(), plan_onehot])\ny = df['churn'].to_numpy()\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\nmodel = LogisticRegression(max_iter=1000).fit(X_train, y_train)\nprint(f'accuracy: {model.score(X_test, y_test):.4f}')",
            ("for name, col in df[['tenure', 'monthly']].items():\n    print(name, round(col.mean(), 3))",
             "for name, col in df[['tenure', 'monthly']].iteritems():\n    print(name, round(col.mean(), 3))"),
        ],
    },
]

# Notebooks that need an older environment than ENV_2022.
LEGACY_ENV = {"pandas_ix_lookup": ENV_2019}


def _notebook(spec: dict, version: int):
    """version 0 = modern source, 1 = legacy source."""
    cells = [new_markdown_cell(f"# {spec['title']}" + (f"\n{spec['intro']}" if spec.get("intro") else ""))]
    for cell in spec["cells"]:
        source = cell[version] if isinstance(cell, tuple) else cell
        cells.append(new_code_cell(source))
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def build(spec: dict, env: LegacyEnv):
    # 1. Authentic outputs: the old code in its old environment.
    then = execute_notebook(_notebook(spec, 1), kernel_manager_factory=env.kernel_manager, timeout=180)
    if not then.ok:
        raise SystemExit(f"{spec['name']}: the old code fails in {env.spec.describe()}: {then.error_summary()}")
    saved = then.notebook
    saved.metadata["revive_benchmark"] = {k: spec[k] for k in ("name", "breaks", "kind")}
    nbformat.validate(saved)

    # 2. Solvable: a correct modern version reproduces those outputs today.
    modern = execute_notebook(_notebook(spec, 0), timeout=180)
    if not modern.ok:
        raise SystemExit(f"{spec['name']}: the modern solution fails today: {modern.error_summary()}")
    solvable = compare_notebooks(saved, modern.notebook)
    if not solvable.verified:
        raise SystemExit(f"{spec['name']}: the modern solution does not reproduce the authentic outputs: {solvable.summary()}\n"
                         + "\n".join(f"  cell {c.cell_index}: {c.detail}" for c in solvable.failures))

    # 3. Broken today.
    today = execute_notebook(saved, timeout=180)
    if spec["kind"] == "error":
        if today.ok:
            raise SystemExit(f"{spec['name']}: expected the old code to fail today, but it runs")
        observed = today.error_summary().splitlines()[0]
    else:
        if not today.ok:
            raise SystemExit(f"{spec['name']}: expected a silent change, but it fails: {today.error_summary()}")
        report = compare_notebooks(saved, today.notebook)
        if report.verified:
            raise SystemExit(f"{spec['name']}: expected drift, but the outputs still match")
        observed = report.summary()
    return saved, observed


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.ipynb"):
        old.unlink()
    manifest = []
    work = Path(tempfile.mkdtemp(prefix="revive-bench-envs-"))
    envs: dict[str, LegacyEnv] = {}
    try:
        for spec in SPECS:
            env_spec = LEGACY_ENV.get(spec["name"], ENV_2022)
            key = env_spec.describe()
            if key not in envs:
                print(f"building {key} ...", flush=True)
                envs[key] = LegacyEnv.create(work / f"env{len(envs)}", env_spec)
            saved, observed = build(spec, envs[key])
            save_notebook(saved, OUT_DIR / f"{spec['name']}.ipynb")
            entry = {"name": spec["name"], "title": spec["title"], "breaks": spec["breaks"], "kind": spec["kind"],
                     "written_for": key, "observed_today": observed}
            manifest.append(entry)
            print(f"{spec['name']:<26} {spec['kind']:<7} {observed}")

            if spec["name"] == "legacy_churn_model":
                save_notebook(saved, ROOT / "examples" / "legacy_churn_model.ipynb")
            if spec["name"] in NO_OUTPUT_VARIANTS:
                stripped = strip_outputs(saved)  # metadata (e.g. the Python version) stays, as when a user clears outputs
                name = f"{spec['name']}_no_outputs"
                save_notebook(stripped, OUT_DIR / f"{name}.ipynb")
                manifest.append({"name": name, "title": f"{spec['title']} (saved without outputs)", "breaks": spec["breaks"],
                                 "kind": "no_outputs", "base": spec["name"], "written_for": key, "as_of": NO_OUTPUT_AS_OF,
                                 "observed_today": "no saved outputs: " + observed})
    finally:
        shutil.rmtree(work, ignore_errors=True)

    (OUT_DIR.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(manifest)} notebooks written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
