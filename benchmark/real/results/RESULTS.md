# Revive on real public notebooks

- Notebooks: **4**, written by other people (see benchmark/real/README.md). Saved outputs are the authors' own.
- Verified repairs: **2/4**
- Ran at some point during repair: 3/4
- Times a fix ran but the output check caught different numbers: 0
- Total tokens: 135,262
- Attempts allowed per notebook: up to 12, stopping after 4 in a row without progress

| Notebook | What broke | Checked against | Outcome | Attempts | Cells matched | Time (s) |
|---|---|---|---|---|---|---|
| hom_04_training_linear_models | np.infty (removed in NumPy 2.0), astype(np.int) (removed in NumPy 1.24), LogisticRegression(multi_class=...) (removed in scikit-learn 1.8) | saved outputs | failed | 4 |  | 268.3 |
| hom_06_decision_trees | scipy.stats.mode changed its output shape in SciPy 1.11 (keepdims) | saved outputs | already_working | 0 | 17/17 | 29.3 |
| pdsh_03_06_concat_and_append | DataFrame.append (removed in pandas 2.0), called through a string the notebook evaluates | saved outputs | verified | 1 | 14/14 | 22.1 |
| pdsh_03_04_missing_values | fillna(method=...) (removed in pandas 3.0) | saved outputs | verified | 1 | 26/26 | 35.8 |
