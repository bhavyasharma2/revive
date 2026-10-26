# Real notebooks

Public notebooks written by other people, used to test Revive outside its own benchmark. They were picked because they are self-contained (they generate their data or use scikit-learn's bundled datasets), run on a CPU, and call APIs that NumPy, pandas, scikit-learn or SciPy have since removed or changed. Their saved outputs are the authors' own.

| Notebook | Source | Author | Licence |
|---|---|---|---|
| `hom_04_training_linear_models` | [handson-ml, chapter 4](https://github.com/ageron/handson-ml/blob/master/04_training_linear_models.ipynb) | Aurélien Géron | Apache 2.0, unchanged |
| `hom_06_decision_trees` | [handson-ml, chapter 6](https://github.com/ageron/handson-ml/blob/master/06_decision_trees.ipynb) | Aurélien Géron | Apache 2.0, unchanged |
| `pdsh_03_06_concat_and_append` | [Python Data Science Handbook 3.06](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.06-Concat-And-Append.ipynb) | Jake VanderPlas | MIT (code) |
| `pdsh_03_04_missing_values` | [Python Data Science Handbook 3.04](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.04-Missing-Values.ipynb) | Jake VanderPlas | MIT (code) |

The Python Data Science Handbook's code is MIT-licensed but its text is CC BY-NC-ND, which does not allow modified copies, so only the code cells (with their saved outputs) are included, under a short note that credits the source. The licences are in `licenses/`.

Run them: `python scripts/run_benchmark.py --set real` (writes `benchmark/real/results/` and records every run, including failures, for the web app to replay).
