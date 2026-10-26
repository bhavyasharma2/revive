# Revive benchmark results

- Notebooks: **20**, all broken against current NumPy/pandas/scikit-learn. Saved outputs are authentic: produced by the old code on old libraries.
- Verified repairs: **20/20**
- Saved without outputs, verified against the Time Machine's rebuilt original environment: **3/3**
- Time Machine faithfulness: the rebuilt original reproduced the author's real (hidden) outputs in **3/3**
- Ran at some point during repair: 20/20
- Times a fix ran but the output check caught different numbers: 3
- Total tokens: 137,238
- Attempts allowed per notebook: up to 12, stopping after 4 in a row without progress

| Notebook | What broke | Checked against | Outcome | Attempts | Cells matched | Time (s) |
|---|---|---|---|---|---|---|
| numpy_type_aliases | np.float / np.int / np.bool (removed in NumPy 1.24) | saved outputs | verified | 1 | 3/3 | 23.0 |
| numpy2_constants | np.NaN, np.Inf, np.round_ (removed in NumPy 2.0) | saved outputs | verified | 2 | 2/2 | 35.7 |
| numpy2_functions | np.product, np.cumproduct, np.alltrue, np.in1d, np.trapz, np.row_stack | saved outputs | verified | 2 | 3/3 | 48.2 |
| numpy_asscalar | np.asscalar (removed in NumPy 1.23) | saved outputs | verified | 1 | 1/1 | 19.8 |
| pandas_append | DataFrame.append and Series.append (removed in pandas 2.0) | saved outputs | verified | 1 | 2/2 | 24.2 |
| pandas_iteritems | DataFrame.iteritems (removed in pandas 2.0) | saved outputs | verified | 1 | 1/1 | 19.0 |
| pandas_ix_lookup | .ix (removed in pandas 1.0), DataFrame.lookup (removed in 2.0) | saved outputs | verified | 1 | 2/2 | 22.0 |
| pandas_mad_applymap | DataFrame.mad (removed in pandas 2.0), DataFrame.applymap (removed in 3.0) | saved outputs | verified | 1 | 2/2 | 23.0 |
| pandas_io_fill | read_csv(delim_whitespace=...) and fillna(method=...) (removed in pandas 3.0) | saved outputs | verified | 1 | 2/2 | 19.7 |
| pandas_groupby_numeric | groupby().mean() no longer drops non-numeric columns silently | saved outputs | verified | 1 | 1/1 | 25.3 |
| pandas_dates | Series.dt.weekofyear and Series.first (removed in pandas 2.x/3.0) | saved outputs | verified | 2 | 2/2 | 42.7 |
| pandas_dates_no_outputs | Series.dt.weekofyear and Series.first (removed in pandas 2.x/3.0) | rebuilt original | verified | 1 | 2/2 | 41.8 |
| sklearn_encoders | OneHotEncoder(sparse=...) (removed in 1.4), get_feature_names (removed in 1.2) | saved outputs | verified | 2 | 2/2 | 44.3 |
| sklearn_rmse | mean_squared_error(squared=False) (removed in 1.6) | saved outputs | verified | 1 | 1/1 | 30.1 |
| sklearn_renamed_params | base_estimator, algorithm='full', affinity (renamed/removed) | saved outputs | verified | 2 | 3/3 | 49.0 |
| sklearn_ovr_trap | LogisticRegression(multi_class='ovr') removed; deleting the argument silently switches to multinomial | saved outputs | verified | 1 | 1/1 | 28.5 |
| sklearn_kmeans_silent | KMeans default n_init changed from 10 to 'auto' in 1.4: no error, different clusters | saved outputs | verified | 1 | 1/1 | 26.8 |
| sklearn_kmeans_silent_no_outputs | KMeans default n_init changed from 10 to 'auto' in 1.4: no error, different clusters | rebuilt original | verified | 1 | 1/1 | 51.3 |
| legacy_churn_model | DataFrame.append, np.float/np.int, OneHotEncoder(sparse=), iteritems | saved outputs | verified | 2 | 6/6 | 43.7 |
| legacy_churn_model_no_outputs | DataFrame.append, np.float/np.int, OneHotEncoder(sparse=), iteritems | rebuilt original | verified | 1 | 6/6 | 71.1 |
