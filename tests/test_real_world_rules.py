"""Rules learned from running Revive on real notebooks by other authors."""

from __future__ import annotations

import nbformat

from revive.compare import Status, compare_notebooks, compare_text
from revive.notebook import expected_errors
from revive.patches import gaming_reasons
from revive.runner import execute_notebook


def _nb(*cells):
    nb = nbformat.v4.new_notebook()
    nb.cells = list(cells)
    return nb


def _code(source, stdout=None, error=None, result=None):
    cell = nbformat.v4.new_code_cell(source)
    outputs = []
    if stdout is not None:
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stdout))
    if result is not None:
        outputs.append(nbformat.v4.new_output("execute_result", data={"text/plain": result}, execution_count=1))
    if error is not None:
        outputs.append(nbformat.v4.new_output("error", ename=error, evalue="", traceback=[]))
    cell.outputs = outputs
    return cell


# ---------- harmless formatting changes ----------

def test_timeit_lines_are_ignored():
    old = "2.87 ms ± 17.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)"
    new = "512 µs ± 3.2 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)"
    assert compare_text(old, new)[0] == Status.MATCH


def test_figure_size_lines_are_ignored_but_dimensions_elsewhere_are_not():
    assert compare_text("<Figure size 576x288 with 1 Axes>", "<Figure size 800x400 with 1 Axes>")[0] == Status.MATCH
    assert compare_text("image 800x400", "image 800x401")[0] != Status.MATCH


def test_memory_addresses_are_ignored_but_not_inside_dimensions():
    assert compare_text("<Line2D at 0x7f3a2c10>", "<Line2D at 0x10ab>")[0] == Status.MATCH
    assert compare_text("800x400", "800x500")[0] != Status.MATCH


def test_pandas_index_class_rename_is_cosmetic_but_values_are_not():
    old = "Indexes have overlapping values: Int64Index([0, 1], dtype='int64')"
    assert compare_text(old, "Indexes have overlapping values: Index([0, 1], dtype='int64')")[0] == Status.MATCH
    assert compare_text(old, "Indexes have overlapping values: Index([0, 2], dtype='int64')")[0] == Status.DRIFT


def test_pandas_string_dtype_is_cosmetic():
    assert compare_text("0    A\n1    B\ndtype: object", "0    A\n1    B\ndtype: str")[0] == Status.MATCH
    assert compare_text("0    A\n1    B\ndtype: object", "0    A\n1    C\ndtype: str")[0] == Status.CHANGED


FULL_TREE = """DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2, presort=False, random_state=42, splitter='best')"""


def test_sklearn_changed_only_printout_matches_full_printout():
    assert compare_text(FULL_TREE, "DecisionTreeClassifier(max_depth=2, random_state=42)")[0] == Status.MATCH
    nested_old = f"GridSearchCV(cv=3, estimator={FULL_TREE}, n_jobs=-1)"
    nested_new = "GridSearchCV(cv=3, estimator=DecisionTreeClassifier(max_depth=2, random_state=42), n_jobs=-1)"
    assert compare_text(nested_old, nested_new)[0] == Status.MATCH


def test_sklearn_printout_with_a_different_or_new_parameter_is_a_change():
    assert compare_text(FULL_TREE, "DecisionTreeClassifier(max_depth=3, random_state=42)")[0] != Status.MATCH
    assert compare_text(FULL_TREE, "DecisionTreeClassifier(max_depth=2, random_state=42, ccp_alpha=0.1)")[0] != Status.MATCH
    assert compare_text(FULL_TREE, "RandomForestClassifier(max_depth=2, random_state=42)")[0] != Status.MATCH


# ---------- cells that are meant to fail ----------

def test_expected_errors_are_read_from_saved_outputs():
    nb = _nb(_code("x = 1"), _code("1 + None", error="TypeError"), nbformat.v4.new_markdown_cell("text"))
    assert expected_errors(nb) == {1: "TypeError"}


def test_an_intentional_error_still_counts_and_the_rest_runs():
    nb = _nb(_code("vals = [1, None]"), _code("sum(vals)", error="TypeError"), _code("print('after')", stdout="after\n"))
    run = execute_notebook(nb, timeout=60)
    assert run.ok, run.error_summary()
    assert all("tags" not in c.metadata for c in run.notebook.cells)  # the helper tag is removed again
    report = compare_notebooks(nb, run.notebook)
    assert report.verified
    assert report.cells[1].status == Status.MATCH and "TypeError" in report.cells[1].detail


def test_an_intentional_error_that_changes_type_is_a_real_failure():
    nb = _nb(_code("import math\nmath.no_such_function()", error="TypeError"), _code("print('after')", stdout="after\n"))
    run = execute_notebook(nb, timeout=60)
    assert not run.ok and run.failed_cell == 0 and run.error_name == "AttributeError"


def test_an_intentional_error_that_no_longer_happens_is_reported():
    original = _nb(_code("1 + None", error="TypeError"))
    executed = _nb(_code("1 + 1", result="2"))
    report = compare_notebooks(original, executed)
    assert not report.verified and "raised nothing" in report.cells[0].detail


# ---------- patches that fake the check ----------

def _gaming(before_src, after_src, saved=None):
    original = _nb(_code(before_src, stdout=saved))
    patched = _nb(_code(after_src))
    return gaming_reasons(original, original, patched, original)


def test_pasting_expected_output_into_code_is_rejected():
    saved = "ValueError: Indexes have overlapping values: Int64Index([0, 1], dtype='int64')\n"
    before = "try:\n    pd.concat([x, y], verify_integrity=True)\nexcept ValueError as e:\n    print('ValueError:', e)"
    after = before.replace("print('ValueError:', e)", "print('ValueError:', str(e).replace('Index(', 'Int64Index('))")
    assert any("expected output" in r for r in _gaming(before, after, saved))


def test_print_hooks_and_monkey_patching_are_rejected():
    assert any("__repr__" in r for r in _gaming("tree", "type(tree).__repr__ = lambda self: 'old'\ntree"))
    assert any("monkey-patching" in r for r in _gaming("df1.append(df2)", "df1.append = lambda o: pd.concat([df1, o])\ndf1.append(df2)"))


def test_ordinary_fixes_are_not_flagged():
    assert _gaming("km = KMeans(algorithm='full')", "km = KMeans(algorithm='lloyd', n_init=10)", "1975.62\n") == []
    assert _gaming("pd.read_csv(f, delim_whitespace=True)", "pd.read_csv(f, sep=r'\\s+')") == []
    assert _gaming("df = df.append(row, ignore_index=True)", "df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)") == []
    # A string the notebook already used elsewhere may be reused.
    original = _nb(_code("display('pd.concat([df1, df2])')", stdout="pd.concat([df1, df2])\n"), _code("display('df1.append(df2)')"))
    patched = _nb(_code("display('pd.concat([df1, df2])')"), _code("display('pd.concat([df1, df2])')"))
    assert gaming_reasons(original, original, patched, original) == []


# ---------- plots (Raschka, Python Machine Learning, ch11) ----------

def test_old_matplotlib_figure_text_is_not_compared():
    # Saved before matplotlib 2.0: the figure's text stand-in was "<matplotlib.figure.Figure at 0x...>".
    # Today plt.show() leaves "<Figure size 640x480 with 1 Axes>" or nothing; plots are not compared.
    fig_old = nbformat.v4.new_output(
        "display_data", data={"image/png": "AAAA", "text/plain": "<matplotlib.figure.Figure at 0x7f1c2b3d4e50>"}
    )
    fig_new = nbformat.v4.new_output("display_data", data={"image/png": "BBBB", "text/plain": "<Figure size 640x480 with 1 Axes>"})
    original = _nb(_code("plt.scatter(x, y)\nplt.show()"), _code("print(1)", stdout="1\n"))
    original.cells[0].outputs = [fig_old]
    for outputs in ([], [fig_new]):
        executed = _nb(_code("plt.scatter(x, y)\nplt.show()"), _code("print(1)", stdout="1\n"))
        executed.cells[0].outputs = outputs
        report = compare_notebooks(original, executed)
        assert report.cells[0].status == Status.SKIPPED
        assert report.verified


def test_replacing_plt_show_with_a_figure_expression_is_rejected():
    before = "plt.scatter(x, y)\nplt.tight_layout()\nplt.show()"
    for last in ("plt.gcf()", "fig = plt.gcf(); fig", "fig = plt.figure(); fig", "plt.figure()"):
        reasons = _gaming(before, before.replace("plt.show()", last))
        assert any("plt.show()" in r for r in reasons), last
    # Changing other lines of a plotting cell is fine.
    assert _gaming(before, before.replace("plt.scatter(x, y)", "plt.scatter(x, y, c='r')")) == []
