# Symbolic Regression Benchmarks

This folder contains benchmarks where the objective is **symbolic regression**: given a tabular dataset \((X, y)\), discover an interpretable equation that predicts \(y\) from \(X\).

SkyDiscover treats each candidate as a **Python program**. The evaluator loads the dataset, splits into train/validation, and calls the candidate program’s `fit_and_predict(...)` to produce predictions and (optionally) a human-readable equation.

## Contract (candidate program)

Your `initial_program.py` (and any evolved programs) must define:

- `fit_and_predict(X_train, y_train, X_test) -> dict`

The returned dict must include:

- `y_pred`: 1D array-like of predictions for `X_test`

Optional keys (strongly recommended):

- `equation`: string representation of the discovered equation
- `equation_sympy`: a SymPy expression or string parseable by SymPy

## Running the included example

From the repo root:

```bash
uv run skydiscover-run benchmarks/symbolic_regression/toy_friedman1/initial_program.py \
  benchmarks/symbolic_regression/toy_friedman1/evaluator.py \
  -c benchmarks/symbolic_regression/toy_friedman1/config_evox.yaml
```

This uses the EvoX search strategy (self-evolving search database) to evolve the candidate program for better validation performance and simpler equations.

## Adapting to your dataset (SRBench-style)

SRBench datasets are typically tabular with a single target column. To adapt:

- Put your dataset as `data.csv` in a new benchmark folder.
- Ensure the target column name matches `target` in the config (default: `y`).
- Update `feature_cols` if you want to restrict inputs (otherwise all non-target numeric columns are used).

The SRBench project is a good source of datasets and conventions: [cavalab/srbench](https://github.com/cavalab/srbench).

