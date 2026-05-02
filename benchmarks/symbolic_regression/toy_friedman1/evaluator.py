"""
Symbolic regression evaluator.

The candidate program must define:
  fit_and_predict(X_train, y_train, X_test) -> dict with key "y_pred"

Optional return keys:
  - equation: str
  - equation_sympy: sympy.Expr | str
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    csv_path: str
    target: str = "y"
    feature_cols: Optional[Tuple[str, ...]] = None
    seed: int = 0
    val_frac: float = 0.25


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


SPEC = DatasetSpec(csv_path=_here("data.csv"), target="y", feature_cols=None, seed=0, val_frac=0.25)


def _load_csv(spec: DatasetSpec) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    import pandas as pd

    df = pd.read_csv(spec.csv_path)
    if spec.target not in df.columns:
        raise ValueError(f"Target column '{spec.target}' not found in CSV columns={list(df.columns)}")

    if spec.feature_cols is None:
        feature_cols = tuple(c for c in df.columns if c != spec.target)
    else:
        missing = [c for c in spec.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols contains missing columns: {missing}")
        feature_cols = tuple(spec.feature_cols)

    X = df.loc[:, feature_cols].to_numpy(dtype=float)
    y = df.loc[:, spec.target].to_numpy(dtype=float)
    return X, y, feature_cols


def _split_train_val(X: np.ndarray, y: np.ndarray, seed: int, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _equation_complexity(equation_sympy: Any) -> int:
    if equation_sympy is None:
        return 0
    try:
        import sympy as sp

        expr = equation_sympy
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        # Count nodes in expression tree.
        return int(sum(1 for _ in sp.preorder_traversal(expr)))
    except Exception:
        return 0


def _equation_complexity_fallback(equation_str: Any) -> int:
    """
    If SymPy isn't available, estimate complexity from a string form.
    We keep it simple: count non-empty additive terms.
    """
    if not isinstance(equation_str, str):
        return 0
    # Count '+'-separated terms (rough proxy for complexity).
    parts = [p.strip() for p in equation_str.split("+")]
    return int(sum(1 for p in parts if p))


def _load_candidate(program_path: str):
    # IMPORTANT: Under Python 3.13, dataclasses may consult sys.modules during class creation.
    # If we don't register the module before exec_module(), dataclass decoration can fail.
    module_name = f"candidate_program_{abs(hash(program_path))}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load candidate program from {program_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Returns a dict of metrics. Higher combined_score is better.
    """
    try:
        X, y, feature_cols = _load_csv(SPEC)
        X_tr, y_tr, X_val, y_val = _split_train_val(X, y, seed=SPEC.seed, val_frac=SPEC.val_frac)

        candidate = _load_candidate(program_path)
        if not hasattr(candidate, "fit_and_predict"):
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "error": "Candidate must define fit_and_predict(X_train, y_train, X_test) -> dict",
            }

        out = candidate.fit_and_predict(X_tr, y_tr, X_val)
        if not isinstance(out, dict) or "y_pred" not in out:
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "error": "fit_and_predict must return dict with key 'y_pred'",
            }

        y_pred = np.asarray(out["y_pred"], dtype=float).reshape(-1)
        if y_pred.shape[0] != y_val.shape[0]:
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "error": f"y_pred has wrong length {y_pred.shape[0]} (expected {y_val.shape[0]})",
            }

        if not np.all(np.isfinite(y_pred)):
            return {
                "validity": 0.0,
                "combined_score": 0.0,
                "error": "Non-finite predictions encountered",
            }

        rmse = _rmse(y_val, y_pred)
        r2 = _r2(y_val, y_pred)

        eq = out.get("equation")
        eq_sym = out.get("equation_sympy")
        if eq_sym is not None:
            complexity = _equation_complexity(eq_sym)
        else:
            complexity = _equation_complexity_fallback(eq)

        # Combined objective: prefer high R^2 but penalize complexity.
        # Keep it bounded and stable: map R^2 into [0, 1] via clamp, then subtract a small penalty.
        r2_clamped = float(max(0.0, min(1.0, r2)))
        penalty = 0.0025 * float(complexity)
        combined = r2_clamped - penalty

        metrics: Dict[str, Any] = {
            "validity": 1.0,
            "combined_score": float(combined),
            "r2_val": float(r2),
            "rmse_val": float(rmse),
            "equation_complexity": float(complexity),
        }

        # Helpful context for the run (kept small; these are not used directly for scoring).
        if isinstance(eq, str):
            metrics["has_equation"] = 1.0
        else:
            metrics["has_equation"] = 0.0

        # Provide short artifacts via metrics (SkyDiscover stores artifacts separately, but metrics are visible in DB).
        if isinstance(eq, str):
            metrics["equation_preview_len"] = float(min(len(eq), 500))

        # Include feature count in metrics for debugging/monitoring.
        metrics["num_features"] = float(len(feature_cols))
        metrics["num_train"] = float(X_tr.shape[0])
        metrics["num_val"] = float(X_val.shape[0])

        return metrics

    except Exception as e:
        return {
            "validity": 0.0,
            "combined_score": 0.0,
            "error": f"{e}",
            "traceback": traceback.format_exc(),
        }

