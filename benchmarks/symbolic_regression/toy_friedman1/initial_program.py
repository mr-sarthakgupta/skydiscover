"""
Baseline symbolic regression candidate.

The evaluator will call:
  fit_and_predict(X_train, y_train, X_test) -> {"y_pred": ..., "equation": ..., "equation_sympy": ...}

This baseline is "PySR-inspired" in spirit (searching over a small library of nonlinear transforms),
but kept lightweight: it fits a sparse linear model over transformed features, then converts it to
a SymPy expression for interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Transform:
    name: str

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sympy(self, s):
        raise NotImplementedError

    def to_str(self, var: str) -> str:
        raise NotImplementedError


class Identity(Transform):
    def __init__(self):
        super().__init__(name="id")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x

    def sympy(self, s):
        return s

    def to_str(self, var: str) -> str:
        return var


class Square(Transform):
    def __init__(self):
        super().__init__(name="sq")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x * x

    def sympy(self, s):
        return s ** 2

    def to_str(self, var: str) -> str:
        return f"({var})**2"


class Sin(Transform):
    def __init__(self):
        super().__init__(name="sin")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def sympy(self, s):
        import sympy as sp

        return sp.sin(s)

    def to_str(self, var: str) -> str:
        return f"sin({var})"


class Cos(Transform):
    def __init__(self):
        super().__init__(name="cos")

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def sympy(self, s):
        import sympy as sp

        return sp.cos(s)

    def to_str(self, var: str) -> str:
        return f"cos({var})"


class Log1pAbs(Transform):
    def __init__(self, eps: float = 1e-12):
        super().__init__(name="log1pabs")
        self.eps = float(eps)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.abs(x) + self.eps)

    def sympy(self, s):
        import sympy as sp

        return sp.log(1 + sp.Abs(s) + self.eps)

    def to_str(self, var: str) -> str:
        # Keep it readable; evaluator treats this as informational.
        return f"log(1 + abs({var}) + {self.eps:g})"


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma <= 1e-12, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def _make_library(n_features: int) -> List[Tuple[int, Transform]]:
    transforms: List[Transform] = [Identity(), Square(), Sin(), Cos(), Log1pAbs()]
    lib: List[Tuple[int, Transform]] = []
    for j in range(n_features):
        for t in transforms:
            lib.append((j, t))
    return lib


def _build_design_matrix(X: np.ndarray, lib: List[Tuple[int, Transform]]) -> np.ndarray:
    feats = []
    for j, t in lib:
        feats.append(t.apply(X[:, j]))
    Phi = np.column_stack(feats)
    # Add bias term as last column.
    Phi = np.column_stack([Phi, np.ones((Phi.shape[0], 1), dtype=float)])
    return Phi


def _fit_sparse_linear(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit a sparse linear model. Uses LassoCV when available, otherwise ridge fallback.
    """
    try:
        from sklearn.linear_model import LassoCV

        model = LassoCV(
            cv=5,
            n_alphas=50,
            fit_intercept=False,
            max_iter=20000,
            random_state=0,
        )
        model.fit(Phi, y)
        return model.coef_.astype(float)
    except Exception:
        # Stable fallback
        lam = 1e-3
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        b = Phi.T @ y
        w = np.linalg.solve(A, b)
        return w.astype(float)


def _predict(Phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    return Phi @ w


def _to_sympy(
    w: np.ndarray,
    lib: List[Tuple[int, Transform]],
    feature_symbols: List,
    coef_threshold: float = 1e-8,
):
    import sympy as sp

    expr = sp.Float(0.0)
    # Last coefficient is bias (because we appended ones).
    bias = float(w[-1])
    if abs(bias) > coef_threshold:
        expr += sp.Float(bias)

    for k, (j, t) in enumerate(lib):
        ck = float(w[k])
        if abs(ck) <= coef_threshold:
            continue
        expr += sp.Float(ck) * t.sympy(feature_symbols[j])

    return sp.simplify(expr)


def _to_equation_string(
    w: np.ndarray,
    lib: List[Tuple[int, Transform]],
    n_features: int,
    coef_threshold: float = 1e-8,
) -> str:
    """
    Build a readable equation string without requiring SymPy.
    Variables are named x1..xN (matching evaluator conventions).
    """
    terms: List[str] = []

    bias = float(w[-1])
    if abs(bias) > coef_threshold:
        terms.append(f"{bias:.6g}")

    for k, (j, t) in enumerate(lib):
        ck = float(w[k])
        if abs(ck) <= coef_threshold:
            continue
        var = f"x{j+1}"
        feat = t.to_str(var)
        terms.append(f"({ck:.6g})*({feat})")

    if not terms:
        return "0.0"
    return " + ".join(terms)


def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> Dict[str, object]:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_test = np.asarray(X_test, dtype=float)

    # Standardize features for numerical stability. (We keep equation in standardized space;
    # the evaluator only uses predictions, and equation is for interpretability.)
    Xs_train, mu, sigma = _standardize(X_train)
    Xs_test = (X_test - mu) / sigma

    lib = _make_library(Xs_train.shape[1])
    Phi_tr = _build_design_matrix(Xs_train, lib)
    Phi_te = _build_design_matrix(Xs_test, lib)

    w = _fit_sparse_linear(Phi_tr, y_train)
    y_pred = _predict(Phi_te, w)

    eq_str = _to_equation_string(w, lib, Xs_train.shape[1])

    try:
        import sympy as sp

        feature_symbols = [sp.Symbol(f"x{i+1}") for i in range(Xs_train.shape[1])]
        expr = _to_sympy(w, lib, feature_symbols)
        eq_sym = expr
    except Exception:
        eq_sym = None

    return {
        "y_pred": np.asarray(y_pred, dtype=float),
        "equation": eq_str,
        "equation_sympy": eq_sym,
    }

