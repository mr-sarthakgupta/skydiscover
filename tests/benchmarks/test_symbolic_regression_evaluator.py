"""Smoke tests for symbolic regression benchmark evaluator."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "symbolic_regression" / "toy_friedman1"
EVAL_PATH = BENCH / "evaluator.py"
INITIAL_PATH = BENCH / "initial_program.py"


def _load_evaluator():
    name = "sym_reg_eval_test_mod"
    spec = importlib.util.spec_from_file_location(name, EVAL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _fast_gp_bo(monkeypatch):
    """Keep GP-BO evaluation fast in CI."""
    monkeypatch.setenv("SKYD_SR_GP_N_CALLS", "10")


def test_symbolic_regression_evaluator_smoke():
    pytest.importorskip("pandas")
    pytest.importorskip("sympy")

    ev = _load_evaluator()
    r = ev.evaluate(str(INITIAL_PATH))
    assert r["validity"] == 1.0, r.get("error", r)
    assert "structural_complexity" in r
    assert "r2_val" in r
    assert r.get("has_equation_template") == 1.0


def test_structural_complexity_ignores_floats():
    pytest.importorskip("sympy")

    ev = _load_evaluator()
    import sympy as sp

    x = sp.Symbol("x")
    # many Float coefficients should not blow up structural complexity vs template-like form
    expr = sp.Float(2.718) * sp.sin(x) + sp.Float(1.414) * x**2
    s_struct = ev._structural_complexity(expr)
    s_full = ev._equation_complexity_full(expr)
    assert s_struct < s_full
    assert s_struct >= 5
