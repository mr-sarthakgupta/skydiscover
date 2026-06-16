"""Regression tests for disease-relevant symbolic-regression anti-hacking guards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest

BENCH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "pysr_symbolic"
    / "disease_relevant_noninhibited_all"
)
EVAL_PATH = BENCH / "evaluator.py"
INITIAL_PATH = BENCH / "initial_program.py"


def _load_evaluator():
    name = "disease_relevant_eval_guard_test_mod"
    spec = importlib.util.spec_from_file_location(name, EVAL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_initial_program_satisfies_static_reward_guard():
    ev = _load_evaluator()

    ev.validate_candidate_source(str(INITIAL_PATH))


def test_rejects_top_level_spray_and_pray(tmp_path):
    program = tmp_path / "top_level_spray.py"
    program.write_text(
        dedent(
            '''
from __future__ import annotations
from typing import Any
import numpy as np
import sympy as sp
from numpy.typing import NDArray
from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)

_dummy_X = np.array([[0.0, 1.0], [1.0, 1.0]])
_dummy_y = np.array([0.0, 1.0])
x = feature_symbols(2)
c = constant_symbols(1)
_candidates = [c[0] * x[0], c[0] * x[0] ** 2, sp.sin(c[0] * x[0])]
_scores = [
    evaluate_expression(expr, _dummy_X, _dummy_y, _dummy_X, _dummy_y, constants=c)
    for expr in _candidates
]
_BEST_EXPR = _candidates[
    max(range(len(_scores)), key=lambda i: _scores[i]["combined_score"])
]

def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    return evaluate_expression(_BEST_EXPR, X_train, y_train, X_val, y_val, constants=c)
'''
        ).lstrip()
    )
    ev = _load_evaluator()

    with pytest.raises(ValueError, match="top-level executable code"):
        ev.validate_candidate_source(str(program))


def test_allows_non_executed_main_guard(tmp_path):
    program = tmp_path / "with_main_guard.py"
    program.write_text(
        dedent(
            '''
from __future__ import annotations
from typing import Any
import sympy as sp
from numpy.typing import NDArray
from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)

def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(1)
    expression = c[0] * x[0]
    return evaluate_expression(expression, X_train, y_train, X_val, y_val, constants=c)

if __name__ == "__main__":
    print("local smoke only")
'''
        ).lstrip()
    )
    ev = _load_evaluator()

    ev.validate_candidate_source(str(program))


def test_rejects_main_guard_else_branch(tmp_path):
    program = tmp_path / "main_guard_else.py"
    program.write_text(
        dedent(
            '''
from __future__ import annotations
from typing import Any
import sympy as sp
from numpy.typing import NDArray
from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)

def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(1)
    expression = c[0] * x[0]
    return evaluate_expression(expression, X_train, y_train, X_val, y_val, constants=c)

if __name__ == "__main__":
    pass
else:
    print("executes during import")
'''
        ).lstrip()
    )
    ev = _load_evaluator()

    with pytest.raises(ValueError, match="top-level executable code"):
        ev.validate_candidate_source(str(program))


@pytest.mark.parametrize(
    ("import_line", "helper_header", "message"),
    [
        (
            "import sympy",
            "def helper():\n    pass",
            "sympy must be imported canonically",
        ),
        (
            "import numpy as sp\nimport sympy as sp",
            "def helper():\n    pass",
            "numpy may only be imported",
        ),
        (
            "from pysr_harness.equation_session import evaluate_expression as _real_evaluate_expression, constant_symbols, feature_symbols",
            "def helper():\n    pass",
            "harness imports may not use aliases",
        ),
        (
            "from pysr_harness.equation_session import constant_symbols, evaluate_expression, feature_symbols",
            "def evaluate_expression(*args, **kwargs):\n    pass",
            "shadows a protected",
        ),
    ],
)
def test_rejects_harness_aliases_and_shadowed_scorer_names(
    tmp_path, import_line, helper_header, message
):
    program = tmp_path / "shadowed_scorer.py"
    program.write_text(
        dedent(
            f'''
from __future__ import annotations
from typing import Any
from numpy.typing import NDArray
{import_line}
from pysr_harness.equation_session import constant_symbols, evaluate_expression, feature_symbols

{helper_header}

def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(1)
    expression = c[0] * x[0]
    return evaluate_expression(expression, X_train, y_train, X_val, y_val, constants=c)
'''
        ).lstrip()
    )
    ev = _load_evaluator()

    with pytest.raises(ValueError, match=message):
        ev.validate_candidate_source(str(program))


@pytest.mark.parametrize(
    ("function_header", "message"),
    [
        ("@evaluate_expression\ndef evaluate_symbolic_candidate", "decorators"),
        ("def evaluate_symbolic_candidate", "default arguments"),
    ],
)
def test_rejects_import_time_function_expressions(tmp_path, function_header, message):
    program = tmp_path / "import_time_function_expr.py"
    signature_tail = (
        "(\n"
        "    X_train: NDArray,\n"
        "    y_train: NDArray,\n"
        "    X_val: NDArray,\n"
        "    y_val: NDArray,\n"
        "    leaked=evaluate_expression,\n"
        ") -> dict[str, Any]:"
        if message == "default arguments"
        else "(\n"
        "    X_train: NDArray,\n"
        "    y_train: NDArray,\n"
        "    X_val: NDArray,\n"
        "    y_val: NDArray,\n"
        ") -> dict[str, Any]:"
    )
    program.write_text(
        dedent(
            f'''
from __future__ import annotations
from typing import Any
import sympy as sp
from numpy.typing import NDArray
from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)

{function_header}{signature_tail}
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(1)
    expression = c[0] * x[0]
    return evaluate_expression(expression, X_train, y_train, X_val, y_val, constants=c)
'''
        ).lstrip()
    )
    ev = _load_evaluator()

    with pytest.raises(ValueError, match=message):
        ev.validate_candidate_source(str(program))


def test_rejects_runtime_evaluated_function_annotations(tmp_path):
    program = tmp_path / "annotation_stash.py"
    program.write_text(
        dedent(
            '''
from typing import Any
import sympy as sp
from numpy.typing import NDArray
from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)

def stash() -> feature_symbols(2)[0]:
    pass

def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    c = constant_symbols(1)
    expression = c[0] * stash.__annotations__["return"]
    return evaluate_expression(expression, X_train, y_train, X_val, y_val, constants=c)
'''
        ).lstrip()
    )
    ev = _load_evaluator()

    with pytest.raises(ValueError, match="annotations require"):
        ev.validate_candidate_source(str(program))


@pytest.mark.parametrize(
    ("return_args", "extra_body", "message"),
    [
        (
            "expression, X_val, y_val, X_val, y_val, constants=c",
            "",
            "canonical data arguments",
        ),
        (
            "expression, X_train, y_train, X_val, y_val, constants=c, initial_values=[y_val[0]]",
            "",
            "dataset array 'y_val'",
        ),
        (
            "expression, X_train, y_train, X_val, y_val, constants=c",
            "    leak = X_val.shape[0]\n    expression = expression + 0 * leak\n",
            "dataset array 'X_val'",
        ),
        (
            "expression, X_train, y_train, X_val, y_val, constants=c",
            "    y_val *= 0\n",
            "dataset array 'y_val'",
        ),
        (
            "expression, X_train, y_train, X_val, y_val, constants=c, initial_values=[y_train[0]]",
            "",
            "dataset array 'y_train'",
        ),
        (
            "expression, X_train, y_train, X_val, y_val, constants=c",
            "    n = X_train.shape[0]\n    expression = expression + 0 * n\n",
            "dataset array 'X_train'",
        ),
    ],
)
def test_rejects_validation_split_leaks(tmp_path, return_args, extra_body, message):
    program = tmp_path / "validation_leak.py"
    program.write_text(
        dedent(
            f'''
from __future__ import annotations
from typing import Any
import sympy as sp
from numpy.typing import NDArray
from pysr_harness.equation_session import (
    constant_symbols,
    evaluate_expression,
    feature_symbols,
)

def evaluate_symbolic_candidate(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
) -> dict[str, Any]:
    x = feature_symbols(X_train.shape[1])
    c = constant_symbols(1)
    expression = c[0] * x[0]
{extra_body}    return evaluate_expression({return_args})
'''
        ).lstrip()
    )
    ev = _load_evaluator()

    with pytest.raises(ValueError, match=message):
        ev.validate_candidate_source(str(program))


def test_aggregate_rejects_failed_datasets():
    ev = _load_evaluator()

    result = ev._aggregate_per_dataset_results(
        {
            "easy": {
                "nmse_val": 0.1,
                "combined_score": 0.9,
                "equation": "x0",
                "equation_template": "x0",
            },
            "hard": {
                "nmse_val": float("inf"),
                "combined_score": 0.0,
                "error": "overflow",
            },
        }
    )

    assert result["combined_score"] == 0.0
    assert result["n_successful"] == 0
    assert "failed on 1/2 datasets" in result["error"]


def test_aggregate_applies_small_complexity_penalty():
    ev = _load_evaluator()

    simple = ev._aggregate_per_dataset_results(
        {
            "dataset": {
                "nmse_val": 1.0,
                "combined_score": 0.5,
                "complexity": 0.0,
                "equation": "x0",
                "equation_template": "x0",
            },
        }
    )
    complex_result = ev._aggregate_per_dataset_results(
        {
            "dataset": {
                "nmse_val": 1.0,
                "combined_score": 0.5,
                "complexity": 160.0,
                "equation": "x0",
                "equation_template": "x0",
            },
        }
    )

    assert simple["fit_score"] == 0.5
    assert simple["combined_score"] == 0.5
    assert complex_result["fit_score"] == 0.5
    assert complex_result["combined_score"] == pytest.approx(0.49)
    assert complex_result["parsimony_penalty_factor"] == pytest.approx(0.80)
