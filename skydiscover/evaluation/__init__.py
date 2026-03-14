import os
from typing import Optional, Union

from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.evaluator import Evaluator
from skydiscover.evaluation.llm_judge import LLMJudge

__all__ = [
    "EvaluationResult",
    "Evaluator",
    "ContainerizedEvaluator",
    "LLMJudge",
    "create_evaluator",
]


def create_evaluator(
    config,
    llm_judge: Optional[LLMJudge] = None,
    max_concurrent: int = 4,
) -> Union[Evaluator, ContainerizedEvaluator]:
    """Return the right evaluator for the given config.

    If config.evaluation_file points to a directory containing a Dockerfile
    and evaluate.sh, returns a ContainerizedEvaluator.  Otherwise returns
    the standard Python Evaluator.
    """
    path = config.evaluation_file or ""
    if (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "Dockerfile"))
        and os.path.exists(os.path.join(path, "evaluate.sh"))
    ):
        return ContainerizedEvaluator(path, config, max_concurrent=max_concurrent)
    return Evaluator(config, llm_judge=llm_judge, max_concurrent=max_concurrent)
