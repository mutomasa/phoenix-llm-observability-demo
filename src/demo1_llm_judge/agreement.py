"""Judge の判定と人手ラベルの一致率を検証する。

Judge 自体を Experiment の task とし、出力ラベルと人手ラベルを
コード Evaluator で比較する（Phoenix で Judge の判定品質を確認できる）。
"""

from collections.abc import Callable, Sequence
from statistics import mean
from typing import Any

from phoenix.client.experiments import create_evaluator
from phoenix.client.resources.experiments.types import Evaluator, ExperimentEvaluationRun
from phoenix.evals import Score

from src.demo1_llm_judge.evaluators import NOT_APPLICABLE, RubricJudgeEvaluator

AGREEMENT_PREFIX = "agreement_"


def score_to_dict(score: Score) -> dict[str, Any]:
    return {"score": score.score, "label": score.label, "reason": score.explanation}


def judge_candidate(judges: Sequence[RubricJudgeEvaluator], input: dict, expected: dict) -> dict:  # noqa: A002
    """データセットの人手作成回答（candidate_answer）を全観点で判定する。"""
    eval_input = {
        "input": input,
        "expected": expected,
        "output": {"answer": expected["candidate_answer"]},
    }
    return {judge.name: score_to_dict(judge.evaluate(eval_input)[0]) for judge in judges}


def make_judge_task(judges: Sequence[RubricJudgeEvaluator]) -> Callable[[dict, dict], dict]:
    def judge_candidate_answer(input: dict, expected: dict) -> dict:  # noqa: A002
        return judge_candidate(judges, input, expected)

    return judge_candidate_answer


def compare_labels(judge_label: str | None, human_label: str | None) -> dict[str, Any]:
    if human_label is None or judge_label == NOT_APPLICABLE:
        return {"score": None, "label": NOT_APPLICABLE, "explanation": "人手ラベルなし"}
    matched = judge_label == human_label
    return {
        "score": 1.0 if matched else 0.0,
        "label": "match" if matched else "mismatch",
        "explanation": f"judge={judge_label}, human={human_label}",
    }


def make_agreement_evaluator(metric_name: str) -> Evaluator:
    """1観点の Judge ラベルと人手ラベルを比較するコード Evaluator を生成する。"""

    @create_evaluator(name=f"{AGREEMENT_PREFIX}{metric_name}", kind="code")
    def evaluator(output: dict, metadata: dict) -> dict:
        judge_label = output.get(metric_name, {}).get("label")
        human_label = metadata.get("human_labels", {}).get(metric_name)
        return compare_labels(judge_label, human_label)

    return evaluator


def mean_scores_by_name(runs: Sequence[ExperimentEvaluationRun]) -> dict[str, float]:
    """Evaluator 名ごとの平均スコア（スコアなしは除外）。"""
    grouped: dict[str, list[float]] = {}
    for run in runs:
        score = (run.result or {}).get("score") if isinstance(run.result, dict) else None
        if score is not None:
            grouped.setdefault(run.name, []).append(float(score))
    return {name: mean(values) for name, values in sorted(grouped.items())}


def overall_agreement(per_metric: dict[str, float]) -> float | None:
    values = [v for k, v in per_metric.items() if k.startswith(AGREEMENT_PREFIX)]
    return mean(values) if values else None
