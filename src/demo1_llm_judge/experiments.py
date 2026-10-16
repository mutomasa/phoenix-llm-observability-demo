"""Phoenix Experiment の実行。"""

from collections.abc import Sequence
from dataclasses import dataclass

from phoenix.client import Client
from phoenix.client.resources.datasets import Dataset
from phoenix.client.resources.experiments.types import RanExperiment

from src.demo1_llm_judge.agreement import (
    make_agreement_evaluator,
    make_judge_task,
    mean_scores_by_name,
)
from src.demo1_llm_judge.evaluators import RubricJudgeEvaluator
from src.demo1_llm_judge.generation import AnswerGenerator, make_generation_task

JUDGE_VALIDATION = "judge-validation"


@dataclass(frozen=True)
class ExperimentSummary:
    name: str
    experiment_id: str
    mean_scores: dict[str, float]


def summarize(name: str, experiment: RanExperiment) -> ExperimentSummary:
    return ExperimentSummary(
        name=name,
        experiment_id=experiment["experiment_id"],
        mean_scores=mean_scores_by_name(experiment["evaluation_runs"]),
    )


class ExperimentRunner:
    def __init__(self, client: Client, dataset: Dataset, judges: Sequence[RubricJudgeEvaluator]):
        self._client = client
        self._dataset = dataset
        self._judges = list(judges)

    def run_generation(self, generator: AnswerGenerator) -> ExperimentSummary:
        """プロンプト別に回答を生成し、Judge で評価する。"""
        variant = generator.variant
        experiment = self._client.experiments.run_experiment(
            dataset=self._dataset,
            task=make_generation_task(generator),
            evaluators=self._judges,
            experiment_name=variant.name,
            experiment_description=variant.description,
            experiment_metadata={
                "prompt_variant": variant.name,
                "system_prompt": variant.system_prompt,
            },
            timeout=180,
        )
        return summarize(variant.name, experiment)

    def run_judge_validation(self) -> ExperimentSummary:
        """人手作成の回答を Judge で判定し、人手ラベルとの一致率を測る。"""
        evaluators = [make_agreement_evaluator(judge.name) for judge in self._judges]
        experiment = self._client.experiments.run_experiment(
            dataset=self._dataset,
            task=make_judge_task(self._judges),
            evaluators=evaluators,
            experiment_name=JUDGE_VALIDATION,
            experiment_description="Judge の判定と人手ラベルの一致率検証",
            timeout=300,
        )
        return summarize(JUDGE_VALIDATION, experiment)
