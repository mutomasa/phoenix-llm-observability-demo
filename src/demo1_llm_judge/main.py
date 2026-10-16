"""Demo 1: LLM as a Judge の CLI。

uv run python -m src.demo1_llm_judge.main --experiment all
"""

import argparse
import sys
from pathlib import Path

from src.common.config import Settings, load_settings
from src.common.llm import (
    LLMConnectionError,
    check_llm_health,
    create_judge_llm,
    create_llm_client,
    judge_invocation_params,
)
from src.common.phoenix_client import PhoenixUnavailableError, create_phoenix_client
from src.common.tracing import setup_tracing, shutdown_tracing
from src.demo1_llm_judge.agreement import overall_agreement
from src.demo1_llm_judge.dataset import DatasetFormatError, load_samples, upload_dataset
from src.demo1_llm_judge.evaluators import create_judge_evaluators
from src.demo1_llm_judge.experiments import (
    JUDGE_VALIDATION,
    ExperimentRunner,
    ExperimentSummary,
)
from src.demo1_llm_judge.generation import PROMPT_VARIANTS, AnswerGenerator

DEFAULT_DATASET = Path("datasets/qa_samples.jsonl")
EXPERIMENT_CHOICES = [*PROMPT_VARIANTS, JUDGE_VALIDATION, "all"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo 1: LLM as a Judge with Arize Phoenix")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment", choices=EXPERIMENT_CHOICES, default="all")
    return parser.parse_args(argv)


def selected_experiments(choice: str) -> list[str]:
    return [*PROMPT_VARIANTS, JUDGE_VALIDATION] if choice == "all" else [choice]


def print_summary(summary: ExperimentSummary) -> None:
    print(f"\n=== Experiment: {summary.name} ({summary.experiment_id}) ===")
    for name, value in summary.mean_scores.items():
        print(f"  {name:<28} {value:.3f}")
    if summary.name == JUDGE_VALIDATION:
        agreement = overall_agreement(summary.mean_scores)
        if agreement is not None:
            print(f"  {'overall agreement':<28} {agreement:.3f}")


def build_runner(settings: Settings, dataset_path: Path) -> ExperimentRunner:
    client = create_phoenix_client(settings.phoenix)
    dataset = upload_dataset(client, load_samples(dataset_path))
    judges = create_judge_evaluators(
        create_judge_llm(settings.llm), judge_invocation_params(settings.llm)
    )
    return ExperimentRunner(client, dataset, judges)


def run_experiment(runner: ExperimentRunner, name: str, settings: Settings) -> ExperimentSummary:
    if name == JUDGE_VALIDATION:
        return runner.run_judge_validation()
    generator = AnswerGenerator(create_llm_client(settings.llm), PROMPT_VARIANTS[name])
    return runner.run_generation(generator)


def run(args: argparse.Namespace, settings: Settings) -> None:
    check_llm_health(settings.llm.base_url)
    runner = build_runner(settings, args.dataset)
    # LangChain を計装し、生成LLM呼び出しを Experiment の Trace に含める。
    # Experiment 内の Span は Phoenix が Experiment 専用 Project に振り替える。
    tracer_provider = setup_tracing(
        settings.phoenix.judge_project_name, settings.phoenix.collector_endpoint
    )
    try:
        for name in selected_experiments(args.experiment):
            print_summary(run_experiment(runner, name, settings))
    finally:
        shutdown_tracing(tracer_provider)
    print(f"\nPhoenix: {settings.phoenix.base_url}/datasets")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args, load_settings())
    except (LLMConnectionError, PhoenixUnavailableError, DatasetFormatError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
