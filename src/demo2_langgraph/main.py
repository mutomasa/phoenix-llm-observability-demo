"""Demo 2: LangGraph Tracing の CLI。

uv run python -m src.demo2_langgraph.main --scenario normal|retry|error
"""

import argparse
import sys

from openinference.instrumentation import using_attributes

from src.common.config import Settings, load_settings
from src.common.llm import LLMConnectionError, check_llm_health, create_llm_client
from src.common.phoenix_client import PhoenixUnavailableError, check_phoenix_health
from src.common.tracing import setup_tracing, shutdown_tracing
from src.demo2_langgraph.scenarios import SCENARIOS, Scenario, build_agent
from src.demo2_langgraph.state import AgentState, initial_state
from src.demo2_langgraph.tools import ToolExecutionError, create_default_registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo 2: LangGraph Tracing with Arize Phoenix")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="normal")
    parser.add_argument("--question", help="シナリオ既定の質問を上書きする")
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args(argv)


def resolve_question(scenario: Scenario, override: str | None) -> str:
    return override or scenario.question


def print_result(state: AgentState) -> None:
    print(f"plan        : {state.get('plan')}")
    print(f"tool        : {state.get('tool_name')}({state.get('tool_input')!r})")
    print(f"tool_result : {state.get('tool_result')}")
    print(f"answer      : {state.get('answer')}")
    print(f"review      : {state.get('review_result')} (retry_count={state.get('retry_count')})")
    print(f"status      : {state.get('status')}")


def invoke_agent(args: argparse.Namespace, settings: Settings) -> AgentState:
    scenario = SCENARIOS[args.scenario]
    agent = build_agent(
        scenario, create_llm_client(settings.llm), create_default_registry(), args.max_retries
    )
    print(f"[{scenario.name}] {scenario.description}")
    with using_attributes(metadata={"scenario": scenario.name}, tags=[f"scenario:{scenario.name}"]):
        return agent.invoke(initial_state(resolve_question(scenario, args.question)))


def run(args: argparse.Namespace, settings: Settings) -> int:
    check_llm_health(settings.llm.base_url)
    check_phoenix_health(settings.phoenix.base_url)
    tracer_provider = setup_tracing(
        settings.phoenix.tracing_project_name, settings.phoenix.collector_endpoint
    )
    try:
        print_result(invoke_agent(args, settings))
        return 0
    except ToolExecutionError as exc:
        print(f"[TOOL ERROR] {exc}（PhoenixでエラーSpanを確認してください）", file=sys.stderr)
        return 2
    finally:
        shutdown_tracing(tracer_provider)
        print(f"Phoenix: {settings.phoenix.base_url}/projects")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args, load_settings())
    except (LLMConnectionError, PhoenixUnavailableError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
