from datetime import date
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.common.llm import LLMClient
from src.demo2_langgraph.graph import route_after_review
from src.demo2_langgraph.nodes import MAX_RETRIES_EXCEEDED, AnswerGenerator, Reviewer, parse_plan
from src.demo2_langgraph.scenarios import SCENARIOS, Scenario, build_agent
from src.demo2_langgraph.state import initial_state
from src.demo2_langgraph.tools import (
    Calculator,
    CurrentDate,
    KnowledgeSearch,
    ToolExecutionError,
    create_default_registry,
)

KNOWLEDGE = Path(__file__).parent.parent / "datasets" / "knowledge.json"
PLAN = '{"tool": "calculator", "tool_input": "128*46", "plan": "掛け算する"}'


def fake_llm(*responses: str) -> LLMClient:
    return LLMClient(FakeListChatModel(responses=list(responses)))


def run_agent(scenario: Scenario, llm: LLMClient, max_retries: int = 2):
    agent = build_agent(scenario, llm, create_default_registry(KNOWLEDGE), max_retries)
    steps = []
    final = {}
    for update in agent.stream(initial_state(scenario.question), stream_mode="updates"):
        for node, values in update.items():
            steps.append(node)
            final.update(values)
    return steps, final


# --- tools ---


@pytest.mark.parametrize(
    ("expr", "expected"), [("128*46", "5888"), ("128 × 46", "5888"), ("7/2", "3.5")]
)
def test_calculator(expr, expected):
    assert Calculator().run(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "1 +", "open('x')"])
def test_calculator_rejects_unsafe_or_invalid_input(expr):
    with pytest.raises(ToolExecutionError):
        Calculator().run(expr)


def test_knowledge_search_finds_entry():
    assert KnowledgeSearch(KNOWLEDGE).run("Arize Phoenixとは").startswith("Arize Phoenix")


def test_knowledge_search_returns_not_found():
    assert "見つかりません" in KnowledgeSearch(KNOWLEDGE).run("unknown")


def test_current_date_uses_injected_clock():
    assert CurrentDate(lambda: date(2026, 1, 2)).run("") == "2026-01-02"


def test_failing_tool_registry_raises():
    registry = create_default_registry(KNOWLEDGE).with_failure("current_date")
    with pytest.raises(ToolExecutionError, match="injected"):
        registry.get("current_date").run("")


# --- nodes ---


def test_parse_plan_extracts_json_from_text():
    plan = parse_plan(f"はい。\n{PLAN}\n以上", ["calculator"], "Q")
    assert (plan.tool, plan.tool_input) == ("calculator", "128*46")


@pytest.mark.parametrize("text", ["no json", '{"tool": "web_search", "tool_input": "x"}'])
def test_parse_plan_falls_back_to_knowledge_search(text):
    plan = parse_plan(text, ["calculator", "knowledge_search"], "Q")
    assert (plan.tool, plan.tool_input) == ("knowledge_search", "Q")


def test_answer_prompt_includes_review_feedback():
    state = {"question": "Q", "tool_result": "R", "review_feedback": "具体的に"}
    assert "具体的に" in AnswerGenerator.build_prompt(state)


def test_reviewer_stops_at_retry_limit():
    reviewer = Reviewer(fake_llm("c"), forced_retries=10, max_retries=2)
    result = reviewer({"question": "Q", "tool_result": "R", "answer": "A", "retry_count": 2})
    assert result["status"] == MAX_RETRIES_EXCEEDED
    assert route_after_review(result) == "__end__"


# --- graph scenarios ---


def test_normal_scenario_runs_each_node_once():
    steps, final = run_agent(SCENARIOS["normal"], fake_llm(PLAN, "5888です", "問題なし"))
    assert steps == ["planner", "tool_executor", "answer_generator", "reviewer"]
    assert final["tool_result"] == "5888"
    assert (final["review_result"], final["status"]) == ("PASS", "completed")


def test_retry_scenario_regenerates_answer_once():
    llm = fake_llm(PLAN, "回答1", "もっと具体的に", "回答2", "OK")
    steps, final = run_agent(SCENARIOS["retry"], llm)
    assert steps == [
        "planner",
        "tool_executor",
        "answer_generator",
        "reviewer",
        "answer_generator",
        "reviewer",
    ]
    assert (final["answer"], final["retry_count"], final["review_result"]) == ("回答2", 1, "PASS")


def test_retry_limit_ends_graph_as_failure():
    scenario = Scenario(name="t", question="Q", description="", forced_retries=99)
    llm = fake_llm(PLAN, *["a", "r"] * 3)
    steps, final = run_agent(scenario, llm, max_retries=2)
    assert steps.count("answer_generator") == 3
    assert final["status"] == MAX_RETRIES_EXCEEDED


def test_error_scenario_raises_tool_error():
    plan = '{"tool": "current_date", "tool_input": "", "plan": "日付"}'
    with pytest.raises(ToolExecutionError):
        run_agent(SCENARIOS["error"], fake_llm(plan))
