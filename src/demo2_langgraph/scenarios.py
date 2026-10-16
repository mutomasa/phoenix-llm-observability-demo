"""比較シナリオ（normal / retry / error）の定義と Agent の組み立て。"""

from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from src.common.llm import LLMClient
from src.demo2_langgraph.graph import AgentNodes, build_graph
from src.demo2_langgraph.nodes import AnswerGenerator, Planner, Reviewer, ToolExecutor
from src.demo2_langgraph.tools import ToolRegistry


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    description: str
    forced_retries: int = 0
    failing_tool: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "normal": Scenario(
        name="normal",
        question="128 × 46 はいくつですか？",
        description="Case A: Planner → Tool → Answer → Reviewer → END",
    ),
    "retry": Scenario(
        name="retry",
        question="Arize Phoenixとは何ですか？",
        description="Case B: Reviewer が1回 RETRY を返し、回答を再生成する",
        forced_retries=1,
    ),
    "error": Scenario(
        name="error",
        question="今日の日付を教えてください。",
        description="Case C: Tool Executor で例外を発生させる",
        failing_tool="current_date",
    ),
}


def scenario_registry(scenario: Scenario, registry: ToolRegistry) -> ToolRegistry:
    if scenario.failing_tool is None:
        return registry
    return registry.with_failure(scenario.failing_tool)


def build_agent(
    scenario: Scenario, llm: LLMClient, registry: ToolRegistry, max_retries: int = 2
) -> CompiledStateGraph:
    tools = scenario_registry(scenario, registry)
    nodes = AgentNodes(
        planner=Planner(llm, tools),
        tool_executor=ToolExecutor(tools),
        answer_generator=AnswerGenerator(llm),
        reviewer=Reviewer(llm, forced_retries=scenario.forced_retries, max_retries=max_retries),
    )
    return build_graph(nodes)
