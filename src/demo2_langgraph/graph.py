"""Planner → Tool Executor → Answer Generator → Reviewer のグラフ構築。"""

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.demo2_langgraph.nodes import (
    MAX_RETRIES_EXCEEDED,
    AnswerGenerator,
    Planner,
    Reviewer,
    ToolExecutor,
)
from src.demo2_langgraph.state import AgentState

PLANNER = "planner"
TOOL_EXECUTOR = "tool_executor"
ANSWER_GENERATOR = "answer_generator"
REVIEWER = "reviewer"


@dataclass(frozen=True)
class AgentNodes:
    planner: Planner
    tool_executor: ToolExecutor
    answer_generator: AnswerGenerator
    reviewer: Reviewer


def route_after_review(state: AgentState) -> str:
    """PASS または Retry 上限到達なら終了し、それ以外は回答を再生成する。"""
    if state.get("review_result") == "PASS" or state.get("status") == MAX_RETRIES_EXCEEDED:
        return END
    return ANSWER_GENERATOR


def build_graph(nodes: AgentNodes) -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node(PLANNER, nodes.planner)
    graph.add_node(TOOL_EXECUTOR, nodes.tool_executor)
    graph.add_node(ANSWER_GENERATOR, nodes.answer_generator)
    graph.add_node(REVIEWER, nodes.reviewer)
    graph.add_edge(START, PLANNER)
    graph.add_edge(PLANNER, TOOL_EXECUTOR)
    graph.add_edge(TOOL_EXECUTOR, ANSWER_GENERATOR)
    graph.add_edge(ANSWER_GENERATOR, REVIEWER)
    graph.add_conditional_edges(REVIEWER, route_after_review, [ANSWER_GENERATOR, END])
    return graph.compile()
