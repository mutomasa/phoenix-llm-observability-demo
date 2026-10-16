"""LangGraph Agent の State 定義。"""

from typing import Literal, TypedDict

ReviewResult = Literal["PASS", "RETRY", ""]


class AgentState(TypedDict, total=False):
    question: str
    plan: str
    tool_name: str
    tool_input: str
    tool_result: str
    answer: str
    review_result: ReviewResult
    review_feedback: str
    retry_count: int
    status: str


def initial_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        plan="",
        tool_name="",
        tool_input="",
        tool_result="",
        answer="",
        review_result="",
        review_feedback="",
        retry_count=0,
        status="running",
    )
