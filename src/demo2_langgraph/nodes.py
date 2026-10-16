"""LangGraph の各ノード。各ノードは State を受け取り、更新分の dict を返す。"""

import json
import re

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Tracer
from pydantic import BaseModel, ValidationError

from src.common.llm import LLMClient
from src.common.tracing import langchain_parent_context
from src.demo2_langgraph.state import AgentState
from src.demo2_langgraph.tools import ToolExecutionError, ToolRegistry

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
MAX_RETRIES_EXCEEDED = "max_retries_exceeded"


class ToolPlan(BaseModel):
    tool: str
    tool_input: str
    plan: str = ""


def parse_plan(text: str, valid_tools: list[str], fallback_input: str) -> ToolPlan:
    """LLM 出力から JSON を取り出す。不正な場合は knowledge_search にフォールバックする。"""
    match = _JSON_OBJECT.search(text)
    try:
        plan = ToolPlan.model_validate(json.loads(match.group(0))) if match else None
    except (json.JSONDecodeError, ValidationError):
        plan = None
    if plan is None or plan.tool not in valid_tools:
        return ToolPlan(tool="knowledge_search", tool_input=fallback_input, plan="fallback")
    return plan


class Planner:
    SYSTEM_PROMPT = (
        "あなたはタスクプランナーです。質問に答えるために使うツールを1つ選び、"
        "JSONのみを出力してください。\n"
        '形式: {"tool": "<ツール名>", "tool_input": "<ツールへの入力>", '
        '"plan": "<処理方針を一文で>"}\n利用可能なツール:\n'
    )

    def __init__(self, llm: LLMClient, registry: ToolRegistry) -> None:
        self._llm = llm
        self._registry = registry

    def __call__(self, state: AgentState) -> dict:
        system_prompt = self.SYSTEM_PROMPT + self._registry.describe()
        text = self._llm.generate(system_prompt, f"質問: {state['question']}")
        plan = parse_plan(text, self._registry.names, state["question"])
        return {"plan": plan.plan, "tool_name": plan.tool, "tool_input": plan.tool_input}


class ToolExecutor:
    """ツールを実行し、OpenInference の TOOL Span として記録する。"""

    def __init__(self, registry: ToolRegistry, tracer: Tracer | None = None) -> None:
        self._registry = registry
        self._tracer = tracer or trace.get_tracer(__name__)

    def __call__(self, state: AgentState) -> dict:
        name, tool_input = state["tool_name"], state["tool_input"]
        with self._tracer.start_as_current_span(
            f"tool.{name}", context=langchain_parent_context()
        ) as span:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value
            )
            span.set_attribute(SpanAttributes.TOOL_NAME, name)
            span.set_attribute(SpanAttributes.INPUT_VALUE, tool_input)
            result = self._run(span, name, tool_input)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, result)
            span.set_status(Status(StatusCode.OK))
        return {"tool_result": result}

    def _run(self, span: trace.Span, name: str, tool_input: str) -> str:
        try:
            return self._registry.get(name).run(tool_input)
        except ToolExecutionError as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


class AnswerGenerator:
    SYSTEM_PROMPT = (
        "あなたは親切なアシスタントです。"
        "ツールの実行結果を根拠に、質問へ日本語で簡潔に答えてください。"
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def __call__(self, state: AgentState) -> dict:
        return {"answer": self._llm.generate(self.SYSTEM_PROMPT, self.build_prompt(state))}

    @staticmethod
    def build_prompt(state: AgentState) -> str:
        prompt = f"質問: {state['question']}\nツール結果: {state['tool_result']}"
        if state.get("review_feedback"):
            prompt += f"\nレビュー指摘（必ず反映すること）: {state['review_feedback']}"
        return prompt


class Reviewer:
    """LLM で回答をレビューする。

    `forced_retries` 回までは LLM の判定にかかわらず RETRY とし、
    それ以降は PASS とすることでシナリオを決定的に再現する。
    """

    SYSTEM_PROMPT = (
        "あなたはレビュアーです。"
        "回答が質問に正しく答えているか確認し、改善点を一文で述べてください。"
    )
    FORCED_FEEDBACK = "より具体的に、ツール結果の数値や用語を明示して答えてください。"

    def __init__(self, llm: LLMClient, forced_retries: int = 0, max_retries: int = 2) -> None:
        self._llm = llm
        self._forced_retries = forced_retries
        self._max_retries = max_retries

    def __call__(self, state: AgentState) -> dict:
        comment = self._llm.generate(self.SYSTEM_PROMPT, self.build_prompt(state))
        retry_count = state.get("retry_count", 0)
        if retry_count >= self._forced_retries:
            return {"review_result": "PASS", "review_feedback": comment, "status": "completed"}
        return self._retry(retry_count, comment)

    def _retry(self, retry_count: int, comment: str) -> dict:
        """RETRY を返す。上限を超える場合は失敗状態で終了させる。"""
        exceeded = retry_count >= self._max_retries
        return {
            "review_result": "RETRY",
            "review_feedback": f"{self.FORCED_FEEDBACK}（LLMコメント: {comment}）",
            "retry_count": retry_count if exceeded else retry_count + 1,
            "status": MAX_RETRIES_EXCEEDED if exceeded else "retrying",
        }

    @staticmethod
    def build_prompt(state: AgentState) -> str:
        return (
            f"質問: {state['question']}\n"
            f"ツール結果: {state['tool_result']}\n"
            f"回答: {state['answer']}"
        )
