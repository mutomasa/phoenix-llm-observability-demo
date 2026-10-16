from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.common.tracing import create_tracer_provider


def test_create_tracer_provider_sets_project_name():
    provider = create_tracer_provider("unit-test-project", "http://127.0.0.1:9/v1/traces")
    assert isinstance(provider, TracerProvider)
    assert provider.resource.attributes["openinference.project.name"] == "unit-test-project"
    provider.shutdown()


def test_tracer_provider_records_spans_in_memory():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("parent"):
        with provider.get_tracer("test").start_as_current_span("child"):
            pass
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["child"].parent.span_id == spans["parent"].context.span_id


# --- LangGraph + OpenInference ---

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from openinference.instrumentation.langchain import LangChainInstrumentor  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from src.common.llm import LLMClient  # noqa: E402
from src.demo2_langgraph.graph import AgentNodes, build_graph  # noqa: E402
from src.demo2_langgraph.nodes import (  # noqa: E402
    AnswerGenerator,
    Planner,
    Reviewer,
    ToolExecutor,
)
from src.demo2_langgraph.state import initial_state  # noqa: E402
from src.demo2_langgraph.tools import ToolExecutionError, create_default_registry  # noqa: E402

KNOWLEDGE = Path(__file__).parent.parent / "datasets" / "knowledge.json"


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield provider, exporter
    instrumentor.uninstrument()


def build_traced_agent(provider, responses, failing_tool=None):
    llm = LLMClient(FakeListChatModel(responses=responses))
    registry = create_default_registry(KNOWLEDGE)
    if failing_tool:
        registry = registry.with_failure(failing_tool)
    nodes = AgentNodes(
        planner=Planner(llm, registry),
        tool_executor=ToolExecutor(registry, provider.get_tracer("test")),
        answer_generator=AnswerGenerator(llm),
        reviewer=Reviewer(llm),
    )
    return build_graph(nodes)


def spans_by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_graph_nodes_are_traced_with_hierarchy(traced):
    provider, exporter = traced
    plan = '{"tool": "calculator", "tool_input": "2*3", "plan": "p"}'
    build_traced_agent(provider, [plan, "6", "OK"]).invoke(initial_state("2*3は？"))
    spans = spans_by_name(exporter)

    for node in ["planner", "tool_executor", "answer_generator", "reviewer"]:
        assert spans[node].parent.span_id == spans["LangGraph"].context.span_id

    tool_span = spans["tool.calculator"]
    assert tool_span.attributes["openinference.span.kind"] == "TOOL"
    assert tool_span.attributes["output.value"] == "6"
    assert tool_span.status.status_code == StatusCode.OK
    assert tool_span.parent.span_id == spans["tool_executor"].context.span_id

    llm_spans = [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert len(llm_spans) == 3


def test_tool_error_is_recorded_on_span(traced):
    provider, exporter = traced
    plan = '{"tool": "current_date", "tool_input": "", "plan": "p"}'
    with pytest.raises(ToolExecutionError):
        build_traced_agent(provider, [plan], failing_tool="current_date").invoke(
            initial_state("今日は？")
        )
    spans = spans_by_name(exporter)
    assert spans["tool.current_date"].status.status_code == StatusCode.ERROR
    assert spans["tool_executor"].status.status_code == StatusCode.ERROR
    assert spans["LangGraph"].status.status_code == StatusCode.ERROR
