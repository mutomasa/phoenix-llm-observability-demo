"""OpenTelemetry / OpenInference によるトレーシング設定。"""

from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from phoenix.otel import register


def create_tracer_provider(project_name: str, endpoint: str) -> TracerProvider:
    """Phoenix の指定 Project へ OTLP HTTP で Span を送る TracerProvider を生成する。"""
    return register(
        project_name=project_name,
        endpoint=endpoint,
        protocol="http/protobuf",
        batch=True,
        verbose=False,
    )


def instrument_langchain(tracer_provider: TracerProvider) -> LangChainInstrumentor:
    """LangChain / LangGraph の自動計装を有効化する。"""
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    return instrumentor


def setup_tracing(project_name: str, endpoint: str) -> TracerProvider:
    tracer_provider = create_tracer_provider(project_name, endpoint)
    instrument_langchain(tracer_provider)
    return tracer_provider


def shutdown_tracing(tracer_provider: TracerProvider) -> None:
    """バッファ中の Span を送信してから終了する。"""
    tracer_provider.force_flush()
    tracer_provider.shutdown()


def langchain_parent_context() -> Context | None:
    """実行中の LangChain/LangGraph ノードの Span を親とする Context を返す。

    OpenInference の LangChain 計装は OTel の current context を更新しないため、
    手動 Span をノードの子にするにはこの Context を明示的に渡す必要がある。
    """
    parent = get_current_span()
    return trace.set_span_in_context(parent) if parent is not None else None
