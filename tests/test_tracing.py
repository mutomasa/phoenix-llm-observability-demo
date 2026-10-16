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
