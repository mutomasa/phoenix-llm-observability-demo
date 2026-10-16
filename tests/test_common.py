import httpx
import openai
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.common.config import load_settings
from src.common.llm import (
    LLMClient,
    LLMConnectionError,
    build_extra_body,
    strip_thinking,
)
from src.common.phoenix_client import PhoenixUnavailableError, check_phoenix_health


def test_load_settings_uses_defaults_for_empty_env():
    settings = load_settings(env={})
    assert settings.llm.model == "Qwen/Qwen3-1.7B"
    assert settings.llm.enable_thinking is False
    assert settings.phoenix.judge_project_name == "demo-llm-judge"
    assert settings.phoenix.tracing_project_name == "demo-langgraph-tracing"


def test_load_settings_reads_env_values():
    settings = load_settings(
        env={"LLM_MODEL": "Qwen/Qwen3-0.6B", "LLM_MAX_TOKENS": "256", "LLM_ENABLE_THINKING": "true"}
    )
    assert settings.llm.model == "Qwen/Qwen3-0.6B"
    assert settings.llm.max_tokens == 256
    assert settings.llm.enable_thinking is True


def test_build_extra_body_disables_thinking():
    settings = load_settings(env={})
    assert build_extra_body(settings.llm) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_strip_thinking_removes_think_block():
    assert strip_thinking("<think>\nreasoning\n</think>\n\n答え") == "答え"


def test_llm_client_returns_model_output():
    client = LLMClient(FakeListChatModel(responses=["<think></think>こんにちは"]))
    assert client.generate("system", "user") == "こんにちは"


class _UnreachableModel(FakeListChatModel):
    def invoke(self, *args, **kwargs):
        request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
        raise openai.APIConnectionError(request=request)


def test_llm_client_wraps_connection_error():
    client = LLMClient(_UnreachableModel(responses=[""]))
    with pytest.raises(LLMConnectionError):
        client.generate("system", "user")


def test_check_phoenix_health_raises_when_unreachable():
    with pytest.raises(PhoenixUnavailableError):
        check_phoenix_health("http://127.0.0.1:9", timeout=0.5)
