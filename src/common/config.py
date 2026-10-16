"""環境変数からアプリケーション設定を読み込む。"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    enable_thinking: bool
    timeout_seconds: float


@dataclass(frozen=True)
class PhoenixSettings:
    base_url: str
    collector_endpoint: str
    judge_project_name: str
    tracing_project_name: str


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    phoenix: PhoenixSettings


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_llm_settings(env: Mapping[str, str]) -> LLMSettings:
    return LLMSettings(
        base_url=env.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=env.get("VLLM_API_KEY", "EMPTY"),
        model=env.get("LLM_MODEL", "Qwen/Qwen3-1.7B"),
        temperature=float(env.get("LLM_TEMPERATURE", "0")),
        max_tokens=int(env.get("LLM_MAX_TOKENS", "512")),
        enable_thinking=_to_bool(env.get("LLM_ENABLE_THINKING", "false")),
        timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "60")),
    )


def build_phoenix_settings(env: Mapping[str, str]) -> PhoenixSettings:
    return PhoenixSettings(
        base_url=env.get("PHOENIX_BASE_URL", "http://localhost:6006"),
        collector_endpoint=env.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
        judge_project_name=env.get("JUDGE_PROJECT_NAME", "demo-llm-judge"),
        tracing_project_name=env.get("TRACING_PROJECT_NAME", "demo-langgraph-tracing"),
    )


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """`.env` と環境変数から設定を生成する。`env` を渡すとそれだけを使う（テスト用）。"""
    if env is None:
        load_dotenv()
        env = os.environ
    return Settings(llm=build_llm_settings(env), phoenix=build_phoenix_settings(env))
