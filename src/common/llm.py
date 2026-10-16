"""両デモで共有するLLMクライアント。

- 生成・Agent用: LangChain `ChatOpenAI` を `LLMClient` で包む（OpenInferenceで自動計装される）
- Judge用: Phoenix Evals の `LLM` を同じ設定から生成する
どちらも vLLM の OpenAI互換API に接続する。
"""

import re

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from phoenix.evals import LLM

from src.common.config import LLMSettings

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMConnectionError(RuntimeError):
    """vLLM に接続できない場合に送出する。"""


def strip_thinking(text: str) -> str:
    """Qwen3 が出力する `<think>...</think>` ブロックを除去する。"""
    return _THINK_BLOCK.sub("", text).strip()


def build_extra_body(settings: LLMSettings) -> dict:
    """vLLM 固有のリクエストパラメータ（Qwen3の思考モード切替）を生成する。"""
    return {"chat_template_kwargs": {"enable_thinking": settings.enable_thinking}}


def create_chat_model(settings: LLMSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        max_retries=1,
        extra_body=build_extra_body(settings),
    )


def create_judge_llm(settings: LLMSettings) -> LLM:
    """Phoenix Evals 用の LLM ラッパーを生成する。"""
    return LLM(
        provider="openai",
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        sync_client_kwargs={"timeout": settings.timeout_seconds},
        async_client_kwargs={"timeout": settings.timeout_seconds},
    )


def judge_invocation_params(settings: LLMSettings) -> dict:
    """Phoenix Evals の Evaluator に渡すモデル呼び出しパラメータ。"""
    return {
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "extra_body": build_extra_body(settings),
    }


class LLMClient:
    """system/user プロンプトからテキストを生成する薄いクライアント。

    任意の `BaseChatModel` を受け取るため、テストではフェイクモデルに差し替えられる。
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        try:
            response = self._chat_model.invoke(messages)
        except openai.APIConnectionError as exc:
            raise LLMConnectionError(f"LLMサーバーに接続できません: {exc}") from exc
        return strip_thinking(str(response.content))


def create_llm_client(settings: LLMSettings) -> LLMClient:
    return LLMClient(create_chat_model(settings))
