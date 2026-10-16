"""評価対象となる QA アプリケーション（回答生成）。"""

from collections.abc import Callable
from dataclasses import dataclass

from src.common.llm import LLMClient


@dataclass(frozen=True)
class PromptVariant:
    name: str
    description: str
    system_prompt: str


PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "baseline": PromptVariant(
        name="baseline",
        description="短く答えるだけのシンプルなプロンプト",
        system_prompt="あなたはアシスタントです。質問に日本語で一文で答えてください。",
    ),
    "detailed": PromptVariant(
        name="detailed",
        description="根拠文書の遵守と網羅性を指示したプロンプト",
        system_prompt=(
            "あなたは正確さを重視するアシスタントです。以下を守って日本語で回答してください。\n"
            "- 根拠文書が与えられた場合は、その内容だけに基づいて答える\n"
            "- 質問が求める要素をすべて漏れなく含める\n"
            "- 不確かな情報や推測を付け加えない\n"
            "- 3文以内で簡潔にまとめる"
        ),
    ),
}


def build_user_prompt(question: str, context: str) -> str:
    if not context:
        return f"質問: {question}"
    return f"根拠文書:\n{context}\n\n質問: {question}"


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient, variant: PromptVariant) -> None:
        self._llm_client = llm_client
        self._variant = variant

    @property
    def variant(self) -> PromptVariant:
        return self._variant

    def generate(self, question: str, context: str = "") -> str:
        return self._llm_client.generate(
            self._variant.system_prompt, build_user_prompt(question, context)
        )


def make_generation_task(generator: AnswerGenerator) -> Callable[[dict], dict]:
    """Phoenix Experiment の task 関数を生成する（引数名 `input` で入力がバインドされる）。"""

    def generate_answer(input: dict) -> dict:  # noqa: A002 - Phoenixの引数名バインディングに合わせる
        return {"answer": generator.generate(input["question"], input.get("context", ""))}

    return generate_answer
