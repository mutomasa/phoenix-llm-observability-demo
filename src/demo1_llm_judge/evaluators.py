"""Phoenix Evals の `LLMEvaluator` を拡張した LLM Judge。

Judge は Structured Outputs（JSON Schema）で `{score, label, reason}` を返す。
小型モデルは不正なJSONや範囲外の値を返しうるため、Pydanticで検証し失敗時はリトライする。
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from phoenix.evals import LLM, LLMEvaluator, Score, bind_evaluator
from pydantic import BaseModel, Field, ValidationError

from src.demo1_llm_judge.dataset import Label

NOT_APPLICABLE = "not_applicable"
PARSE_ERROR = "parse_error"

JUDGE_INPUT_MAPPING = {
    "question": "input.question",
    "context": "input.context",
    "reference": "expected.reference",
    "answer": "output.answer",
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string", "enum": ["good", "partial", "bad"]},
        "reason": {"type": "string"},
    },
    "required": ["score", "label", "reason"],
}


class JudgeOutputError(ValueError):
    """Judge の出力がスキーマに適合しない場合に送出する。"""


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    label: Label
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    criterion: str
    requires_context: bool = False


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "correctness",
        "回答の内容が正解と整合しているか。誤った記述があれば減点する。"
        "記述が正しければ、情報が不足していても減点しない。",
    ),
    MetricSpec(
        "relevance",
        "回答が質問と同じ話題について答えているか。内容の正誤や情報の過不足は一切考慮しない。"
        "誤った内容でも、質問の話題に答えていれば good とする。",
    ),
    MetricSpec(
        "faithfulness",
        "回答が根拠文書の内容だけに基づいているか。根拠文書にない情報や矛盾する情報があれば減点する。",
        requires_context=True,
    ),
    MetricSpec(
        "completeness",
        "正解に含まれる要素のうち、回答が触れている要素の割合。"
        "内容の正誤は考慮せず、要素に言及していれば網羅しているとみなす。",
    ),
)

_PROMPT_TEMPLATE = """あなたは厳格な採点者です。次の観点で回答を評価してください。

評価観点（{metric}）: {criterion}

[質問]
{{{{question}}}}

[根拠文書]
{{{{context}}}}

[正解]
{{{{reference}}}}

[回答]
{{{{answer}}}}

採点基準:
- good（score 0.8〜1.0）: 観点を十分に満たす
- partial（score 0.4〜0.7）: 一部満たす
- bad（score 0.0〜0.3）: 満たさない

score・label・reason（日本語で1〜2文の評価理由）をJSONで返してください。"""


def build_judge_prompt(metric: MetricSpec) -> str:
    return _PROMPT_TEMPLATE.format(metric=metric.name, criterion=metric.criterion)


def parse_verdict(raw: Any) -> JudgeVerdict:
    try:
        return JudgeVerdict.model_validate(raw)
    except ValidationError as exc:
        raise JudgeOutputError(f"Judgeの出力が不正です: {raw!r}") from exc


class RubricJudgeEvaluator(LLMEvaluator):
    """1つの評価観点を 0〜1 のスコア・ラベル・理由で判定する Judge。"""

    def __init__(
        self,
        metric: MetricSpec,
        llm: LLM,
        max_attempts: int = 3,
        invocation_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=metric.name,
            llm=llm,
            prompt_template=build_judge_prompt(metric),
            schema=JUDGE_SCHEMA,
            direction="maximize",
            **(invocation_params or {}),
        )
        self.metric = metric
        self.max_attempts = max_attempts

    def _evaluate(self, eval_input: dict[str, Any]) -> list[Score]:
        if self.metric.requires_context and not str(eval_input.get("context", "")).strip():
            return [self._not_applicable_score()]
        prompt = self.prompt_template.render(variables=eval_input)
        return [self._score_from_attempts(prompt)]

    async def _async_evaluate(self, eval_input: dict[str, Any]) -> list[Score]:
        return await asyncio.to_thread(self._evaluate, eval_input)

    def _score_from_attempts(self, prompt: str) -> Score:
        errors: list[str] = []
        for _ in range(self.max_attempts):
            try:
                return self._verdict_to_score(self._request_verdict(prompt), len(errors) + 1)
            except (JudgeOutputError, ValueError) as exc:
                errors.append(str(exc))
        return self._parse_error_score(errors)

    def _request_verdict(self, prompt: str) -> JudgeVerdict:
        raw = self.llm.generate_object(
            prompt=prompt, schema=self.schema, **self.invocation_parameters
        )
        return parse_verdict(raw)

    def _verdict_to_score(self, verdict: JudgeVerdict, attempts: int) -> Score:
        return self._make_score(
            score=verdict.score,
            label=verdict.label,
            explanation=verdict.reason,
            metadata={"model": self.llm.model, "attempts": attempts},
        )

    def _not_applicable_score(self) -> Score:
        return self._make_score(
            score=None, label=NOT_APPLICABLE, explanation="根拠文書がないため評価対象外"
        )

    def _parse_error_score(self, errors: list[str]) -> Score:
        return self._make_score(
            score=None,
            label=PARSE_ERROR,
            explanation=f"{self.max_attempts}回試行してもJudge出力を解釈できませんでした",
            metadata={"errors": errors},
        )

    def _make_score(self, **fields: Any) -> Score:
        return Score(name=self.name, kind=self.kind, direction=self.direction, **fields)


def create_judge_evaluators(
    llm: LLM,
    invocation_params: dict[str, Any] | None = None,
    metrics: tuple[MetricSpec, ...] = METRICS,
) -> list[RubricJudgeEvaluator]:
    """Experiment の `input/expected/output` を Judge の入力にマッピングした Evaluator 群。"""
    return [
        bind_evaluator(
            evaluator=RubricJudgeEvaluator(m, llm, invocation_params=invocation_params),
            input_mapping=JUDGE_INPUT_MAPPING,
        )
        for m in metrics
    ]
