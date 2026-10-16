"""評価データセット（JSONL）の読み込みと Phoenix Dataset への登録。"""

import json
from pathlib import Path
from typing import Literal

from phoenix.client import Client
from phoenix.client.resources.datasets import Dataset
from pydantic import BaseModel, ValidationError

DATASET_NAME = "qa-evaluation-dataset"

Label = Literal["good", "partial", "bad"]
AnswerType = Literal["correct", "partially_incorrect", "irrelevant", "incomplete", "hallucination"]


class DatasetFormatError(ValueError):
    """JSONL の形式が不正な場合に送出する。"""


class HumanLabels(BaseModel):
    correctness: Label
    relevance: Label
    completeness: Label
    faithfulness: Label | None = None


class QASample(BaseModel):
    id: str
    answer_type: AnswerType
    question: str
    reference: str
    context: str = ""
    candidate_answer: str
    human_labels: HumanLabels


def parse_sample(line: str, line_number: int) -> QASample:
    try:
        return QASample.model_validate(json.loads(line))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise DatasetFormatError(f"{line_number}行目の形式が不正です: {exc}") from exc


def load_samples(path: Path) -> list[QASample]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [parse_sample(line, i) for i, line in enumerate(lines, start=1) if line.strip()]


def to_input(sample: QASample) -> dict:
    return {"question": sample.question, "context": sample.context}


def to_output(sample: QASample) -> dict:
    return {"reference": sample.reference, "candidate_answer": sample.candidate_answer}


def to_metadata(sample: QASample) -> dict:
    return {
        "sample_id": sample.id,
        "answer_type": sample.answer_type,
        "human_labels": sample.human_labels.model_dump(),
    }


def find_dataset(client: Client, name: str) -> Dataset | None:
    names = {d["name"] for d in client.datasets.list()}
    return client.datasets.get_dataset(dataset=name) if name in names else None


def upload_dataset(client: Client, samples: list[QASample], name: str = DATASET_NAME) -> Dataset:
    """同名 Dataset があれば再利用し、なければ作成する。"""
    existing = find_dataset(client, name)
    if existing is not None:
        return existing
    return client.datasets.create_dataset(
        name=name,
        dataset_description="LLM as a Judge 用のQA評価データセット（人手の期待ラベル付き）",
        inputs=[to_input(s) for s in samples],
        outputs=[to_output(s) for s in samples],
        metadata=[to_metadata(s) for s in samples],
    )
