from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.common.llm import LLMClient
from src.demo1_llm_judge import main as demo1_main
from src.demo1_llm_judge.dataset import (
    DatasetFormatError,
    load_samples,
    to_input,
    to_metadata,
    to_output,
)
from src.demo1_llm_judge.generation import (
    PROMPT_VARIANTS,
    AnswerGenerator,
    build_user_prompt,
    make_generation_task,
)

DATASET = Path(__file__).parent.parent / "datasets" / "qa_samples.jsonl"


def test_dataset_has_enough_samples_of_every_type():
    samples = load_samples(DATASET)
    assert len(samples) >= 20
    assert {s.answer_type for s in samples} == {
        "correct",
        "partially_incorrect",
        "irrelevant",
        "incomplete",
        "hallucination",
    }
    assert len({s.id for s in samples}) == len(samples)


def test_faithfulness_label_exists_only_with_context():
    for sample in load_samples(DATASET):
        assert (sample.human_labels.faithfulness is not None) == bool(sample.context)


def test_load_samples_reports_invalid_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="1行目"):
        load_samples(path)


def test_phoenix_record_conversion():
    sample = load_samples(DATASET)[0]
    assert set(to_input(sample)) == {"question", "context"}
    assert set(to_output(sample)) == {"reference", "candidate_answer"}
    assert to_metadata(sample)["human_labels"]["correctness"] == "good"


def test_build_user_prompt_includes_context_only_when_given():
    assert build_user_prompt("Q", "") == "質問: Q"
    assert "根拠文書:\nC" in build_user_prompt("Q", "C")


def test_generation_task_returns_answer():
    client = LLMClient(FakeListChatModel(responses=["回答です"]))
    task = make_generation_task(AnswerGenerator(client, PROMPT_VARIANTS["baseline"]))
    assert task({"question": "Q", "context": ""}) == {"answer": "回答です"}


def test_two_prompt_variants_are_defined():
    assert {"baseline", "detailed"} <= set(PROMPT_VARIANTS)


def test_selected_experiments_all():
    assert demo1_main.selected_experiments("all") == ["baseline", "detailed", "judge-validation"]


def test_main_fails_gracefully_when_vllm_is_down(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    assert demo1_main.main(["--experiment", "baseline"]) == 1
    assert "LLMサーバーに接続できません" in capsys.readouterr().err
