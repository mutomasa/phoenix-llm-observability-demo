import pytest

from src.demo1_llm_judge.agreement import (
    compare_labels,
    judge_candidate,
    make_agreement_evaluator,
    overall_agreement,
)
from src.demo1_llm_judge.evaluators import (
    METRICS,
    NOT_APPLICABLE,
    PARSE_ERROR,
    JudgeOutputError,
    RubricJudgeEvaluator,
    build_judge_prompt,
    create_judge_evaluators,
    parse_verdict,
)

GOOD = {"score": 0.9, "label": "good", "reason": "正しい"}
CORRECTNESS = METRICS[0]
FAITHFULNESS = next(m for m in METRICS if m.requires_context)


def prompt_text(prompt):
    return "\n".join(m["content"] for m in prompt)


def experiment_input(context=""):
    return {
        "input": {"question": "Q?", "context": context},
        "expected": {"reference": "R", "candidate_answer": "C"},
        "output": {"answer": "A"},
    }


def test_parse_verdict_accepts_valid_json():
    verdict = parse_verdict(GOOD)
    assert (verdict.score, verdict.label) == (0.9, "good")


@pytest.mark.parametrize(
    "raw",
    [
        {"score": 1.5, "label": "good", "reason": "x"},
        {"score": 0.5, "label": "excellent", "reason": "x"},
        {"score": 0.5, "label": "good"},
        "not json",
    ],
)
def test_parse_verdict_rejects_invalid_output(raw):
    with pytest.raises(JudgeOutputError):
        parse_verdict(raw)


def test_build_judge_prompt_has_template_variables():
    prompt = build_judge_prompt(CORRECTNESS)
    for var in ["{{question}}", "{{context}}", "{{reference}}", "{{answer}}"]:
        assert var in prompt


def test_judge_returns_score_label_reason(make_judge_llm):
    judge = create_judge_evaluators(make_judge_llm([GOOD]), metrics=(CORRECTNESS,))[0]
    score = judge.evaluate(experiment_input())[0]
    assert (score.name, score.score, score.label, score.explanation) == (
        "correctness",
        0.9,
        "good",
        "正しい",
    )
    assert score.kind == "llm"


def test_judge_maps_experiment_fields_into_prompt(make_judge_llm):
    llm = make_judge_llm([GOOD])
    judge = create_judge_evaluators(llm, metrics=(CORRECTNESS,))[0]
    judge.evaluate(experiment_input())
    text = prompt_text(llm.prompts[0])
    assert "Q?" in text and "[回答]\nA" in text


def test_judge_passes_invocation_params(make_judge_llm):
    llm = make_judge_llm([GOOD])
    judge = RubricJudgeEvaluator(CORRECTNESS, llm, invocation_params={"temperature": 0})
    judge.evaluate({"question": "q", "context": "", "reference": "r", "answer": "a"})
    assert llm.kwargs[0] == {"temperature": 0}


def test_judge_retries_after_invalid_output(make_judge_llm):
    llm = make_judge_llm([{"label": "good"}, ValueError("bad json"), GOOD])
    judge = create_judge_evaluators(llm, metrics=(CORRECTNESS,))[0]
    score = judge.evaluate(experiment_input())[0]
    assert score.label == "good"
    assert score.metadata["attempts"] == 3


def test_judge_returns_parse_error_after_max_attempts(make_judge_llm):
    llm = make_judge_llm([{"x": 1}] * 3)
    judge = create_judge_evaluators(llm, metrics=(CORRECTNESS,))[0]
    score = judge.evaluate(experiment_input())[0]
    assert score.label == PARSE_ERROR
    assert score.score is None
    assert len(score.metadata["errors"]) == 3


def test_faithfulness_is_not_applicable_without_context(make_judge_llm):
    llm = make_judge_llm([])
    judge = create_judge_evaluators(llm, metrics=(FAITHFULNESS,))[0]
    score = judge.evaluate(experiment_input(context=""))[0]
    assert score.label == NOT_APPLICABLE
    assert llm.prompts == []


def test_faithfulness_is_judged_with_context(make_judge_llm):
    judge = create_judge_evaluators(make_judge_llm([GOOD]), metrics=(FAITHFULNESS,))[0]
    assert judge.evaluate(experiment_input(context="根拠"))[0].label == "good"


def test_judge_candidate_evaluates_candidate_answer(make_judge_llm):
    llm = make_judge_llm([GOOD])
    judges = create_judge_evaluators(llm, metrics=(CORRECTNESS,))
    result = judge_candidate(
        judges, {"question": "Q", "context": ""}, {"reference": "R", "candidate_answer": "CAND"}
    )
    assert result == {"correctness": {"score": 0.9, "label": "good", "reason": "正しい"}}
    assert "[回答]\nCAND" in prompt_text(llm.prompts[0])


@pytest.mark.parametrize(
    ("judge", "human", "expected"),
    [
        ("good", "good", 1.0),
        ("bad", "good", 0.0),
        ("good", None, None),
        (NOT_APPLICABLE, "good", None),
    ],
)
def test_compare_labels(judge, human, expected):
    assert compare_labels(judge, human)["score"] == expected


def test_agreement_evaluator_reads_output_and_metadata():
    evaluator = make_agreement_evaluator("correctness")
    result = evaluator.evaluate(
        output={"correctness": {"label": "partial"}},
        metadata={"human_labels": {"correctness": "partial"}},
    )
    assert evaluator.name == "agreement_correctness"
    assert result["label"] == "match"


def test_overall_agreement_averages_agreement_metrics_only():
    assert overall_agreement({"agreement_a": 1.0, "agreement_b": 0.5, "correctness": 0.0}) == 0.75
    assert overall_agreement({"correctness": 0.3}) is None
