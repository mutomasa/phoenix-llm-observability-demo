"""vLLM / Phoenix 切断時に CLI が分かりやすいエラーで終了することを確認する。"""

import pytest

from src.demo1_llm_judge import main as demo1_main
from src.demo2_langgraph import main as demo2_main

UNREACHABLE = "http://127.0.0.1:9"


@pytest.fixture
def vllm_up(monkeypatch):
    monkeypatch.setattr(demo1_main, "check_llm_health", lambda url: None)
    monkeypatch.setattr(demo2_main, "check_llm_health", lambda url: None)


def test_demo1_reports_phoenix_down(monkeypatch, capsys, vllm_up):
    monkeypatch.setenv("PHOENIX_BASE_URL", UNREACHABLE)
    assert demo1_main.main(["--experiment", "baseline"]) == 1
    assert "Phoenixに接続できません" in capsys.readouterr().err


def test_demo2_reports_vllm_down(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_BASE_URL", f"{UNREACHABLE}/v1")
    assert demo2_main.main(["--scenario", "normal"]) == 1
    assert "LLMサーバーに接続できません" in capsys.readouterr().err


def test_demo2_reports_phoenix_down(monkeypatch, capsys, vllm_up):
    monkeypatch.setenv("PHOENIX_BASE_URL", UNREACHABLE)
    assert demo2_main.main(["--scenario", "normal"]) == 1
    assert "Phoenixに接続できません" in capsys.readouterr().err


def test_demo1_reports_invalid_dataset(monkeypatch, capsys, tmp_path, vllm_up):
    broken = tmp_path / "broken.jsonl"
    broken.write_text("not json\n", encoding="utf-8")
    monkeypatch.setattr(demo1_main, "create_phoenix_client", lambda settings: object())
    assert demo1_main.main(["--dataset", str(broken), "--experiment", "baseline"]) == 1
    assert "1行目" in capsys.readouterr().err
