# Phoenix LLM Observability Demo — 開発方針

## 1. プロジェクト概要

### リポジトリ名

`phoenix-llm-observability-demo`

### 目的

Arize Phoenixを使用して、LLMアプリケーションの評価（Evaluation）とAI Agentの実行追跡（Tracing）を学習・検証する2つのデモを開発する。

**デモ1：LLM as a Judge**

小型QwenモデルをLLM Judgeとして使用し、生成回答の品質を評価する。評価結果はArize PhoenixのWebコンソールで可視化する。

**デモ2：LangGraph Tracing**

LangGraphで構築したAI Agentの実行過程をOpenTelemetry / OpenInferenceで追跡し、Arize PhoenixでTrace、Span、Latency、LLM入出力を可視化する。

### 基本方針

- Python + uvで開発する。
- Arize PhoenixはDocker Composeで起動する。
- 可視化はPhoenix標準Webコンソールのみを使用する。
- Streamlit、React、FastAPIなどの独自UIは作成しない。
- LLMはローカルGPUのNVIDIA RTX A1000（VRAM 8GB）で実行する。
- 推論エンジンにはvLLMを使用する。
- LLMの第一候補は`Qwen/Qwen3-1.7B`とする。
- 外部の有料LLM APIは使用しない。
- 2つのデモは独立してCLIから実行可能にする。
- Phoenixにはデモごとに異なるProjectを作成する。
- 依存ライブラリは最新版の互換性を確認し、`uv.lock`に固定する。

---

## 2. 技術スタック

| 分類 | 技術 | 役割 |
|---|---|---|
| Language | Python 3.11+ | アプリケーション |
| Package Manager | uv | 依存関係管理 |
| Observability | Arize Phoenix | 評価・Trace可視化 |
| Container | Docker Compose | Phoenix起動 |
| LLM Serving | vLLM | ローカルLLM推論 |
| LLM | Qwen3-1.7B | 生成・Judge |
| Agent Framework | LangGraph | AI Agent実行 |
| Tracing | OpenTelemetry | 分散トレース |
| Instrumentation | OpenInference | LLM/Agent計装 |
| Evaluation | Phoenix Evals | LLM as a Judge |
| Data | JSONL | 評価データセット |
| Testing | pytest | 自動テスト |

## 3. システムアーキテクチャ

```text
              Developer / CLI
                     |
          +----------+----------+
          |                     |
          v                     v
  Demo 1: Evaluation    Demo 2: LangGraph
  LLM as a Judge        AI Agent
          |                     |
          v                     v
   Phoenix Evals       OpenInference
          |                     |
          +----------+----------+
                     |
                     v
           OpenTelemetry / SDK
                     |
                     v
          +--------------------+
          |   Arize Phoenix    |
          |      Docker        |
          |                    |
          | Datasets           |
          | Experiments        |
          | Evaluation Results |
          | Traces / Spans      |
          +--------------------+
                     |
                     v
             Web Console
          http://localhost:6006

  Both demos use:
          |
          v
    OpenAI-compatible API
          |
          v
    vLLM (localhost:8000)
          |
          v
    Qwen3-1.7B
          |
          v
    NVIDIA RTX A1000 8GB
```

### コンポーネント構成

**Phoenix**

- Docker上で稼働する。
- Web UIは6006番ポートを使用する。
- OTLP HTTPでTraceを受信する。
- データはDocker Volumeに永続化する。

**vLLM**

- ホストOS上で実行する。
- OpenAI互換APIを提供する。
- GPUメモリ消費を抑制する。
- Qwen3-1.7Bが実行できない場合、Qwen3-0.6Bに切り替える。

**Python**

- Phoenix SDKを使用する。
- LLMは共通クライアントを通して呼び出す。
- デモ1とデモ2のコードは分離する。

---

## 4. Demo 1 — LLM as a Judge

### 4.1 目的

LLMによる生成回答を別のLLM呼び出しで評価し、品質を定量的に確認する。

### 4.2 評価対象

シンプルな質問応答アプリケーションを作成する。

質問例：

```json
{
  "question": "Pythonのリストとタプルの違いは？",
  "reference": "リストは変更可能で、タプルは変更不可である。"
}
```

Qwenが回答を生成し、別のJudge呼び出しで回答品質を判定する。

生成用とJudge用は同じモデルを利用するが、プロンプトと呼び出しを分離する。

### 4.3 評価指標

| 評価項目 | 説明 | スコア |
|---|---|---|
| Correctness | 正解との整合性 | 0〜1 |
| Relevance | 質問との関連性 | 0〜1 |
| Faithfulness | 提示した根拠との整合性 | 0〜1 |
| Completeness | 必要情報の網羅性 | 0〜1 |

Faithfulnessは根拠文書を持つサンプルにのみ適用する。

各評価結果には次の情報を含める。

```json
{
  "score": 0.8,
  "label": "good",
  "reason": "主要な概念を正しく説明しているが、例が不足している。"
}
```

Judgeには構造化JSONを要求する。vLLMのStructured Outputs機能とPhoenix Evalsの互換性を検証し、必要に応じてJSON Schema検証とリトライを実装する。

### 4.4 評価データセット

最低20件を用意する。

- 正しい回答
- 一部誤った回答
- 無関係な回答
- 不完全な回答
- ハルシネーションを含む回答

各サンプルには事前に人手の期待評価を設定し、Judgeの判定品質も検証する。

### 4.5 処理フロー

```text
Load Evaluation Dataset
          |
          v
    Generate Answer
          |
          v
    Qwen via vLLM
          |
          v
   LLM Judge (Qwen)
          |
          v
    Evaluation Result
          |
          v
 Phoenix Dataset / Experiment
          |
          v
   Phoenix Web Console
```

### 4.6 Phoenixでの確認項目

Project名：`demo-llm-judge`

Dataset名：`qa-evaluation-dataset`

以下を確認できるようにする。

1. 質問と期待回答
2. Qwenの生成回答
3. Judgeによるスコア
4. Judgeの評価理由
5. サンプルごとの評価結果
6. Experiment間の比較

少なくとも2種類の生成プロンプトでExperimentを実行し、評価スコアを比較できるようにする。

### 4.7 実装要件

- Phoenix ClientでDatasetを登録する。
- Phoenix Evalsの現行SDKでJudgeを実装する。
- vLLMのOpenAI互換APIを利用する。
- 評価結果をPhoenix Experimentに紐付ける。
- CLIで評価を実行できるようにする。
- JSONパース失敗時に適切なエラー処理を行う。
- Judgeの判定と人手ラベルの一致率を計算する。
- Judgeの評価処理も必要に応じてTraceを取得する。

---

## 5. Demo 2 — LangGraph Tracing

### 5.1 目的

LangGraphを使用したAI Agentの実行過程をPhoenixで可視化する。

### 5.2 Agent構成

以下の4ノードを持つシンプルなAgentを実装する。

```text
          START
            |
            v
         Planner
            |
            v
       Tool Executor
            |
            v
      Answer Generator
            |
            v
         Reviewer
            |
      +-----+------+
      |            |
     PASS         RETRY
      |            |
      v            |
     END <---------+
          (retry limit)
```

実際の実装では、ReviewerがRETRYを返した場合、上限回数以内でAnswer Generatorに戻す。上限到達時は失敗状態または最終回答として終了する。

### 5.3 ノードの責務

| Node | 役割 |
|---|---|
| Planner | 質問を解析し処理方針を決定 |
| Tool Executor | Pythonツールを実行 |
| Answer Generator | LLMで回答を生成 |
| Reviewer | 回答を検証しPASS/RETRY判定 |

Tool Executorには次のような安全なローカルツールを実装する。

- Calculator
- Simple Knowledge Search（ローカルJSON）
- Current Date Tool

外部APIは使用しない。

### 5.4 LangGraph State

TypedDictまたはPydanticでStateを定義する。

```python
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    plan: str
    tool_result: str
    answer: str
    review_result: str
    retry_count: int
```

### 5.5 Tracing

OpenInferenceのLangChain Instrumentorを使用する。

必要に応じてOpenTelemetryによるカスタムSpanも追加する。

取得する情報：

| 情報 | 内容 |
|---|---|
| Trace ID | 実行単位の識別子 |
| Span ID | 処理単位の識別子 |
| Parent Span | 親子関係 |
| Node Name | 実行ノード |
| Input | 入力 |
| Output | 出力 |
| Latency | 処理時間 |
| Error | エラー |
| Token Usage | 取得できる場合のトークン数 |

OpenInferenceによる自動計装で取得できない独自処理は、手動Spanを追加する。

### 5.6 Phoenixでの確認項目

Project名：`demo-langgraph-tracing`

以下をPhoenix Webコンソールで確認する。

1. LangGraphの実行Trace
2. 各ノードのSpan
3. LLM呼び出しの入力・出力
4. ツール呼び出し結果
5. 各SpanのLatency
6. 親子Spanの階層
7. ReviewerによるRetry
8. エラー発生時のSpan

### 5.7 比較シナリオ

以下の3ケースを実行する。

**Case A：正常実行**

Planner → Tool → Answer → Reviewer → END

**Case B：Retry**

Planner → Tool → Answer → Reviewer → Answer → Reviewer → END

**Case C：Tool Error**

Tool Executorが例外を発生させ、Phoenixでエラーを確認する。

RetryやErrorはテスト用フラグで再現可能にし、デモを決定的に実行できるようにする。

---

## 6. ディレクトリ構成

```text
phoenix-llm-observability-demo/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── docker-compose.yml
│
├── docs/
│   └── development.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── llm.py
│   │   ├── phoenix_client.py
│   │   └── tracing.py
│   │
│   ├── demo1_llm_judge/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── dataset.py
│   │   ├── generation.py
│   │   ├── evaluators.py
│   │   └── experiments.py
│   │
│   └── demo2_langgraph/
│       ├── __init__.py
│       ├── main.py
│       ├── graph.py
│       ├── state.py
│       ├── nodes.py
│       └── tools.py
│
├── datasets/
│   └── qa_samples.jsonl
│
├── scripts/
│   ├── check_environment.py
│   ├── start_vllm.sh
│   └── check_services.sh
│
└── tests/
    ├── test_evaluators.py
    ├── test_graph.py
    └── test_tracing.py
```

---

## 7. 環境構築

### 7.1 Phoenix Docker

`docker-compose.yml`

```yaml
services:
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "127.0.0.1:6006:6006"
    environment:
      PHOENIX_WORKING_DIR: /mnt/data
    volumes:
      - phoenix_data:/mnt/data
    restart: unless-stopped

volumes:
  phoenix_data:
```

開始：

```bash
docker compose up -d
```

Webコンソール：

`http://localhost:6006`

開発時はバージョン検証後にDockerイメージを固定する。

### 7.2 vLLM

モデル：

`Qwen/Qwen3-1.7B`

起動コマンド：

```bash
uv run vllm serve Qwen/Qwen3-1.7B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096
```

RTX A1000のCUDA Compute Capability、vLLMのサポート状況、CUDA環境を事前確認する。

GPUメモリ不足の場合はコンテキスト長を削減し、必要ならQwen3-0.6Bへ切り替える。

### 7.3 Python

```bash
uv init
uv add arize-phoenix phoenix-client
uv add arize-phoenix-evals
uv add langgraph langchain-openai
uv add openinference-instrumentation-langchain
uv add opentelemetry-sdk
uv add opentelemetry-exporter-otlp
uv add python-dotenv pydantic
uv add --dev pytest ruff
```

依存ライブラリ名・SDKのAPIは実装時点の公式ドキュメントで確認すること。

### 7.4 環境変数

`.env.example`

```dotenv
PHOENIX_BASE_URL=http://localhost:6006
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces

VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
LLM_MODEL=Qwen/Qwen3-1.7B

LLM_TEMPERATURE=0
LLM_MAX_TOKENS=512

JUDGE_PROJECT_NAME=demo-llm-judge
TRACING_PROJECT_NAME=demo-langgraph-tracing
```

実際のPhoenix SDK設定では、管理API用URLとOTLP送信先を区別する。

---

## 8. CLI仕様

### Demo 1

```bash
uv run python -m src.demo1_llm_judge.main
```

任意で次のオプションを実装する。

```bash
uv run python -m src.demo1_llm_judge.main \
  --dataset datasets/qa_samples.jsonl \
  --experiment baseline
```

### Demo 2

```bash
uv run python -m src.demo2_langgraph.main \
  --scenario normal
```

```bash
uv run python -m src.demo2_langgraph.main \
  --scenario retry
```

```bash
uv run python -m src.demo2_langgraph.main \
  --scenario error
```

---

## 9. Claude Code開発手順

### Phase 1：基盤構築

- [ ] uvでプロジェクトを初期化
- [ ] ディレクトリを作成
- [ ] Docker ComposeでPhoenixを起動
- [ ] vLLM + Qwenを起動
- [ ] LLM API接続を確認
- [ ] Phoenix SDKの接続を確認

### Phase 2：Demo 1

- [ ] QAデータセット20件を作成
- [ ] Qwenによる回答生成
- [ ] LLM Judgeを実装
- [ ] JSONスコアの検証
- [ ] Phoenix Dataset登録
- [ ] Experiment実行
- [ ] Phoenixで評価結果を確認

### Phase 3：Demo 2

- [ ] LangGraph State定義
- [ ] Planner実装
- [ ] Tool Executor実装
- [ ] Answer Generator実装
- [ ] Reviewer実装
- [ ] OpenInference計装
- [ ] PhoenixへのTrace送信
- [ ] 正常・Retry・Errorの3ケースを検証

### Phase 4：品質確認

- [ ] pytestテスト
- [ ] ruffによるLint
- [ ] vLLM切断時のエラー処理
- [ ] Phoenix切断時のエラー処理
- [ ] Retry上限の検証
- [ ] Traceの親子関係の確認
- [ ] READMEに起動・停止手順を記載

---

## 10. 完了条件

### Demo 1

- 20件以上の評価データで実行できる。
- 生成LLMとJudge LLMがローカルQwenで動作する。
- PhoenixのDatasetとExperimentに結果が登録される。
- 各質問のスコアと評価理由がPhoenix上で確認できる。
- 2つのExperimentを比較できる。

### Demo 2

- LangGraphの4ノードが実行できる。
- PhoenixでTraceの階層が表示される。
- LLM呼び出しとTool実行が区別できる。
- Retry時の複数Spanが確認できる。
- Error時に失敗Spanが記録される。
- 各Spanの処理時間を確認できる。

### 共通

- RTX A1000 8GBでLLMを実行する。
- PhoenixはDockerで稼働する。
- 評価・可視化に独自Web UIを使用しない。
- READMEにデモ操作方法を記載する。
- 外部の商用LLM APIに依存しない。

---

## 11. Claude Codeへの指示

この`development.md`に従ってプロジェクトを実装してください。

開発時には以下を守ってください。

1. 最初にPhoenix Evals、Phoenix Client、OpenInference、LangGraphの現在の公式APIを調査してください。
2. 古いAPIを使用せず、互換性のあるライブラリの組み合わせを選択してください。
3. `uv`で依存関係を管理してください。
4. 共通のLLMクライアントを用意し、両デモから再利用してください。
5. Phoenixでの可視化を最優先にしてください。
6. 不要な独自Web UIは作成しないでください。
7. GPUがない環境でもモックを使って単体テストできる設計にしてください。
8. 外部商用LLMは使用しないでください。
9. 公式ドキュメントで確認できないAPIを推測して実装しないでください。
10. 実装完了後、各デモの実行手順とPhoenix上での確認ポイントをREADMEに記載してください。

**最終ゴール：ローカルのQwen + vLLMを利用して、LLM EvaluationとLangGraph Tracingという2つの主要機能をArize PhoenixのWebコンソールだけで体験できるデモを完成させる。**
