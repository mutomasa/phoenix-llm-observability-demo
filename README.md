# phoenix-llm-observability-demo

Arize Phoenix を使って、ローカルの Qwen3 + vLLM で次の2つを体験するデモです。

| デモ | 内容 | Phoenix での確認先 |
|---|---|---|
| Demo 1：LLM as a Judge | Qwen が生成した回答を、別の Qwen 呼び出し（Judge）で 4 観点採点する | Datasets → `qa-evaluation-dataset` → Experiments |
| Demo 2：LangGraph Tracing | 4ノードの LangGraph Agent の実行を OpenInference で追跡する | Projects → `demo-langgraph-tracing` |

可視化には Phoenix 標準の Web コンソールだけを使います。外部の商用 LLM API は使いません。
開発方針は [docs/development.md](docs/development.md)、進捗は [todo.md](todo.md) を参照してください。

## 動作環境

| 項目 | 検証時のバージョン |
|---|---|
| GPU | NVIDIA RTX A1000 8GB（Compute Capability 8.6） |
| Python | 3.12 以上（uv で管理） |
| Phoenix（Docker） | `arizephoenix/phoenix:20.23.0` |
| LLM | `Qwen/Qwen3-1.7B`（vLLM、OpenAI 互換 API） |
| 主なライブラリ | arize-phoenix-client 3.5.0 / arize-phoenix-evals 3.9.2 / arize-phoenix-otel 0.17.2 / langgraph 1.2.15 / langchain-openai 1.7.2 / openinference-instrumentation-langchain 0.1.79 |

## セットアップ

```bash
uv sync                     # 依存関係をインストール（uv.lock に固定済み）
cp .env.example .env        # 必要に応じて編集
uv run python scripts/check_environment.py   # GPU とライブラリのバージョンを確認
```

## 起動と停止

### 1. Phoenix

```bash
docker compose up -d        # 起動 → http://localhost:6006
docker compose down         # 停止（データは phoenix_data ボリュームに残る）
docker compose down -v      # 停止してデータも削除
```

### 2. vLLM + Qwen3

```bash
./scripts/start_vllm.sh     # 既に :8000 で起動済みなら、それを再利用する
```

- vLLM は CUDA に依存する大きなパッケージなので、本プロジェクトの依存関係には入れず `uvx` で実行します。
- GPU メモリが足りない場合は `VLLM_MAX_MODEL_LEN=2048` を指定するか、`LLM_MODEL=Qwen/Qwen3-0.6B` に切り替えてください。
- Qwen3 の思考モード（`<think>`）は、リクエストの `chat_template_kwargs.enable_thinking=false` で無効にしています（`LLM_ENABLE_THINKING`）。
- 停止するときは、起動したターミナルで `Ctrl+C` を押します。

### 3. 疎通確認

```bash
./scripts/check_services.sh
# [OK]   Phoenix: http://localhost:6006/healthz
# [OK]   vLLM: http://localhost:8000/v1/models
```

---

## Demo 1：LLM as a Judge

```bash
uv run python -m src.demo1_llm_judge.main                     # 全 Experiment（約8分）
uv run python -m src.demo1_llm_judge.main --experiment baseline
uv run python -m src.demo1_llm_judge.main \
  --dataset datasets/qa_samples.jsonl --experiment judge-validation
```

| `--experiment` | タスク | 評価 |
|---|---|---|
| `baseline` | 一文で答えるだけのプロンプトで回答を生成する | Judge × 4 観点 |
| `detailed` | 根拠文書を守り、漏れなく答えるよう指示したプロンプトで回答を生成する | Judge × 4 観点 |
| `judge-validation` | 人手で作った回答を Judge で判定する | Judge のラベルと人手ラベルの一致（`agreement_*`） |
| `all`（既定） | 上の3つを順に実行する | |

### 仕組み

- **データセット**（`datasets/qa_samples.jsonl`）：25 件。内訳は、正しい回答・一部誤った回答・無関係な回答・不完全な回答・ハルシネーションを含む回答が各 5 件です。根拠文書付きは 11 件で、全件に人手の期待ラベル（good / partial / bad）を付けています。
- **Judge**（`src/demo1_llm_judge/evaluators.py`）：Phoenix Evals の `LLMEvaluator` を拡張しています。
  - 4 観点：Correctness / Relevance / Faithfulness / Completeness。Faithfulness は根拠文書があるサンプルだけを評価し、ないサンプルは `not_applicable` とします。
  - vLLM の Structured Outputs（JSON Schema）で `{score: 0〜1, label, reason}` を出力させます。
  - 出力は Pydantic で検証し、不正なら最大 3 回まで再試行します。それでも解釈できなければ `parse_error`（score は空）を記録します。

### Phoenix での確認ポイント

`http://localhost:6006` → **Datasets** → `qa-evaluation-dataset`

1. **Examples** タブ：質問（`input.question` / `input.context`）と期待回答（`output.reference`）、人手ラベル（`metadata.human_labels`）。
2. **Experiments** タブ：`baseline` と `detailed` を選んで **Compare** を開くと、サンプルごとに回答と 4 観点のスコアを横に並べて比較できます。
3. 各セルのスコアにカーソルを合わせると、Judge の評価理由（`explanation`）が表示されます。
4. 各 Run の **Trace** を開くと、`Task: generate_answer` の下に `ChatOpenAI`（LLM の入出力とトークン数）が見えます。Judge の評価にも Trace があります。
5. `judge-validation` では `agreement_*` 列で、Judge と人手ラベルの一致・不一致をサンプルごとに確認できます。

> Experiment 中の Trace は、Phoenix が Experiment ごとに作る専用の Project に記録されます。そのため Projects 一覧の `demo-llm-judge` には表示されません（#4）。

### 実行結果の例（Qwen3-1.7B）

| 観点 | baseline | detailed |
|---|---|---|
| correctness | 0.684 | **0.748** |
| relevance | 0.764 | **0.784** |
| faithfulness | **0.873** | 0.773 |
| completeness | 0.704 | **0.744** |

`judge-validation` での一致率は、correctness 0.72、relevance 0.56、faithfulness 0.73、completeness 0.64、全体 0.66 でした。小型モデルの Judge は観点の切り分けが苦手です（#3）。

---

## Demo 2：LangGraph Tracing

```bash
uv run python -m src.demo2_langgraph.main --scenario normal   # Case A
uv run python -m src.demo2_langgraph.main --scenario retry    # Case B
uv run python -m src.demo2_langgraph.main --scenario error    # Case C（終了コード 2）
uv run python -m src.demo2_langgraph.main --scenario normal --question "LangGraphとは？"
```

```text
START → planner → tool_executor → answer_generator → reviewer ─┬─ PASS → END
                                         ↑                     │
                                         └──── RETRY ──────────┘（上限 --max-retries を超えたら status=max_retries_exceeded で END）
```

| ノード | 役割 |
|---|---|
| planner | LLM が使うツールとその入力を JSON で決める |
| tool_executor | `calculator`（AST で評価し `eval` は使わない）/ `knowledge_search`（`datasets/knowledge.json`）/ `current_date` を実行する |
| answer_generator | ツールの結果とレビューの指摘をもとに、LLM で回答を生成する |
| reviewer | LLM で回答をレビューする。シナリオのフラグで RETRY を決定的に再現する |

| シナリオ | 質問 | 期待するフロー |
|---|---|---|
| `normal` | 128 × 46 はいくつですか？ | Planner → Tool → Answer → Reviewer → END |
| `retry` | Arize Phoenixとは何ですか？ | … → Answer → Reviewer(RETRY) → Answer → Reviewer → END |
| `error` | 今日の日付を教えてください。 | `current_date` で例外が発生する |

### Phoenix での確認ポイント

`http://localhost:6006` → **Projects** → `demo-langgraph-tracing`

1. **Traces** タブ：1 回の実行が `LangGraph` をルートとする 1 本の Trace になります。
2. 各ノードの Span：`planner` / `tool_executor` / `answer_generator` / `reviewer`（CHAIN）。
3. LLM 呼び出し：ノードの下にある `ChatOpenAI`（LLM）で、入力メッセージ、出力、トークン数を確認できます。
4. ツール呼び出し：`tool.<name>`（TOOL）で、入力と結果を確認できます。LLM の Span とは種別で区別できます。
5. Latency：Trace の一覧と各 Span で処理時間を確認できます。
6. 親子関係：`LangGraph → tool_executor → tool.calculator` のような階層で表示されます。
7. Retry：`retry` シナリオでは `answer_generator` と `reviewer` が 2 回ずつ並びます。
8. エラー：`error` シナリオでは `tool.current_date` → `tool_executor` → `LangGraph` が ERROR になり、例外のイベントが記録されます。
9. メタデータ `scenario` とタグ `scenario:<name>` でフィルタできます。

---

## 開発

```bash
uv run pytest               # 単体テスト（GPU・Phoenix 不要。LLM はフェイクモデルで置き換える）
uv run ruff check .         # Lint
uv run ruff format .        # フォーマット
```

```text
src/
├── common/            # 設定・共通LLMクライアント・Phoenix Client・トレーシング
├── demo1_llm_judge/   # dataset / generation / evaluators / agreement / experiments / main
└── demo2_langgraph/   # state / tools / nodes / graph / scenarios / main
datasets/              # qa_samples.jsonl（Demo 1）・knowledge.json（Demo 2）
scripts/               # start_vllm.sh / check_services.sh / check_environment.py
tests/
```

### エラー処理

| 状況 | 挙動 |
|---|---|
| vLLM に接続できない | `[ERROR] LLMサーバーに接続できません …` を表示して終了コード 1 |
| Phoenix に接続できない | `[ERROR] Phoenixに接続できません …` を表示して終了コード 1 |
| JSONL の形式が不正 | 該当する行番号を表示して終了コード 1 |
| Judge の出力が不正 | 最大 3 回まで再試行し、だめなら `parse_error` として記録 |
| ツールの例外（Demo 2） | エラーの Span を Phoenix に送ってから終了コード 2 |

## 既知の課題

[Issues](https://github.com/mutomasa/phoenix-llm-observability-demo/issues) を参照してください。
