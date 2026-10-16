# TODO — Phoenix LLM Observability Demo

`docs/development.md` に基づく開発タスク。各フェーズを機能ブランチで開発し、PRでmainへマージする。
問題・制約は GitHub Issues に記録する。

## Phase 0：調査（main）

- [x] Phoenix Evals / Phoenix Client / OpenInference / LangGraph の現行APIを公式ドキュメントで確認
- [x] ローカルGPU・vLLM稼働状況の確認（RTX A1000 8GB, CC 8.6, Qwen3-1.7Bが既に:8000で稼働）
- [x] todo.md 作成

## Phase 1：基盤構築（`feature/project-setup`, `feature/common`）

- [x] uvでプロジェクト初期化・依存追加・`uv.lock`固定
- [x] ディレクトリ構成作成
- [x] `docker-compose.yml`（Phoenixイメージのバージョン固定）
- [x] `.env.example` / `.gitignore`
- [x] `scripts/start_vllm.sh` / `scripts/check_services.sh` / `scripts/check_environment.py`
- [x] `src/common/config.py`：環境変数ロード
- [x] `src/common/llm.py`：共通LLMクライアント（OpenAI互換・Qwen3 thinking無効化・モック可能）
- [x] `src/common/phoenix_client.py`：Phoenix Client生成・疎通確認
- [x] `src/common/tracing.py`：`phoenix.otel.register` によるTracer設定
- [x] Phoenix起動・vLLM接続・SDK接続を確認

## Phase 2：Demo 1 — LLM as a Judge（`feature/demo1-llm-judge`）

- [x] QAデータセット20件以上（正解/一部誤り/無関係/不完全/ハルシネーション + 人手期待ラベル）
- [x] JSONLローダー（Pydantic検証）
- [x] Qwenによる回答生成（プロンプト2種：baseline / detailed）
- [x] Judge実装（Correctness / Relevance / Faithfulness / Completeness）
- [x] JSON Schema検証とリトライ
- [x] Phoenix Dataset登録（`qa-evaluation-dataset`）
- [x] Experiment実行（2種類のプロンプト）
- [x] Judgeと人手ラベルの一致率計算
- [x] CLI（`--dataset`, `--experiment`）
- [x] 単体テスト（モックLLM）

## Phase 3：Demo 2 — LangGraph Tracing（`feature/demo2-langgraph`）

- [x] AgentState定義
- [x] ツール：Calculator / Knowledge Search（ローカルJSON）/ Current Date
- [x] Planner / Tool Executor / Answer Generator / Reviewer ノード
- [x] Retry上限付き条件分岐グラフ
- [x] シナリオフラグ（normal / retry / error）で決定的に再現
- [x] OpenInference LangChain Instrumentor 計装 + 手動Span
- [x] CLI（`--scenario`）
- [x] 単体テスト（モックLLM・InMemorySpanExporter）

## Phase 4：品質確認（`feature/quality-docs`）

- [x] pytest全件パス
- [x] ruff lint
- [x] vLLM切断時のエラー処理
- [x] Phoenix切断時のエラー処理
- [x] Retry上限の検証
- [x] Traceの親子関係の確認
- [x] 実機（vLLM + Phoenix）で両デモ実行
- [x] READMEに起動・停止手順とPhoenix確認ポイントを記載

## 未解決（Issues）

- [ ] #3 Qwen3-1.7B Judge と人手ラベルの一致率改善
- [ ] #4 Experiment の Trace を `demo-llm-judge` Project でも見られるようにする
- [ ] #5 `scripts/start_vllm.sh` 単独起動での再検証（現在は他プロジェクトの vLLM を共有）
