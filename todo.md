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
- [ ] `src/common/config.py`：環境変数ロード
- [ ] `src/common/llm.py`：共通LLMクライアント（OpenAI互換・Qwen3 thinking無効化・モック可能）
- [ ] `src/common/phoenix_client.py`：Phoenix Client生成・疎通確認
- [ ] `src/common/tracing.py`：`phoenix.otel.register` によるTracer設定
- [ ] Phoenix起動・vLLM接続・SDK接続を確認

## Phase 2：Demo 1 — LLM as a Judge（`feature/demo1-llm-judge`）

- [ ] QAデータセット20件以上（正解/一部誤り/無関係/不完全/ハルシネーション + 人手期待ラベル）
- [ ] JSONLローダー（Pydantic検証）
- [ ] Qwenによる回答生成（プロンプト2種：baseline / detailed）
- [ ] Judge実装（Correctness / Relevance / Faithfulness / Completeness）
- [ ] JSON Schema検証とリトライ
- [ ] Phoenix Dataset登録（`qa-evaluation-dataset`）
- [ ] Experiment実行（2種類のプロンプト）
- [ ] Judgeと人手ラベルの一致率計算
- [ ] CLI（`--dataset`, `--experiment`）
- [ ] 単体テスト（モックLLM）

## Phase 3：Demo 2 — LangGraph Tracing（`feature/demo2-langgraph`）

- [ ] AgentState定義
- [ ] ツール：Calculator / Knowledge Search（ローカルJSON）/ Current Date
- [ ] Planner / Tool Executor / Answer Generator / Reviewer ノード
- [ ] Retry上限付き条件分岐グラフ
- [ ] シナリオフラグ（normal / retry / error）で決定的に再現
- [ ] OpenInference LangChain Instrumentor 計装 + 手動Span
- [ ] CLI（`--scenario`）
- [ ] 単体テスト（モックLLM・InMemorySpanExporter）

## Phase 4：品質確認（`feature/quality-docs`）

- [ ] pytest全件パス
- [ ] ruff lint
- [ ] vLLM切断時のエラー処理
- [ ] Phoenix切断時のエラー処理
- [ ] Retry上限の検証
- [ ] Traceの親子関係の確認
- [ ] 実機（vLLM + Phoenix）で両デモ実行
- [ ] READMEに起動・停止手順とPhoenix確認ポイントを記載
