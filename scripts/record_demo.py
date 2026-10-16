"""Phoenix Webコンソールのデモ操作を録画し、README用のGIFを作成する。

事前に両デモを実行しておくこと（Trace と Experiment が必要）。

    uv run --with playwright python scripts/record_demo.py
    # → docs/assets/demo.gif（Playwrightで録画した動画をffmpegで変換）
"""

import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

B = os.environ.get("PHOENIX_BASE_URL", "http://localhost:6006")
OUTPUT_DIR = Path("docs/assets")
W, H = 1280, 800


def project_id(name: str) -> str:
    projects = httpx.get(f"{B}/v1/projects").json()["data"]
    return next(p["id"] for p in projects if p["name"] == name)


def dataset_id(name: str) -> str:
    datasets = httpx.get(f"{B}/v1/datasets").json()["data"]
    return next(d["id"] for d in datasets if d["name"] == name)


def latest_experiment_id(dataset: str, name: str) -> str:
    experiments = httpx.get(f"{B}/v1/datasets/{dataset}/experiments").json()["data"]
    matched = [
        e for e in experiments if (e.get("metadata") or {}).get("name", e.get("name")) == name
    ]
    return max(matched, key=lambda e: e["created_at"])["id"]


def build_urls() -> dict[str, str]:
    ds = dataset_id("qa-evaluation-dataset")
    exp = {n: latest_experiment_id(ds, n) for n in ["baseline", "detailed", "judge-validation"]}
    compare = f"{B}/datasets/{ds}/compare"
    return {
        "traces": f"{B}/projects/{project_id('demo-langgraph-tracing')}/traces",
        "compare": f"{compare}?experimentId={exp['baseline']}&experimentId={exp['detailed']}",
        "validation": f"{compare}?experimentId={exp['judge-validation']}",
    }


CAPTION_JS = """t => {
  let el = document.getElementById('__cap');
  if (!el) { el = document.createElement('div'); el.id='__cap'; document.body.appendChild(el); }
  el.textContent = t;
  Object.assign(el.style, {position:'fixed', left:'50%', bottom:'28px',
    transform:'translateX(-50%)',
    background:'rgba(20,20,30,.92)', color:'#fff', padding:'12px 26px', borderRadius:'10px',
    font:'600 22px sans-serif', zIndex:99999, border:'2px solid #7c5cff', whiteSpace:'nowrap',
    pointerEvents:'none'});
}"""


def caption(pg, text, wait=0):
    pg.evaluate(CAPTION_JS, text)
    pg.wait_for_timeout(wait)


def click(pg, locator, wait=1800):
    locator.scroll_into_view_if_needed()
    locator.evaluate("e => { e.style.outline='3px solid #ffd400'; e.style.outlineOffset='2px'; }")
    pg.wait_for_timeout(700)
    locator.click()
    pg.mouse.move(W - 4, H - 4)
    pg.wait_for_timeout(wait)


def goto(pg, url, text, wait=2200):
    pg.goto(url)
    pg.wait_for_timeout(1800)
    caption(pg, text, wait)


def demo2(pg, urls):
    goto(
        pg, urls["traces"], "Demo 2: LangGraph Agent の Trace 一覧（normal / retry / error）", 3000
    )
    row = (
        pg.get_by_role("row")
        .filter(has_text="Arize Ph")
        .first.get_by_role("link", name="LangGraph")
    )
    click(pg, row, 1500)
    caption(pg, "retry シナリオ: ノード階層と Reviewer による再生成", 3500)
    click(pg, pg.get_by_text("tool.knowledge_search"), 300)
    caption(pg, "TOOL Span: ツールの入力と結果", 2800)
    click(pg, pg.get_by_text("ChatOpenAI").nth(1), 300)
    caption(pg, "LLM Span: Qwen3-1.7B への入力メッセージと出力・トークン数", 3500)

    goto(pg, urls["traces"], "error シナリオの Trace を開く", 1200)
    row = (
        pg.get_by_role("row")
        .filter(has_text="今日の日")
        .first.get_by_role("link", name="LangGraph")
    )
    click(pg, row, 1500)
    click(pg, pg.get_by_text("tool.current_date"), 300)
    caption(pg, "Tool の例外が ERROR Span として親まで伝播", 2200)
    click(pg, pg.get_by_role("tab", name="Events"), 300)
    caption(pg, "Events に例外とスタックトレースが記録される", 3000)


def demo1(pg, urls):
    goto(
        pg,
        urls["compare"],
        "Demo 1: LLM as a Judge — baseline vs detailed の Experiment 比較",
        3500,
    )
    pg.mouse.move(W * 0.6, H * 0.75)
    for _ in range(3):
        pg.mouse.wheel(0, 110)
        pg.wait_for_timeout(400)
    pg.mouse.move(W - 4, H - 4)
    caption(pg, "サンプルごとの回答と Judge スコア（correctness / relevance / ...）", 2500)
    pg.mouse.move(626, 606)
    caption(pg, "スコアにホバーすると Judge の評価理由（explanation）を表示", 4000)
    pg.mouse.move(W - 4, H - 4)
    click(pg, pg.get_by_text("Metrics", exact=True), 300)
    caption(pg, "評価指標の平均を Experiment 間で比較", 3500)
    goto(
        pg, urls["validation"], "judge-validation: Judge と人手ラベルの一致率（agreement_*）", 4000
    )


def record(video_dir: str) -> Path:
    urls = build_urls()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=video_dir,
            record_video_size={"width": W, "height": H},
        )
        pg = ctx.new_page()
        demo2(pg, urls)
        demo1(pg, urls)
        ctx.close()
        browser.close()
    return next(Path(video_dir).glob("*.webm"))


def to_gif(video: Path, gif: Path, width: int = 960, fps: int = 5) -> None:
    """2パス（パレット生成→適用）でGIFに変換する。冒頭の読み込み中の画面は除く。"""
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    palette = video.with_suffix(".png")
    ffmpeg = ["ffmpeg", "-v", "error", "-y", "-ss", "1.8", "-i", str(video)]
    subprocess.run(
        [*ffmpeg, "-vf", f"{filters},palettegen=max_colors=128:stats_mode=diff", str(palette)],
        check=True,
    )
    subprocess.run(
        [
            *ffmpeg,
            "-i",
            str(palette),
            "-lavfi",
            f"{filters} [x]; [x][1:v] paletteuse=dither=none:diff_mode=rectangle",
            str(gif),
        ],
        check=True,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        video = record(tmp)
        to_gif(video, OUTPUT_DIR / "demo.gif")
    print(f"saved: {OUTPUT_DIR / 'demo.gif'}")


if __name__ == "__main__":
    main()
