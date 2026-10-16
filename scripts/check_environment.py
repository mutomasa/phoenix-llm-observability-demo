"""GPU・Python・主要ライブラリのバージョンを表示する環境確認スクリプト。"""

import importlib.metadata
import platform
import shutil
import subprocess

PACKAGES = [
    "arize-phoenix-client",
    "arize-phoenix-evals",
    "arize-phoenix-otel",
    "langgraph",
    "langchain-openai",
    "openinference-instrumentation-langchain",
    "opentelemetry-sdk",
]


def print_python() -> None:
    print(f"Python: {platform.python_version()}")


def print_packages() -> None:
    for name in PACKAGES:
        try:
            print(f"  {name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {name}: (not installed)")


def print_gpu() -> None:
    if shutil.which("nvidia-smi") is None:
        print("GPU: nvidia-smi not found")
        return
    query = "--query-gpu=name,memory.total,memory.used,compute_cap"
    result = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"GPU: {result.stdout.strip() or result.stderr.strip()}")


def main() -> None:
    print_python()
    print("Packages:")
    print_packages()
    print_gpu()


if __name__ == "__main__":
    main()
