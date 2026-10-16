"""Phoenix 管理API（Datasets / Experiments）用クライアント。"""

import httpx
from phoenix.client import Client

from src.common.config import PhoenixSettings


class PhoenixUnavailableError(RuntimeError):
    """Phoenix サーバーに接続できない場合に送出する。"""


def check_phoenix_health(base_url: str, timeout: float = 5.0) -> None:
    url = f"{base_url.rstrip('/')}/healthz"
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise PhoenixUnavailableError(
            f"Phoenixに接続できません（{url}）。`docker compose up -d` を確認してください: {exc}"
        ) from exc


def create_phoenix_client(settings: PhoenixSettings) -> Client:
    check_phoenix_health(settings.base_url)
    return Client(base_url=settings.base_url)
