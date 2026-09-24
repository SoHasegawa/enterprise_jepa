from collections.abc import Mapping
from typing import Any

from a2a.types import FileWithBytes, FileWithUri

from common.client_utils import send_message_with_files
from common.purple_protocol import build_purple_request_text


class PurpleAgentError(RuntimeError):
    """Purple Agent から非 completed 応答が返ったときの詳細付き例外。"""

    def __init__(self, url: str, outputs: dict[str, Any]) -> None:
        self.url = url
        self.outputs = outputs
        super().__init__(f"{url} responded with: {outputs}")


class PurpleClient:
    """Purple Agent との会話状態を保持しながら通信する。"""

    def __init__(self) -> None:
        """接続先ごとの context_id を初期化する。"""
        self._context_ids: dict[str, str | None] = {}

    async def send_message(
        self,
        message: str,
        file_payloads: list[FileWithBytes | FileWithUri],
        url: str,
        new_conversation: bool = False,
        request_config: Mapping[str, Any] | None = None,
    ) -> str:
        """Purple Agent にメッセージを送り、最終応答本文を返す。"""
        result = await self.send_message_with_trajectory(
            message=message,
            file_payloads=file_payloads,
            url=url,
            new_conversation=new_conversation,
            request_config=request_config,
            capture_trajectory=False,
        )
        return str(result.get("response", ""))

    async def send_message_with_trajectory(
        self,
        message: str,
        file_payloads: list[FileWithBytes | FileWithUri],
        url: str,
        new_conversation: bool = False,
        request_config: Mapping[str, Any] | None = None,
        capture_trajectory: bool = False,
    ) -> dict[str, Any]:
        """Purple Agent にメッセージを送り、応答と任意の trajectory を返す。"""
        outbound_message = build_purple_request_text(message, request_config)
        outputs = await send_message_with_files(
            message=outbound_message,
            file_payloads=file_payloads,
            base_url=url,
            context_id=None if new_conversation else self._context_ids.get(url),
            streaming=capture_trajectory,
            capture_events=capture_trajectory,
        )
        if outputs.get("status", "completed") != "completed":
            raise PurpleAgentError(url, outputs)
        self._context_ids[url] = outputs.get("context_id")
        return {
            "response": str(outputs.get("response", "")),
            "status": outputs.get("status", "completed"),
            "context_id": outputs.get("context_id"),
            "trajectory": outputs.get("events", []),
            "artifacts": outputs.get("artifacts", []),
        }

    def reset(self) -> None:
        """保持している会話状態をクリアする。"""
        self._context_ids = {}
