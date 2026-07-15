from typing import Protocol, Optional
from dataclasses import dataclass


@dataclass
class NetworkMessage:
    sender: str
    payload: bytes
    timestamp: float


class NetworkInterface(Protocol):
    async def send(self, sender: str, recipient: str, payload: bytes) -> None:
        ...

    async def broadcast(self, sender: str, payload: bytes, recipients: list[str]) -> None:
        ...

    async def receive(self, node_id: str, timeout: float = 1.0) -> Optional[NetworkMessage]:
        ...

    async def register_node(self, node_id: str) -> None:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...
