import asyncio
import logging
import time
from typing import Optional
from core.network import NetworkMessage

logger = logging.getLogger(__name__)


class GrpcTransport:
    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}

    async def start(self) -> None:
        logger.info("GrpcTransport started (in-process message bus, no gRPC server)")

    async def stop(self) -> None:
        logger.info("GrpcTransport stopped")

    async def register_node(self, node_id: str) -> None:
        if node_id not in self._queues:
            self._queues[node_id] = asyncio.Queue()

    async def send(self, sender: str, recipient: str, payload: bytes) -> None:
        q = self._queues.get(recipient)
        if q is not None:
            await q.put(NetworkMessage(sender=sender, payload=payload, timestamp=time.time()))

    async def broadcast(self, sender: str, payload: bytes, recipients: list[str]) -> None:
        for r in recipients:
            await self.send(sender, r, payload)

    async def receive(self, node_id: str, timeout: float = 1.0) -> Optional[NetworkMessage]:
        q = self._queues.get(node_id)
        if q is None:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
