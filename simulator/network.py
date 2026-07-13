import asyncio
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class NetworkMessage:
    sender: str
    recipient: str
    payload: bytes
    delivery_time: float


class EmulatedNetwork:
    def __init__(self, latency_ms: float = 10.0):
        self.latency_ms = latency_ms
        self._queues: dict[str, asyncio.Queue[NetworkMessage]] = {}

    def add_node(self, node_id: str):
        self._queues[node_id] = asyncio.Queue()

    async def send(self, sender: str, recipient: str, payload: bytes):
        if recipient not in self._queues:
            return
        msg = NetworkMessage(
            sender=sender,
            recipient=recipient,
            payload=payload,
            delivery_time=time.time() + self.latency_ms / 1000.0,
        )
        await self._queues[recipient].put(msg)

    async def broadcast(self, sender: str, payload: bytes, recipients: list[str]):
        for r in recipients:
            await self.send(sender, r, payload)

    async def receive(self, node_id: str, timeout: float) -> Optional[NetworkMessage]:
        try:
            msg = await asyncio.wait_for(
                self._queues[node_id].get(), timeout=timeout
            )
            now = time.time()
            delay = msg.delivery_time - now
            if delay > 0:
                await asyncio.sleep(delay)
            return msg
        except asyncio.TimeoutError:
            return None
