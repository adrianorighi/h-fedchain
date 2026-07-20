import json
import logging
from typing import Optional, Callable, Awaitable
import asyncio
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MqttClient:
    def __init__(self, client_id: str, broker_host: str = "localhost", broker_port: int = 1883):
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self._client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self._handlers: dict[str, list[Callable]] = {}
        self._connected = asyncio.Event()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected.set()

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        if topic in self._handlers:
            for handler in self._handlers[topic]:
                try:
                    handler(msg.payload)
                except Exception as e:
                    logger.error("MQTT handler failed for topic %s: %s", topic, e)

    def subscribe(self, topic: str, handler: Callable[[bytes], None]):
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)
        self._client.subscribe(topic)

    def publish(self, topic: str, payload: bytes):
        self._client.publish(topic, payload)

    async def start(self):
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._client.connect, self.broker_host, self.broker_port, 60
        )
        self._client.loop_start()
        await asyncio.wait_for(self._connected.wait(), timeout=5.0)

    async def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
