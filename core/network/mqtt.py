import json
import logging
from typing import Optional, Callable, Awaitable
import asyncio
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


def _subscription_matches(subscription: str, topic: str) -> bool:
    """MQTT topic-filter matching for a single subscription key.

    Supports the multi-level ``#`` wildcard only (the sole wildcard used in
    this codebase). Per MQTT spec, ``sport/tennis/#`` matches
    ``sport/tennis/player1`` AND ``sport/tennis`` itself. Structured so a
    ``+`` (single-level) branch could be added later.
    """
    if subscription.endswith("/#"):
        prefix = subscription[:-2]  # "foo/bar" for "foo/bar/#"
        return topic == prefix or topic.startswith(prefix + "/")
    return subscription == topic


def match_handlers(
    handlers: dict[str, list[Callable]], topic: str
) -> list[Callable]:
    """Return every handler whose subscription filter matches ``topic``.

    Exact-key hits win; otherwise all matching ``/#`` wildcards fire (MQTT
    overlapping-subscription semantics).
    """
    exact = handlers.get(topic)
    if exact:
        return list(exact)
    matched: list[Callable] = []
    for key, key_handlers in handlers.items():
        if key.endswith("/#") and _subscription_matches(key, topic):
            matched.extend(key_handlers)
    return matched


class MqttClient:
    def __init__(self, client_id: str, broker_host: str = "localhost", broker_port: int = 1883):
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self._client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self._handlers: dict[str, list[Callable]] = {}
        self._subscriptions: dict[str, Callable[[bytes], None]] = {}
        self._connected = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _on_connect(self, client, userdata, flags, rc, *args):
        if rc == 0 and self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._handle_connect)
            except RuntimeError:
                return

    def _handle_connect(self):
        self._connected.set()
        for topic in self._subscriptions:
            self._client.subscribe(topic)

    def _on_disconnect(self, client, userdata, *args):
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._handle_disconnect)
        except RuntimeError:
            return

    def _handle_disconnect(self):
        if self._connected.is_set():
            logger.info("MQTT client %s disconnected", self.client_id)
        self._connected.clear()

    def _on_message(self, client, userdata, msg):
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._dispatch, msg.topic, msg.payload)
        except RuntimeError:
            return

    def _dispatch(self, topic: str, payload: bytes):
        for handler in match_handlers(self._handlers, topic):
            try:
                handler(payload)
            except Exception as e:
                logger.error("MQTT handler failed for topic %s: %s", topic, e)

    def subscribe(self, topic: str, handler: Callable[[bytes], None]):
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)
        self._subscriptions[topic] = handler
        if self._connected.is_set():
            self._client.subscribe(topic)

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        """Publish a payload; False (with a warning) when the client is not
        connected or the client rejected the publish, so a dropped message
        is never silent."""
        if not self._connected.is_set():
            logger.warning(
                "MQTT client %s is not connected; dropping publish to %s",
                self.client_id, topic,
            )
            return False
        info = self._client.publish(topic, payload, retain=retain)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(
                "MQTT publish to %s failed for client %s (rc=%s)",
                topic, self.client_id, info.rc,
            )
            return False
        return True

    async def start(self):
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        loop = asyncio.get_running_loop()
        self._loop = loop
        result = await loop.run_in_executor(
            None, self._client.connect, self.broker_host, self.broker_port, 60
        )
        self._client.loop_start()
        await asyncio.wait_for(self._connected.wait(), timeout=5.0)

    async def stop(self):
        self._connected.clear()
        self._client.loop_stop()
        self._client.disconnect()
