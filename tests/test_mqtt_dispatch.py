import asyncio, logging, threading
import pytest
from unittest.mock import MagicMock
from core.network.mqtt import MqttClient

@pytest.mark.asyncio
async def test_handler_runs_in_event_loop_thread():
    client = MqttClient("t1", "localhost", 1883)
    loop = asyncio.get_running_loop()
    client._loop = loop
    seen = {}

    def handler(payload: bytes):
        seen["payload"] = payload
        seen["thread"] = threading.get_ident()

    client._handlers["t/1"] = [handler]
    t = threading.Thread(target=client._on_message,
                         args=(None, None, MagicMock(topic="t/1", payload=b"ola")))
    t.start(); t.join()
    for _ in range(100):
        if "payload" in seen: break
        await asyncio.sleep(0.01)
    assert seen["payload"] == b"ola"
    assert seen["thread"] == threading.get_ident()

def test_publish_passes_retain():
    client = MqttClient("t2")
    client._client = MagicMock()
    client._connected.set()  # connected: publish must reach the client
    client._client.publish.return_value.rc = 0  # MQTT_ERR_SUCCESS
    assert client.publish("t/1", b"x", retain=True) is True
    client._client.publish.assert_called_once_with("t/1", b"x", retain=True)


def test_publish_nonzero_rc_warns_and_returns_false(caplog):
    """A publish the broker rejected (rc != MQTT_ERR_SUCCESS) must not be
    reported as delivered: warn naming the topic and return False."""
    client = MqttClient("t6")
    client._client = MagicMock()
    client._connected.set()
    client._client.publish.return_value.rc = 4  # MQTT_ERR_NO_CONN

    with caplog.at_level(logging.WARNING):
        result = client.publish("t/6", b"x")

    assert result is False
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("t/6" in w for w in warnings), (
        f"expected a failed-publish warning naming the topic, got: {warnings}"
    )


async def test_subscribe_waits_for_connect_and_reconnect_resubscribes():
    """Subscriptions must survive a dropped connection: subscribe() records
    the topic (only hitting the wire when connected), and every recorded
    topic is re-subscribed on each (re)connect."""
    client = MqttClient("t5")
    client._client = MagicMock()
    client._loop = asyncio.get_running_loop()

    client.subscribe("t/a", lambda p: None)
    client.subscribe("t/b", lambda p: None)
    client._client.subscribe.assert_not_called(), (
        "subscribe before connect must only be recorded"
    )
    assert set(client._subscriptions) == {"t/a", "t/b"}

    client._on_connect(None, None, {}, 0, None)
    await asyncio.sleep(0)  # run the marshalled callback
    assert client._connected.is_set(), "connect must set _connected"
    assert sorted(c.args[0] for c in client._client.subscribe.call_args_list) \
        == ["t/a", "t/b"]

    client._on_disconnect(None, None, {}, 0, None)
    await asyncio.sleep(0)
    assert not client._connected.is_set(), "disconnect must clear _connected"

    client._client.subscribe.reset_mock()
    client._on_connect(None, None, {}, 0, None)
    await asyncio.sleep(0)
    resubscribed = sorted(
        c.args[0] for c in client._client.subscribe.call_args_list
    )
    assert resubscribed == ["t/a", "t/b"], (
        f"reconnect must re-subscribe every recorded topic, got {resubscribed}"
    )
    assert client._connected.is_set()


def test_reconnect_delay_set_in_ctor():
    from unittest.mock import patch

    with patch("core.network.mqtt.mqtt.Client") as mock_cls:
        MqttClient("t7")
    mock_cls.return_value.reconnect_delay_set.assert_called_once_with(
        min_delay=1, max_delay=30
    )

def test_publish_when_disconnected_warns_and_returns_false(caplog):
    """A publish on a never-connected client must not be a silent loss:
    warn and report failure instead of handing paho a payload it drops."""
    client = MqttClient("t4")
    client._client = MagicMock()

    with caplog.at_level(logging.WARNING):
        result = client.publish("t/4", b"x")

    assert result is False
    client._client.publish.assert_not_called()
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("t/4" in w for w in warnings), (
        f"expected a dropped-publish warning naming the topic, got: {warnings}"
    )

def test_on_message_dropped_without_loop():
    client = MqttClient("t3")
    called = []
    client._handlers["t/3"] = [called.append]
    client._on_message(None, None, MagicMock(topic="t/3", payload=b"x"))
    assert called == []


# ----------------------------------------------------------------------
# Fix C1 — wildcard ('#') subscription matching in _dispatch
# ----------------------------------------------------------------------

def test_match_handlers_wildcard_matches_child_and_parent():
    """MQTT spec: 'foo/bar/#' matches 'foo/bar' itself AND any topic under
    'foo/bar/'. Topics outside the prefix must not match."""
    from core.network.mqtt import match_handlers

    handler = lambda payload: None
    handlers = {"register/c1/#": [handler]}

    assert match_handlers(handlers, "register/c1/d0") == [handler]
    assert match_handlers(handlers, "register/c1") == [handler]
    assert match_handlers(handlers, "register/c1/d0/x") == [handler]
    assert match_handlers(handlers, "register/c12/d0") == []
    assert match_handlers(handlers, "register/c2") == []
    assert match_handlers(handlers, "register/c10") == []


def test_match_handlers_exact_subscription_regression():
    """Exact-key subscriptions keep working unchanged."""
    from core.network.mqtt import match_handlers

    h1, h2 = (lambda p: None), (lambda p: None)
    handlers = {"gradients/c1": [h1, h2]}

    assert match_handlers(handlers, "gradients/c1") == [h1, h2]
    assert match_handlers(handlers, "gradients/c1/d0") == []
    assert match_handlers(handlers, "gradients/c2") == []
    assert match_handlers(handlers, "unrelated") == []


def test_match_handlers_exact_wins_over_wildcard():
    from core.network.mqtt import match_handlers

    exact, wild = (lambda p: None), (lambda p: None)
    handlers = {"register/c1": [exact], "register/c1/#": [wild]}

    assert match_handlers(handlers, "register/c1") == [exact]


def test_match_handlers_all_matching_wildcards_fire():
    """Overlapping wildcard subscriptions all deliver (MQTT semantics)."""
    from core.network.mqtt import match_handlers

    h_a, h_ab = (lambda p: None), (lambda p: None)
    handlers = {"a/#": [h_a], "a/b/#": [h_ab]}

    assert match_handlers(handlers, "a/b/c") == [h_a, h_ab]
    assert match_handlers(handlers, "a") == [h_a]


def test_dispatch_delivers_wildcard_subscription_payloads():
    """End-to-end through a REAL MqttClient (no broker, no fakes): the fog's
    'register/{cluster}/#' subscription must receive per-device topics AND
    the legacy '{cluster}' topic."""
    client = MqttClient("t8")
    seen: list[bytes] = []
    client.subscribe("register/c1/#", seen.append)

    client._dispatch("register/c1/d0", b"device-registration")
    client._dispatch("register/c1", b"legacy-registration")
    client._dispatch("register/c2/d0", b"other-cluster")
    client._dispatch("gradients/c1", b"no-subscription")

    assert seen == [b"device-registration", b"legacy-registration"]


def test_dispatch_exact_subscription_unchanged():
    client = MqttClient("t9")
    seen: list[bytes] = []
    client.subscribe("gradients/c1", seen.append)

    client._dispatch("gradients/c1", b"g1")
    client._dispatch("gradients/c2", b"g2")

    assert seen == [b"g1"]


# ----------------------------------------------------------------------
# Fix M1 — stop() must not leave the client reporting connected
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_clears_connected_flag():
    client = MqttClient("t10")
    client._client = MagicMock()
    client._connected.set()

    await client.stop()

    assert not client._connected.is_set(), (
        "a stopped client must not report connected"
    )
    assert client.publish("t/x", b"y") is False, (
        "publish after stop must fail instead of handing paho a payload"
    )
    client._client.loop_stop.assert_called_once()
    client._client.disconnect.assert_called_once()
