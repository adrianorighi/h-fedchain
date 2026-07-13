import asyncio
import pytest
from simulator.network import EmulatedNetwork


@pytest.mark.asyncio
async def test_send_and_receive():
    net = EmulatedNetwork(latency_ms=10)
    net.add_node("n1")
    net.add_node("n2")

    await net.send("n1", "n2", b"hello")
    msg = await net.receive("n2", timeout=1.0)
    assert msg is not None
    assert msg.payload == b"hello"
    assert msg.sender == "n1"


@pytest.mark.asyncio
async def test_broadcast():
    net = EmulatedNetwork(latency_ms=10)
    net.add_node("n0")
    net.add_node("n1")
    net.add_node("n2")

    await net.broadcast("n0", b"data", ["n1", "n2"])
    m1 = await net.receive("n1", timeout=1.0)
    m2 = await net.receive("n2", timeout=1.0)
    assert m1 is not None and m2 is not None
    assert m1.payload == b"data"
    assert m2.payload == b"data"


@pytest.mark.asyncio
async def test_timeout_returns_none():
    net = EmulatedNetwork(latency_ms=100)
    net.add_node("n1")
    msg = await net.receive("n1", timeout=0.01)
    assert msg is None
