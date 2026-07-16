from services.identity_service import GlobalIdentityService
from core.pki.ed25519 import generate_keypair, sign


def test_register_and_verify():
    gis = GlobalIdentityService()
    sk, vk = generate_keypair()
    gis.register_node("fog1", "fog", "cluster_0", vk)
    msg = b"test_message"
    sig = sign(sk, msg)
    assert gis.verify_node("fog1", msg, sig) is True


def test_verify_fails_unregistered():
    gis = GlobalIdentityService()
    sk, vk = generate_keypair()
    msg = b"test"
    sig = sign(sk, msg)
    assert gis.verify_node("unknown", msg, sig) is False


def test_revoke():
    gis = GlobalIdentityService()
    sk, vk = generate_keypair()
    gis.register_node("edge1", "edge", "cluster_0", vk)
    gis.revoke_node("edge1")
    msg = b"test"
    sig = sign(sk, msg)
    assert gis.verify_node("edge1", msg, sig) is False


def test_list_active():
    gis = GlobalIdentityService()
    sk1, vk1 = generate_keypair()
    sk2, vk2 = generate_keypair()
    gis.register_node("fog1", "fog", "c0", vk1)
    gis.register_node("fog2", "fog", "c1", vk2)
    assert gis.list_active_nodes("c0") == ["fog1"]
    assert len(gis.list_active_nodes()) == 2


def test_list_active_after_revoke():
    gis = GlobalIdentityService()
    sk, vk = generate_keypair()
    gis.register_node("malicious", "edge", "c0", vk)
    gis.revoke_node("malicious")
    assert "malicious" not in gis.list_active_nodes()


def test_update_last_seen():
    gis = GlobalIdentityService()
    sk, vk = generate_keypair()
    gis.register_node("fog1", "fog", "c0", vk)
    old = gis._registrations["fog1"].last_seen
    gis.update_last_seen("fog1")
    assert gis._registrations["fog1"].last_seen >= old
