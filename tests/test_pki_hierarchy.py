from core.pki.ca import CertificateAuthority
from core.pki.ed25519 import generate_keypair


def test_ca_issue_and_verify():
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    node_sk, node_vk = generate_keypair()
    cert = ca.issue_certificate("node0", node_vk)
    assert ca.verify_certificate(cert, ca_vk) is True


def test_ca_revoke():
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    node_sk, node_vk = generate_keypair()
    cert = ca.issue_certificate("node0", node_vk)
    ca.revoke_certificate("node0")
    assert ca.verify_certificate(cert, ca_vk) is False


def test_expired_cert():
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    node_sk, node_vk = generate_keypair()
    cert = ca.issue_certificate("node0", node_vk, validity_days=-1)
    assert ca.verify_certificate(cert, ca_vk) is False


def test_cert_serialize_roundtrip():
    import pickle
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    node_sk, node_vk = generate_keypair()
    cert = ca.issue_certificate("node0", node_vk)
    data = cert.serialize()
    restored = pickle.loads(data)
    assert restored.node_id == "node0"
    assert restored.public_key == node_vk


def test_verify_with_wrong_ca_key():
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    node_sk, node_vk = generate_keypair()
    cert = ca.issue_certificate("node0", node_vk)
    wrong_sk, wrong_vk = generate_keypair()
    assert ca.verify_certificate(cert, wrong_vk) is False
