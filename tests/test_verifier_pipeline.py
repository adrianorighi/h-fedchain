from core.pki.verifier import VerificationPipeline
from core.pki.ca import CertificateAuthority
from core.pki.ed25519 import generate_keypair, sign
from hfc_types.messages import Gradient, GradientWithProof


def _make_grad(node_id, round_num, sk, data=None):
    from core.pki.gradient import gradient_signed_message
    grad_data = data or [0.1, 0.2]
    msg = gradient_signed_message(node_id, round_num, grad_data)
    sig = sign(sk, msg)
    return GradientWithProof(
        gradient=Gradient(node_id=node_id, round=round_num,
                         data=grad_data,
                         signature=sig),
        snark_proof=None
    )


def test_pipeline_accepts_honest():
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    node_sk, node_vk = generate_keypair()
    cert = ca.issue_certificate("node0", node_vk)
    cert_map = {"node0": cert}
    pipeline = VerificationPipeline(ca, ca_vk, cert_map)
    gp = _make_grad("node0", 1, node_sk)
    result = pipeline.verify(gp)
    assert result.accepted is True
    assert result.pki_ok is True


def test_pipeline_rejects_unregistered():
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    pipeline = VerificationPipeline(ca, ca_vk, {})
    node_sk, _ = generate_keypair()
    gp = _make_grad("unknown", 1, node_sk)
    result = pipeline.verify(gp)
    assert result.accepted is False
    assert result.reason == "pki_no_cert"


def test_pipeline_rejects_wrong_signature():
    ca_sk, ca_vk = generate_keypair()
    ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
    node_sk, node_vk = generate_keypair()
    cert = ca.issue_certificate("node0", node_vk)
    cert_map = {"node0": cert}
    pipeline = VerificationPipeline(ca, ca_vk, cert_map)
    other_sk, _ = generate_keypair()
    gp = _make_grad("node0", 1, other_sk)
    result = pipeline.verify(gp)
    assert result.accepted is False
    assert result.reason == "pki_sig_fail"
