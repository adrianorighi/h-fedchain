from dataclasses import dataclass
from typing import Optional
from core.pki.ed25519 import verify as pki_verify
from core.pki.ca import CertificateAuthority
from core.pki.certificate import Certificate
from hfc_types.messages import Gradient, GradientWithProof


@dataclass
class VerificationResult:
    node_id: str
    gradient_data: Optional[list[float]] = None
    pki_ok: bool = False
    snark_ok: bool = False
    accepted: bool = False
    reason: str = ""


class VerificationPipeline:
    def __init__(self, ca: CertificateAuthority, ca_vk: bytes,
                 cert_map: dict[str, Certificate],
                 use_snark: bool = False):
        self.ca = ca
        self.ca_vk = ca_vk
        self.cert_map = cert_map
        self.use_snark = use_snark

    def verify(self, grad: GradientWithProof, model_hash: bytes = b"") -> VerificationResult:
        result = VerificationResult(node_id=grad.gradient.node_id)

        cert = self.cert_map.get(grad.gradient.node_id)
        if not cert:
            result.reason = "pki_no_cert"
            return result
        if not self.ca.verify_certificate(cert, self.ca_vk):
            result.reason = "pki_cert_invalid"
            return result
        result.pki_ok = True

        from core.pki.gradient import gradient_signed_message
        msg = gradient_signed_message(
            grad.gradient.node_id, grad.gradient.round, grad.gradient.data
        )
        if not pki_verify(cert.public_key, msg, grad.gradient.signature):
            result.reason = "pki_sig_fail"
            return result

        if self.use_snark and grad.snark_proof is not None:
            from zkp.snark import SnarkVerifier
            verifier = SnarkVerifier()
            import asyncio
            if not asyncio.run(verifier.verify(grad.snark_proof, model_hash, cert.public_key)):
                result.reason = "snark_fail"
                result.snark_ok = False
                return result
            result.snark_ok = True

        result.gradient_data = grad.gradient.data
        result.accepted = True
        return result
