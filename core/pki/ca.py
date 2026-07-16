import time
from .certificate import Certificate
from .ed25519 import sign, verify


class CertificateAuthority:
    def __init__(self, ca_id: str, sk: bytes, vk: bytes):
        self.ca_id = ca_id
        self.sk = sk
        self.vk = vk
        self._issued: list[Certificate] = []
        self._revoked: set[str] = set()

    def issue_certificate(self, node_id: str, public_key: bytes, validity_days: int = 365) -> Certificate:
        issued_at = time.time()
        expires_at = issued_at + validity_days * 86400
        to_sign = f"{node_id}:{public_key.hex()}:{issued_at}:{expires_at}".encode()
        sig = sign(self.sk, to_sign)
        cert = Certificate(
            node_id=node_id, public_key=public_key,
            issuer=self.ca_id, issued_at=issued_at,
            expires_at=expires_at, signature=sig,
        )
        self._issued.append(cert)
        return cert

    def verify_certificate(self, cert: Certificate, ca_vk: bytes) -> bool:
        if cert.node_id in self._revoked:
            return False
        if cert.is_expired():
            return False
        to_verify = f"{cert.node_id}:{cert.public_key.hex()}:{cert.issued_at}:{cert.expires_at}".encode()
        return verify(ca_vk, to_verify, cert.signature)

    def revoke_certificate(self, node_id: str):
        self._revoked.add(node_id)
