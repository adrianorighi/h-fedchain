from dataclasses import dataclass, field
from typing import Optional
import time
from core.pki.ca import CertificateAuthority
from core.pki.ed25519 import generate_keypair, sign, verify
from core.pki.certificate import Certificate


@dataclass
class NodeRegistration:
    node_id: str
    node_type: str
    cluster_id: str = ""
    registered_at: float = 0.0
    last_seen: float = 0.0
    active: bool = True


class GlobalIdentityService:
    def __init__(self):
        self.ca_sk, self.ca_vk = generate_keypair()
        self.ca = CertificateAuthority("cloud_ca", self.ca_sk, self.ca_vk)
        self._registrations: dict[str, NodeRegistration] = {}
        self._certificates: dict[str, Certificate] = {}

    def register_node(self, node_id: str, node_type: str,
                     cluster_id: str = "",
                     public_key: Optional[bytes] = None) -> Certificate:
        if public_key is None:
            raise ValueError("public_key é obrigatório")
        cert = self.ca.issue_certificate(node_id, public_key)
        self._certificates[node_id] = cert
        self._registrations[node_id] = NodeRegistration(
            node_id=node_id,
            node_type=node_type,
            cluster_id=cluster_id,
            registered_at=time.time(),
            last_seen=time.time(),
        )
        return cert

    def verify_node(self, node_id: str, message: bytes,
                   signature: bytes) -> bool:
        cert = self._certificates.get(node_id)
        if not cert or cert.is_expired():
            return False
        if node_id in self.ca._revoked:
            return False
        return verify(cert.public_key, message, signature)

    def revoke_node(self, node_id: str):
        self.ca.revoke_certificate(node_id)
        if node_id in self._registrations:
            self._registrations[node_id].active = False

    def get_certificate(self, node_id: str) -> Optional[Certificate]:
        return self._certificates.get(node_id)

    def list_active_nodes(self, cluster_id: Optional[str] = None) -> list[str]:
        nodes = []
        for nid, reg in self._registrations.items():
            if reg.active:
                if cluster_id is None or reg.cluster_id == cluster_id:
                    nodes.append(nid)
        return nodes

    def update_last_seen(self, node_id: str):
        if node_id in self._registrations:
            self._registrations[node_id].last_seen = time.time()
