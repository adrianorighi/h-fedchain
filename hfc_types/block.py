from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Optional

from .messages import MessageType

if TYPE_CHECKING:
    from .crypto import StarkProof


@dataclass
class RegionalOutput:
    delta_w: bytes
    stark_proof: Optional["StarkProof"]
    qc_commit: Optional["QuorumCertificate"]
    n_devices: int
    round_num: int
    cluster_id: str
    delta_w_data: list[float] | None = None
    snark_proofs_total: int = 0
    snark_verify_projected_ms: float = 0.0
    snark_sampled_passed: int = 1
    pipeline_latency_ms: float = 0.0


@dataclass
class GlobalOutput:
    delta_w_inter: bytes
    pi_inter: Optional["StarkProof"]
    n_active_clusters: int
    round_num: int
    regionals: list["RegionalOutput"] = field(default_factory=list)


@dataclass
class QuorumCertificate:
    round: int
    block_hash: bytes
    signatures: list[tuple[str, bytes]]
    msg_type: MessageType

    def is_valid(self, quorum_size: int, vk_map: dict[str, bytes] | None = None) -> bool:
        if len(self.signatures) < quorum_size:
            return False
        if vk_map is not None:
            from core.pki import verify as pki_verify
            for node_id, sig in self.signatures[:quorum_size]:
                vk = vk_map.get(node_id)
                if vk is None:
                    return False
                msg = str(self.round).encode() + self.block_hash + self.msg_type.name.encode()
                if not pki_verify(vk, msg, sig):
                    return False
        return True


@dataclass(frozen=True)
class Block:
    round: int
    gradient_hash: bytes
    qc_commit: Optional["QuorumCertificate"]
    stark_proof: Optional["StarkProof"]
    accepted_devices: list[str]
    rejected_devices: list[str]
    timestamp: float
    prev_hash: bytes
    n: int = 0
    f: int = 0
    hash: bytes = field(init=False)

    def __post_init__(self):
        payload = (
            str(self.round) +
            self.gradient_hash.hex() +
            str(self.timestamp) +
            self.prev_hash.hex()
        ).encode()
        object.__setattr__(self, 'hash', sha256(payload).digest())


@dataclass
class LedgerEntry:
    block: Block
    node_id: str
    stored_at: float
    verified: bool = True
