from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional


@dataclass
class QuorumCertificate:
    round: int
    block_hash: bytes
    signatures: list[tuple[str, bytes]]
    msg_type: str  # PREPARE, PRE_COMMIT, COMMIT

    def is_valid(self, quorum_size: int) -> bool:
        return len(self.signatures) >= quorum_size


@dataclass
class Block:
    round: int
    gradient_hash: bytes
    qc_commit: Optional["QuorumCertificate"]
    stark_proof: Optional[bytes]
    accepted_devices: list[str]
    rejected_devices: list[str]
    timestamp: float
    prev_hash: bytes
    hash: bytes = field(init=False)

    def __post_init__(self):
        payload = (
            str(self.round) +
            self.gradient_hash.hex() +
            str(self.timestamp) +
            self.prev_hash.hex()
        ).encode()
        self.hash = sha256(payload).digest()


@dataclass
class LedgerEntry:
    block: Block
    node_id: str
    stored_at: float
    verified: bool = True
