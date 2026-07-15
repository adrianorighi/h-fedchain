from dataclasses import dataclass
from typing import Optional
from hfc_types.block import Block, QuorumCertificate


@dataclass
class PrepareProposal:
    leader_id: str
    round: int
    block: Block


@dataclass
class VoteMessage:
    node_id: str
    round: int
    phase: str
    block_hash: bytes
    signature: bytes


@dataclass
class QcBroadcast:
    leader_id: str
    round: int
    phase: str
    qc: QuorumCertificate


@dataclass
class ViewChangeMessage:
    node_id: str
    new_view: int
    highest_qc: Optional[QuorumCertificate]
    signature: bytes


@dataclass
class NewViewMessage:
    leader_id: str
    new_view: int
    qc_set: list[bytes]
    signature: bytes


@dataclass
class VrfShareMessage:
    node_id: str
    round: int
    y: bytes
    proof: bytes


@dataclass
class CommitteeResult:
    delta_w: bytes
    stark_proof: Optional[bytes]
    qc_commit: QuorumCertificate
    round: int
