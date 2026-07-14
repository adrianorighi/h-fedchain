from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from hfc_types.crypto import SnarkProof


class MessageType(Enum):
    PREPARE = auto()
    PRE_COMMIT = auto()
    COMMIT = auto()
    QC_COMMIT = auto()
    VRF_SHARE = auto()
    VIEW_CHANGE = auto()
    GRADIENT = auto()


@dataclass
class Gradient:
    node_id: str
    round: int
    data: list[float]
    signature: Optional[bytes] = None


@dataclass
class GradientWithProof:
    gradient: Gradient
    snark_proof: Optional[SnarkProof] = None


@dataclass
class Vote:
    node_id: str
    round: int
    msg_type: MessageType
    block_hash: bytes
    signature: Optional[bytes] = None


@dataclass
class VRFMessage:
    node_id: str
    round: int
    y: bytes
    proof: bytes


@dataclass
class ViewChangeMessage:
    node_id: str
    round: int
    new_view: int
    signature: Optional[bytes] = None


@dataclass
class AggregateGradient:
    node_id: str
    round: int
    gradient: Gradient
    accepted_devices: list[str]
    rejected_devices: list[str]
    total_adversarial: int = 0
    rejected_adversarial: int = 0
    rejected_honest: int = 0
