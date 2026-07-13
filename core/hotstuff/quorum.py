import math

from hfc_types.block import QuorumCertificate as QC
from hfc_types.messages import MessageType


class QuorumCertifier:
    @staticmethod
    def quorum_size(n: int) -> int:
        return math.ceil(2 * n / 3)

    def collect(
        self,
        round: int,
        block_hash: bytes,
        msg_type: MessageType,
        signatures: list[tuple[str, bytes]],
        quorum_size: int,
    ) -> QC:
        if len(signatures) < quorum_size:
            raise ValueError(
                f"Insufficient signatures for quorum: {len(signatures)} < {quorum_size}"
            )
        return QC(
            round=round,
            block_hash=block_hash,
            signatures=signatures[:quorum_size],
            msg_type=msg_type,
        )
