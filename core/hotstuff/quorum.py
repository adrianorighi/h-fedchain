import math

from hfc_types.block import QuorumCertificate as QC
from hfc_types.messages import MessageType
from core.pki import verify as pki_verify


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
        vk_map: dict[str, bytes] | None = None,
    ) -> QC:
        if len(signatures) < quorum_size:
            raise ValueError(
                f"Insufficient signatures for quorum: {len(signatures)} < {quorum_size}"
            )
        if vk_map is not None:
            msg = str(round).encode() + block_hash + msg_type.name.encode()
            for node_id, sig in signatures[:quorum_size]:
                vk = vk_map.get(node_id)
                if vk is None:
                    raise ValueError(f"Unknown voter: {node_id}")
                if not pki_verify(vk, msg, sig):
                    raise ValueError(f"Invalid signature from {node_id}")
        return QC(
            round=round,
            block_hash=block_hash,
            signatures=signatures[:quorum_size],
            msg_type=msg_type,
        )
