import hashlib
import logging
from typing import Optional
from core.hotstuff.quorum import QuorumCertifier
from core.pki import sign as pki_sign
from hfc_types.messages import MessageType
from hfc_types.block import QuorumCertificate

logger = logging.getLogger(__name__)


class InterRegionalConsensus:
    """HotStuff 3-phase consensus between cluster representatives."""

    def __init__(self, n_clusters: int, f: int,
                 interregional_latency_ms: float = 50.0,
                 jitter_ms: float = 0.0):
        self.n = n_clusters
        self.f = f
        self.qc_certifier = QuorumCertifier()
        self.interregional_latency_ms = interregional_latency_ms
        self.jitter_ms = jitter_ms
        self.qc: Optional[QuorumCertificate] = None

    async def run_round(self, regional_outputs: list[tuple[str, bytes]],
                  leader_id: str,
                  round_num: int = 0,
                  vk_map: Optional[dict[str, bytes]] = None,
                  sk_map: Optional[dict[str, bytes]] = None) -> Optional[list[bytes]]:
        quorum = self.qc_certifier.quorum_size(self.n)
        if len(regional_outputs) < quorum:
            logger.warning("inter-regional round aborted: %d inputs < %d quorum",
                           len(regional_outputs), quorum)
            return None

        inputs = [data for _, data in regional_outputs]
        block_hash = hashlib.sha256(b"".join(inputs)).digest()

        phases = [
            ("prepare", MessageType.PREPARE),
            ("pre_commit", MessageType.PRE_COMMIT),
            ("commit", MessageType.COMMIT),
        ]

        for phase_name, msg_type in phases:
            votes: list[tuple[str, bytes]] = []
            for cid, _ in regional_outputs:
                msg = str(round_num).encode() + block_hash + msg_type.name.encode()
                sig = pki_sign(sk_map[cid], msg) if sk_map and cid in sk_map else msg[:32]
                votes.append((cid, sig))
            try:
                qc = self.qc_certifier.collect(
                    round=round_num, block_hash=block_hash,
                    msg_type=msg_type,
                    signatures=votes,
                    quorum_size=quorum,
                    vk_map=vk_map,
                )
            except ValueError:
                logger.warning("inter-regional quorum not reached at %s phase", phase_name)
                return None
            if phase_name == "commit":
                self.qc = qc

        return inputs

    def quorum_size(self) -> int:
        return self.qc_certifier.quorum_size(self.n)
