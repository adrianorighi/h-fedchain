import hashlib
import logging
from typing import Optional
from core.hotstuff.quorum import QuorumCertifier
from hfc_types.messages import MessageType
from hfc_types.block import QuorumCertificate

logger = logging.getLogger(__name__)


class InterRegionalConsensus:
    """HotStuff 3-phase consensus between cluster representatives."""

    def __init__(self, n_clusters: int, f: int):
        self.n = n_clusters
        self.f = f
        self.qc_certifier = QuorumCertifier()
        self.qc: Optional[QuorumCertificate] = None

    def run_round(self, regional_outputs: list[tuple[str, bytes]],
                  leader_id: str) -> Optional[list[bytes]]:
        quorum = self.qc_certifier.quorum_size(self.n)
        if len(regional_outputs) < quorum:
            logger.warning("inter-regional round aborted: %d inputs < %d quorum",
                           len(regional_outputs), quorum)
            return None

        inputs = [data for _, data in regional_outputs]
        block_hash = hashlib.sha256(b"".join(inputs)).digest()

        n_inputs = len(inputs)
        phases = [
            ("prepare", MessageType.PREPARE),
            ("pre_commit", MessageType.PRE_COMMIT),
            ("commit", MessageType.COMMIT),
        ]

        for phase_name, msg_type in phases:
            votes = self._simulate_votes(n_inputs, phase_name, block_hash)
            try:
                qc = self.qc_certifier.collect(
                    round=0, block_hash=block_hash,
                    msg_type=msg_type,
                    signatures=votes,
                    quorum_size=quorum,
                )
            except ValueError:
                logger.warning("inter-regional quorum not reached at %s phase", phase_name)
                return None
            if phase_name == "commit":
                self.qc = qc

        return inputs

    def _simulate_votes(self, n_voters: int, phase: str,
                        block_hash: bytes) -> list[tuple[str, bytes]]:
        votes = []
        for i in range(n_voters):
            node_id = f"repr_{i}"
            sig = (str(0).encode() + block_hash + phase.encode())[:32]
            votes.append((node_id, sig))
        return votes

    def quorum_size(self) -> int:
        return self.qc_certifier.quorum_size(self.n)
