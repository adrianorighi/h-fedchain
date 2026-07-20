import hashlib
from typing import Optional
from core.hotstuff.quorum import QuorumCertifier
from hfc_types.messages import MessageType
from hfc_types.block import QuorumCertificate

P = 2147483647  # Prime for STARK field


class InterRegionalConsensus:
    """HotStuff 3-phase consensus between cluster representatives."""

    def __init__(self, n_clusters: int, f: int):
        self.n = n_clusters
        self.f = f
        self.qc_certifier = QuorumCertifier()
        self.qc: Optional[QuorumCertificate] = None
        self.qc_signatures: list[tuple[str, bytes]] = []

    def run_round(self, regional_outputs: list[tuple[str, bytes]],
                  leader_id: str) -> Optional[list[bytes]]:
        if len(regional_outputs) < self.n - self.f:
            return None

        inputs = [data for _, data in regional_outputs]
        block_hash = hashlib.sha256(b"".join(inputs)).digest()

        n_valid = len(inputs)
        quorum = self.qc_certifier.quorum_size(self.n)

        # Fase 1: PREPARE
        prepare_votes = self._simulate_votes(n_valid, "prepare", block_hash)
        qc_prepare = self.qc_certifier.collect(
            round=0, block_hash=block_hash,
            msg_type=MessageType.PREPARE,
            signatures=prepare_votes,
            quorum_size=quorum,
        )
        if qc_prepare is None:
            return None

        # Fase 2: PRE-COMMIT
        pre_commit_votes = self._simulate_votes(n_valid, "pre_commit", block_hash)
        qc_pre_commit = self.qc_certifier.collect(
            round=0, block_hash=block_hash,
            msg_type=MessageType.PRE_COMMIT,
            signatures=pre_commit_votes,
            quorum_size=quorum,
        )
        if qc_pre_commit is None:
            return None

        # Fase 3: COMMIT
        commit_votes = self._simulate_votes(n_valid, "commit", block_hash)
        qc_commit = self.qc_certifier.collect(
            round=0, block_hash=block_hash,
            msg_type=MessageType.COMMIT,
            signatures=commit_votes,
            quorum_size=quorum,
        )
        if qc_commit is None:
            return None

        self.qc = qc_commit
        self.qc_signatures = commit_votes[:quorum]
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
