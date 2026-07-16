import hashlib
from typing import Optional
from core.hotstuff.quorum import QuorumCertifier


class InterRegionalConsensus:
    """HotStuff simplificado entre representantes de clusters."""

    def __init__(self, n_clusters: int, f: int):
        self.n = n_clusters
        self.f = f
        self.qc = QuorumCertifier()

    def run_round(self, regional_outputs: list[tuple[str, bytes]],
                  leader_id: str) -> Optional[list[bytes]]:
        """Executa consenso inter-regional (1 fase para protótipo).

        Retorna lista de delta_w aprovados ou None se falhar.
        """
        if len(regional_outputs) < self.n - self.f:
            return None

        inputs = [data for _, data in regional_outputs]
        block_hash = hashlib.sha256(b"".join(inputs)).digest()

        # Para protótipo: assume quorum se temos n-f entradas
        # Em produção: 3 fases completas como no engine.py
        if len(inputs) >= self.qc.quorum_size(self.n):
            return inputs
        return None

    def quorum_size(self) -> int:
        return self.qc.quorum_size(self.n)
