import hashlib
from typing import Optional
from core.hotstuff.interregional import InterRegionalConsensus
from zkp.stark import StarkVerifier, StarkProver
from hfc_types.block import GlobalOutput, RegionalOutput


class InterRegionalManager:
    """Gerencia consenso entre clusters e agregação inter-regional."""

    def __init__(self, clusters: Optional[list] = None, n: int = 0, f: int = 1,
                 stark_verifier: Optional[StarkVerifier] = None,
                 stark_prover: Optional[StarkProver] = None):
        self.clusters = clusters or []
        self.n = n or max(len(self.clusters), 1)
        self.f = f
        self.consensus = InterRegionalConsensus(self.n, self.f)
        self.stark_verifier = stark_verifier or StarkVerifier()
        self.stark_prover = stark_prover or StarkProver()
        self.vk_map: dict[str, bytes] = {}

    def set_vk(self, node_id: str, vk: bytes):
        self.vk_map[node_id] = vk

    async def run_round(self, round_num: int) -> GlobalOutput:
        """Executa rodada inter-regional completa.

        Returns:
            GlobalOutput com delta_w consolidado, prova STARK e métricas.
        """
        # 1. Coletar outputs regionais com verificação STARK
        regionals: list[RegionalOutput] = []
        for cluster in self.clusters:
            try:
                output = await cluster.get_regional_output(round_num)
            except RuntimeError:
                continue
            if output is None:
                continue

            if output.stark_proof is not None:
                is_valid = await self.stark_verifier.verify(
                    output.stark_proof, output.stark_proof.public_inputs
                )
                if not is_valid:
                    continue

            regionals.append(output)

        if not regionals:
            return GlobalOutput(
                delta_w_inter=b"", pi_inter=None,
                n_active_clusters=0, round_num=round_num,
            )

        # 2. Eleger líder inter-regional via VRF simplificado
        seed = hashlib.sha256(f"inter_round_{round_num}".encode()).digest()
        candidates = [getattr(c, 'representative_id', c.cluster_id)
                      for c in self.clusters]
        leader_id = self._elect_leader(candidates, seed)

        # 3. Consenso inter-regional
        approved = self.consensus.run_round(
            [(r.cluster_id, r.delta_w) for r in regionals],
            leader_id,
        )
        if not approved:
            return GlobalOutput(
                delta_w_inter=b"", pi_inter=None,
                n_active_clusters=0, round_num=round_num,
            )

        # 4. FedAvg ponderado (placeholder: delta_w são hashes no protótipo)
        n_total = sum(getattr(c, 'n_devices', 1) or 1 for c in self.clusters)
        delta_w_inter = regionals[0].delta_w

        # 5. STARK proof inter-regional (placeholder para protótipo)
        pi_inter = None

        return GlobalOutput(
            delta_w_inter=delta_w_inter,
            pi_inter=pi_inter,
            n_active_clusters=len(regionals),
            round_num=round_num,
        )

    def _elect_leader(self, candidates: list[str], seed: bytes) -> str:
        if not candidates:
            return ""
        return min(candidates, key=lambda nid: int.from_bytes(
            hashlib.sha256(nid.encode() + seed).digest()[:8], 'big'
        ))
