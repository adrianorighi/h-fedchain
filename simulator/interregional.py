import hashlib
import pickle
import time
from typing import Optional
import numpy as np
from core.hotstuff.interregional import InterRegionalConsensus
from core.vrf.election import VRFLeaderElection
from core.pki.ed25519 import generate_keypair

from zkp.stark import StarkVerifier, StarkProver
from hfc_types.block import GlobalOutput, RegionalOutput, Block
from core.ledger.interregional_ledger import InterRegionalLedger


class InterRegionalManager:
    """Gerencia consenso entre clusters e agregação inter-regional."""

    def __init__(self, clusters: Optional[list] = None, n: int = 0, f: int = 1,
                 interregional_latency_ms: float = 50.0, jitter_ms: float = 0.0,
                 stark_verifier: Optional[StarkVerifier] = None,
                 stark_prover: Optional[StarkProver] = None):
        self.clusters = clusters or []
        self.n = n or max(len(self.clusters), 1)
        self.f = f
        self.consensus = InterRegionalConsensus(self.n, self.f, interregional_latency_ms, jitter_ms)
        self.stark_verifier = stark_verifier or StarkVerifier()
        self.stark_prover = stark_prover or StarkProver()
        self.vk_map: dict[str, bytes] = {}
        self.vrf = VRFLeaderElection()
        self._manager_sk, self._manager_vk = generate_keypair()
        self.ledger = InterRegionalLedger()
        self.stage_times: dict[str, float] = {}
        self.last_leader_id: str = ""
        self.last_loss: float = 0.0

    def set_vk(self, node_id: str, vk: bytes):
        self.vk_map[node_id] = vk

    async def run_round(self, round_num: int) -> GlobalOutput:
        """Executa rodada inter-regional completa.

        Returns:
            GlobalOutput com delta_w consolidado, prova STARK e métricas.
        """
        self.stage_times = {}

        # 1. Coletar outputs regionais com verificação STARK
        t0 = time.perf_counter()
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

        # 2. Eleger líder inter-regional via VRF
        seed = hashlib.sha256(f"inter_round_{round_num}".encode()).digest()
        candidates = [c.cluster_id for c in self.clusters]
        t_vrf = time.perf_counter()
        leader_id = self._elect_leader_vrf(candidates, seed)
        self.last_leader_id = leader_id
        self.stage_times["vrf_elect"] = (time.perf_counter() - t_vrf) * 1000

        # 2b. Gerar keypairs PKI para cada representante de cluster
        sk_map: dict[str, bytes] = {}
        vk_map: dict[str, bytes] = {}
        for c in self.clusters:
            cid = c.cluster_id
            sk, vk = generate_keypair()
            sk_map[cid] = sk
            vk_map[cid] = vk

        # 3. Consenso inter-regional com assinaturas PKI reais
        t_cons = time.perf_counter()
        approved = await self.consensus.run_round(
            [(r.cluster_id, r.delta_w) for r in regionals],
            leader_id,
            round_num=round_num,
            vk_map=vk_map,
            sk_map=sk_map,
        )
        self.stage_times["consensus_inter"] = (time.perf_counter() - t_cons) * 1000
        if not approved:
            return GlobalOutput(
                delta_w_inter=b"", pi_inter=None,
                n_active_clusters=0, round_num=round_num,
            )

        # 4. FedAvg ponderado pelo número de dispositivos por cluster
        t_fed = time.perf_counter()
        total_devices = sum(r.n_devices for r in regionals)
        weighted_sum = None
        for r in regionals:
            if r.delta_w_data is None:
                continue
            weight = r.n_devices / total_devices
            grad = np.array(r.delta_w_data)
            if weighted_sum is None:
                weighted_sum = weight * grad
            else:
                weighted_sum += weight * grad

        delta_w_inter = weighted_sum.tolist() if weighted_sum is not None else []
        delta_w_inter_bytes = hashlib.sha256(str(delta_w_inter).encode()).digest()
        self.stage_times["fedavg_weighted"] = (time.perf_counter() - t_fed) * 1000

        # Compute loss proxy = average gradient norm across regional deltas
        grad_norms = [float(np.linalg.norm(np.array(r.delta_w_data)))
                      for r in regionals if r.delta_w_data is not None]
        self.last_loss = float(np.mean(grad_norms)) if grad_norms else 0.0

        # 5. STARK proof inter-regional com dados reais do ledger
        t_stark = time.perf_counter()
        last_prev = self.ledger._hash_chain[-1] if self.ledger._hash_chain else b"\x00" * 32
        inter_block = Block(
            round=round_num,
            gradient_hash=delta_w_inter_bytes,
            qc_commit=None, stark_proof=None,
            accepted_devices=[r.cluster_id for r in regionals],
            rejected_devices=[],
            timestamp=time.time(), prev_hash=last_prev,
            n=self.n, f=self.f,
        )
        pi_inter = await self.stark_prover.generate_proof(inter_block)
        self.stage_times["stark_gen_inter"] = (time.perf_counter() - t_stark) * 1000

        self.ledger.append(round_num, delta_w_inter_bytes, pi_inter.proof_bytes,
                          [r.cluster_id for r in regionals])

        # Bytes inter-cluster (fog→fog): payload regional real (delta_w_data,
        # usado na FedAvg; fallback p/ hash) + provas STARK regionais.
        # Votos QC (64N) contam apenas em bytes_fog_cloud (experiment_runner).
        bytes_fog_inter = (
            sum(
                len(pickle.dumps(r.delta_w_data))
                if r.delta_w_data is not None
                else len(pickle.dumps(r.delta_w))
                for r in regionals
            )
            + sum(
                len(r.stark_proof.proof_bytes)
                for r in regionals
                if r.stark_proof is not None
            )
        )

        return GlobalOutput(
            delta_w_inter=delta_w_inter_bytes,
            pi_inter=pi_inter,
            n_active_clusters=len(regionals),
            round_num=round_num,
            regionals=regionals,
            bytes_fog_inter=bytes_fog_inter,
        )

    def _elect_leader_vrf(self, candidates: list[str], seed: bytes) -> str:
        if not candidates:
            return ""
        best = candidates[0]
        best_gamma = None
        for c in candidates:
            alpha = seed + c.encode()
            gamma, _proof = self.vrf.evaluate(self._manager_sk, alpha)
            gamma_int = int.from_bytes(gamma, 'big')
            if best_gamma is None or gamma_int < best_gamma:
                best_gamma = gamma_int
                best = c
        return best
