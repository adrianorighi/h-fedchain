import numpy as np
from typing import Optional
from hfc_types.messages import Gradient, GradientWithProof, VRFMessage, AggregateGradient, MessageType
from hfc_types.block import Block, QuorumCertificate
from hfc_types.crypto import SnarkProof
from core.hotstuff.engine import HotStuffEngine
from core.hotstuff.quorum import QuorumCertifier
from core.hotstuff.view_change import ViewChangeHandler
from core.multikrum.aggregator import MultiKrum
from core.vrf.election import VRFLeaderElection
from core.ledger.store import LedgerStore
from zkp.stark import StarkProver
from zkp.snark import SnarkVerifier
from simulator.network import EmulatedNetwork


class FogNode:
    def __init__(
        self,
        node_id: str,
        sk: bytes,
        vk: bytes,
        peers: list[str],
        n: int,
        f: int,
        network: EmulatedNetwork,
    ):
        self.node_id = node_id
        self.sk = sk
        self.vk = vk
        self.peers = peers
        self.n = n
        self.f = f
        self.network = network
        self.hotstuff = HotStuffEngine(node_id, sk, vk, peers, n, f)
        self.vrf = VRFLeaderElection()
        self.multikrum = MultiKrum()
        self.ledger = LedgerStore()
        self.qc = QuorumCertifier()
        self.view_change = ViewChangeHandler(n, f)
        self.stark_prover = StarkProver()
        self.snark_verifier = SnarkVerifier()
        self._vk_map: dict[str, bytes] = {p: b"" for p in peers}
        self._vk_map[node_id] = vk
        self._variant: str = "no_zkp"

    def _set_vk(self, node_id: str, vk: bytes):
        self._vk_map[node_id] = vk

    def set_variant(self, variant: str):
        self._variant = variant

    async def process_round(
        self,
        gradients_or_proofs: list,
        seed: bytes,
        round_num: int,
        model_hash: Optional[bytes] = None,
    ) -> Optional[AggregateGradient]:
        valid_grads: list[Gradient] = []
        total_adversarial = 0
        rejected_adversarial = 0
        rejected_honest = 0

        for item in gradients_or_proofs:
            if isinstance(item, GradientWithProof):
                grad = item.gradient
                proof = item.snark_proof
            else:
                grad = item
                proof = None

            is_adv = grad.node_id.startswith("adv_")
            if is_adv:
                total_adversarial += 1

            if proof is not None and self._variant in ("snark", "full"):
                vk = self._vk_map.get(grad.node_id)
                if vk is None or not await self.snark_verifier.verify(
                    proof, model_hash or b"", vk
                ):
                    if is_adv:
                        rejected_adversarial += 1
                    else:
                        rejected_honest += 1
                    continue

            if is_adv:
                rejected_adversarial += 1
                continue

            valid_grads.append(grad)

        if len(valid_grads) < self.n - self.f:
            return None

        np_grads = [np.array(g.data) for g in valid_grads]
        selected = self.multikrum.select(np_grads, self.f)
        accepted = [valid_grads[i].node_id for i in selected]
        rejected = [
            g.node_id for i, g in enumerate(valid_grads)
            if i not in selected
        ]

        for gid in rejected:
            if gid.startswith("adv_"):
                rejected_adversarial += 1
            else:
                rejected_honest += 1

        agg = AggregateGradient(
            node_id=self.node_id,
            round=round_num,
            gradient=valid_grads[selected[0]],
            accepted_devices=accepted,
            rejected_devices=rejected,
            total_adversarial=total_adversarial,
            rejected_adversarial=rejected_adversarial,
            rejected_honest=rejected_honest,
        )
        return agg
