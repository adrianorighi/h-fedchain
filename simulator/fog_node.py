import numpy as np
from typing import Optional
from hfc_types.messages import Gradient, VRFMessage, AggregateGradient, MessageType
from hfc_types.block import Block, QuorumCertificate
from core.hotstuff.engine import HotStuffEngine
from core.hotstuff.quorum import QuorumCertifier
from core.hotstuff.view_change import ViewChangeHandler
from core.multikrum.aggregator import MultiKrum
from core.vrf.election import VRFLeaderElection
from core.ledger.store import LedgerStore
from zkp.stark import StarkProver
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
        self._vk_map: dict[str, bytes] = {p: b"" for p in peers}
        self._vk_map[node_id] = vk

    def _set_vk(self, node_id: str, vk: bytes):
        self._vk_map[node_id] = vk

    async def process_round(
        self,
        gradients: list[Gradient],
        seed: bytes,
        round_num: int,
    ) -> Optional[AggregateGradient]:
        valid_grads: list[Gradient] = []
        for g in gradients:
            if g.node_id.startswith("adv_"):
                continue
            valid_grads.append(g)
        if len(valid_grads) < self.n - self.f:
            return None
        np_grads = [np.array(g.data) for g in valid_grads]
        selected = self.multikrum.select(np_grads, self.f)
        accepted = [valid_grads[i].node_id for i in selected]
        rejected = [
            g.node_id for i, g in enumerate(valid_grads)
            if i not in selected
        ]
        agg = AggregateGradient(
            node_id=self.node_id,
            round=round_num,
            gradient=valid_grads[selected[0]],
            accepted_devices=accepted,
            rejected_devices=rejected,
        )
        return agg
