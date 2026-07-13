from typing import Optional
from hfc_types.block import Block, QuorumCertificate, LedgerEntry
from hfc_types.messages import Vote, MessageType


class HotStuffEngine:
    def __init__(
        self,
        node_id: str,
        sk: bytes,
        vk: bytes,
        peers: list[str],
        n: int,
        f: int,
    ):
        self.node_id = node_id
        self.sk = sk
        self.vk = vk
        self.peers = peers
        self.n = n
        self.f = f
        self._round_voted: set[int] = set()

    async def on_prepare(self, proposal: Block) -> Optional[Vote]:
        if proposal.round in self._round_voted:
            return None
        self._round_voted.add(proposal.round)
        return Vote(
            node_id=self.node_id,
            round=proposal.round,
            msg_type=MessageType.PREPARE,
            block_hash=proposal.hash,
        )

    async def on_pre_commit(
        self, qc: QuorumCertificate
    ) -> Optional[Vote]:
        return Vote(
            node_id=self.node_id,
            round=qc.round,
            msg_type=MessageType.PRE_COMMIT,
            block_hash=qc.block_hash,
        )

    async def on_commit(self, qc: QuorumCertificate) -> Optional[Vote]:
        return Vote(
            node_id=self.node_id,
            round=qc.round,
            msg_type=MessageType.COMMIT,
            block_hash=qc.block_hash,
        )

    async def on_qc_commit(
        self, qc: QuorumCertificate
    ) -> Optional[LedgerEntry]:
        return LedgerEntry(
            block=Block(
                round=qc.round,
                gradient_hash=qc.block_hash,
                qc_commit=qc,
                stark_proof=None,
                accepted_devices=[],
                rejected_devices=[],
                timestamp=0.0,
                prev_hash=b"\x00" * 32,
            ),
            node_id=self.node_id,
            stored_at=0.0,
        )
