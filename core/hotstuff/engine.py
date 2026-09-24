from dataclasses import replace
from typing import Optional
from hfc_types.block import Block, QuorumCertificate, LedgerEntry
from hfc_types.messages import MessageType
from core.hotstuff.quorum import QuorumCertifier
from core.hotstuff.messages import PrepareProposal, VoteMessage
from core.hotstuff.view_change import ViewChangeHandler
from core.pki import sign as pki_sign


class HotStuffEngine:
    def __init__(
        self,
        node_id: str,
        sk: bytes,
        vk: bytes,
        peers: list[str],
        n: int,
        f: int,
        view_change_handler: Optional[ViewChangeHandler] = None,
    ):
        self.node_id = node_id
        self.sk = sk
        self.vk = vk
        self.peers = peers
        self.n = n
        self.f = f
        self.quorum_certifier = QuorumCertifier()
        self.view_change_handler = view_change_handler

        self.round = 0
        self.current_view = 1
        self.state = "IDLE"
        self._voted_prepare: set[int] = set()
        self._voted_pre_commit: set[int] = set()
        self._voted_commit: set[int] = set()
        self._last_proposal: Optional[Block] = None

    async def start_round(self, round_num: int, is_leader: bool) -> None:
        self.round = round_num
        if is_leader:
            self.state = "READY_TO_PROPOSE"
        else:
            self.state = "READY_TO_VOTE"

    async def propose(self, block: Block) -> Optional[PrepareProposal]:
        if self.state != "READY_TO_PROPOSE":
            return None
        self._last_proposal = block
        self.state = "VOTED_PREPARE"
        return PrepareProposal(
            leader_id=self.node_id,
            round=self.round,
            block=block,
        )

    async def on_prepare(self, proposal: Block) -> Optional[VoteMessage]:
        if proposal.round in self._voted_prepare:
            return None
        self._voted_prepare.add(proposal.round)
        self._last_proposal = proposal
        msg = str(self.round).encode() + proposal.hash + MessageType.PREPARE.name.encode()
        sig = pki_sign(self.sk, msg)
        return VoteMessage(
            node_id=self.node_id,
            round=proposal.round,
            phase="prepare",
            block_hash=proposal.hash,
            signature=sig,
        )

    async def on_pre_commit(self, qc: QuorumCertificate) -> Optional[VoteMessage]:
        if qc.round in self._voted_pre_commit:
            return None
        self._voted_pre_commit.add(qc.round)
        msg = str(self.round).encode() + qc.block_hash + MessageType.PRE_COMMIT.name.encode()
        sig = pki_sign(self.sk, msg)
        return VoteMessage(
            node_id=self.node_id,
            round=qc.round,
            phase="pre_commit",
            block_hash=qc.block_hash,
            signature=sig,
        )

    async def on_commit(self, qc: QuorumCertificate) -> Optional[VoteMessage]:
        if qc.round in self._voted_commit:
            return None
        self._voted_commit.add(qc.round)
        msg = str(self.round).encode() + qc.block_hash + MessageType.COMMIT.name.encode()
        sig = pki_sign(self.sk, msg)
        return VoteMessage(
            node_id=self.node_id,
            round=qc.round,
            phase="commit",
            block_hash=qc.block_hash,
            signature=sig,
        )

    async def on_qc_commit(self, qc: QuorumCertificate) -> Optional[LedgerEntry]:
        self.state = "DECIDED"
        if self._last_proposal is not None:
            block = replace(self._last_proposal, qc_commit=qc)
        else:
            block = Block(
                round=qc.round,
                gradient_hash=qc.block_hash,
                qc_commit=qc,
                stark_proof=None,
                accepted_devices=[],
                rejected_devices=[],
                timestamp=0.0,
                prev_hash=b"\x00" * 32,
            )
        return LedgerEntry(
            block=block,
            node_id=self.node_id,
            stored_at=0.0,
        )

    async def collect_votes(
        self,
        round: int,
        block_hash: bytes,
        phase: str,
        votes: list[tuple[str, bytes]],
        vk_map: Optional[dict[str, bytes]] = None,
    ) -> Optional[QuorumCertificate]:
        phase_map = {
            "prepare": MessageType.PREPARE,
            "pre_commit": MessageType.PRE_COMMIT,
            "commit": MessageType.COMMIT,
        }
        msg_type = phase_map.get(phase)
        if msg_type is None:
            return None
        quorum = self.quorum_certifier.quorum_size(self.n)
        if len(votes) < quorum:
            return None
        try:
            return self.quorum_certifier.collect(
                round=round,
                block_hash=block_hash,
                msg_type=msg_type,
                signatures=votes,
                quorum_size=quorum,
                vk_map=vk_map,
            )
        except ValueError:
            return None

    async def handle_timeout(self) -> bool:
        if self.view_change_handler is None:
            return False
        if not self.view_change_handler.should_change_view(timeout=True):
            return False
        self.state = "VIEW_CHANGE"
        vc_msg = self.view_change_handler.create_view_change(
            self.node_id, self.current_view + 1, self.sk
        )
        return True
