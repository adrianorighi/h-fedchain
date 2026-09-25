import logging
from dataclasses import replace
from typing import Optional
from hfc_types.block import Block, QuorumCertificate, LedgerEntry
from hfc_types.messages import MessageType
from core.hotstuff.quorum import QuorumCertifier
from core.hotstuff.messages import PrepareProposal, VoteMessage
from core.hotstuff.view_change import ViewChangeHandler
from core.pki import sign as pki_sign

logger = logging.getLogger(__name__)


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
        self._voted_prepare: dict[int, bytes] = {}
        self._voted_pre_commit: dict[int, bytes] = {}
        self._voted_commit: dict[int, bytes] = {}
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

    @staticmethod
    def _record_vote(
        voted: dict[int, bytes], round_num: int, block_hash: bytes
    ) -> bool:
        previous = voted.get(round_num)
        if previous is not None and previous != block_hash:
            return False
        voted[round_num] = block_hash
        return True

    async def on_prepare(self, proposal: Block) -> Optional[VoteMessage]:
        if not self._record_vote(
            self._voted_prepare, proposal.round, proposal.hash
        ):
            return None
        self._last_proposal = proposal
        msg = (
            str(proposal.round).encode()
            + proposal.hash
            + MessageType.PREPARE.name.encode()
        )
        sig = pki_sign(self.sk, msg)
        return VoteMessage(
            node_id=self.node_id,
            round=proposal.round,
            phase="prepare",
            block_hash=proposal.hash,
            signature=sig,
        )

    async def on_pre_commit(
        self, qc: QuorumCertificate, vk_map: Optional[dict] = None
    ) -> Optional[VoteMessage]:
        if vk_map is not None:
            quorum = self.quorum_certifier.quorum_size(self.n)
            if qc.msg_type is not MessageType.PREPARE:
                logger.warning(
                    "Rejected pre_commit vote for round %d: QC phase is %s, "
                    "expected PREPARE", qc.round, qc.msg_type.name,
                )
                return None
            if not qc.is_valid(quorum, vk_map):
                logger.warning(
                    "Rejected pre_commit vote for round %d: QC invalid under "
                    "vk_map (%d signatures, quorum %d)",
                    qc.round, len(qc.signatures), quorum,
                )
                return None
        if not self._record_vote(
            self._voted_pre_commit, qc.round, qc.block_hash
        ):
            return None
        msg = (
            str(qc.round).encode()
            + qc.block_hash
            + MessageType.PRE_COMMIT.name.encode()
        )
        sig = pki_sign(self.sk, msg)
        return VoteMessage(
            node_id=self.node_id,
            round=qc.round,
            phase="pre_commit",
            block_hash=qc.block_hash,
            signature=sig,
        )

    async def on_commit(
        self, qc: QuorumCertificate, vk_map: Optional[dict] = None
    ) -> Optional[VoteMessage]:
        if vk_map is not None:
            quorum = self.quorum_certifier.quorum_size(self.n)
            if qc.msg_type is not MessageType.PRE_COMMIT:
                logger.warning(
                    "Rejected commit vote for round %d: QC phase is %s, "
                    "expected PRE_COMMIT", qc.round, qc.msg_type.name,
                )
                return None
            if not qc.is_valid(quorum, vk_map):
                logger.warning(
                    "Rejected commit vote for round %d: QC invalid under "
                    "vk_map (%d signatures, quorum %d)",
                    qc.round, len(qc.signatures), quorum,
                )
                return None
        if not self._record_vote(self._voted_commit, qc.round, qc.block_hash):
            return None
        msg = (
            str(qc.round).encode()
            + qc.block_hash
            + MessageType.COMMIT.name.encode()
        )
        sig = pki_sign(self.sk, msg)
        return VoteMessage(
            node_id=self.node_id,
            round=qc.round,
            phase="commit",
            block_hash=qc.block_hash,
            signature=sig,
        )

    async def on_qc_commit(
        self, qc: QuorumCertificate, vk_map: Optional[dict] = None
    ) -> Optional[LedgerEntry]:
        if qc.msg_type is not MessageType.COMMIT:
            logger.warning(
                "Rejected QC commit for round %d: certificate phase is %s, "
                "expected COMMIT", qc.round, qc.msg_type.name,
            )
            return None
        if self._last_proposal is not None:
            if (
                qc.block_hash != self._last_proposal.hash
                or qc.round != self._last_proposal.round
            ):
                logger.warning(
                    "Rejected QC commit for round %d: QC does not match last "
                    "proposal (qc round %d, qc hash %s, proposal hash %s)",
                    qc.round, qc.round, qc.block_hash.hex(),
                    self._last_proposal.hash.hex(),
                )
                return None
        if vk_map is not None:
            quorum = self.quorum_certifier.quorum_size(self.n)
            if not qc.is_valid(quorum, vk_map):
                logger.warning(
                    "Rejected QC commit for round %d: QC invalid under vk_map "
                    "(%d signatures, quorum %d)",
                    qc.round, len(qc.signatures), quorum,
                )
                return None
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
            self.node_id, self.current_view + 1, self.sk, round=self.round
        )
        return True
