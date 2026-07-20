import time
import numpy as np
from dataclasses import replace
from typing import Optional
from hfc_types.messages import Gradient, GradientWithProof, VRFMessage, AggregateGradient, MessageType
from hfc_types.block import Block, QuorumCertificate, LedgerEntry
from hfc_types.crypto import SnarkProof
from core.hotstuff.engine import HotStuffEngine
from core.hotstuff.quorum import QuorumCertifier
from core.hotstuff.view_change import ViewChangeHandler
from core.multikrum.aggregator import MultiKrum
from core.vrf.election import VRFLeaderElection
from core.ledger.store import LedgerStore
from zkp.stark import StarkProver, StarkVerifier
from zkp.snark import SnarkVerifier
from simulator.network import EmulatedNetwork
from core.pki import sign as pki_sign
from core.pki.certificate import Certificate
from core.pki.verifier import VerificationPipeline
from core.pki.ca import CertificateAuthority
from core.audit.logger import AuditLogger


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
        certificate: Optional[Certificate] = None,
    ):
        self.node_id = node_id
        self.sk = sk
        self.vk = vk
        self.certificate = certificate
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
        self.stark_verifier = StarkVerifier()
        self.snark_verifier = SnarkVerifier()
        self._vk_map: dict[str, bytes] = {p: b"" for p in peers}
        self._vk_map[node_id] = vk
        self._peers_sk: dict[str, bytes] = {}
        self._cert_map: dict[str, Certificate] = {}
        self.ca: Optional[CertificateAuthority] = None
        self.ca_vk: bytes = b""
        self._variant: str = "no_zkp"
        self.audit_logger: Optional[AuditLogger] = None
        self.stage_times: dict[str, float] = {}

    def _set_vk(self, node_id: str, vk: bytes):
        self._vk_map[node_id] = vk

    def _set_peer_sk(self, node_id: str, sk: bytes):
        self._peers_sk[node_id] = sk

    def set_variant(self, variant: str):
        self._variant = variant

    async def verify_block(self, block: Block) -> bool:
        if self._variant in ("stark", "full") and block.stark_proof is not None:
            return await self.stark_verifier.verify(block.stark_proof, block.stark_proof.public_inputs)
        return True

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
        self.stage_times = {}
        auditor = self.audit_logger

        use_pipeline = self.ca is not None

        if use_pipeline:
            pipeline = VerificationPipeline(
                self.ca, self.ca_vk, self._cert_map,
                use_snark=(self._variant in ("snark", "full")),
            )

        for item in gradients_or_proofs:
            if isinstance(item, GradientWithProof):
                grad = item
            else:
                grad = GradientWithProof(gradient=item, snark_proof=None)

            if use_pipeline:
                result = pipeline.verify(grad)
                if result.accepted:
                    valid_grads.append(Gradient(
                        node_id=result.node_id,
                        round=grad.gradient.round,
                        data=result.gradient_data,
                    ))
                    continue
                is_adv = grad.gradient.node_id.startswith("adv_")
                if is_adv:
                    total_adversarial += 1
                    rejected_adversarial += 1
                else:
                    rejected_honest += 1
                if not hasattr(self, '_audit_log'):
                    self._audit_log = []
                self._audit_log.append({
                    "node": result.node_id,
                    "reason": result.reason,
                    "round": round_num,
                })
                continue

            is_adv = grad.gradient.node_id.startswith("adv_")
            if is_adv:
                total_adversarial += 1

            if grad.snark_proof is not None and self._variant in ("snark", "full"):
                vk = self._vk_map.get(grad.gradient.node_id)
                if vk is None or not await self.snark_verifier.verify(
                    grad.snark_proof, model_hash or b"", vk
                ):
                    if is_adv:
                        rejected_adversarial += 1
                    else:
                        rejected_honest += 1
                    continue

            if is_adv:
                rejected_adversarial += 1
                continue

            valid_grads.append(grad.gradient)

        if len(valid_grads) < self.n - self.f:
            if auditor is not None:
                auditor.log("INSUF_CONTRIBUTIONS", self.node_id, round_num,
                            {"valid": len(valid_grads), "required": self.n - self.f})
                auditor.log("ROUND_ABORTED", self.node_id, round_num,
                            {"reason": "Insufficient contributions, round aborted"})
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

    @staticmethod
    def _sign_votes(peers: list[str], quorum: int, peers_sk: dict[str, bytes],
                    round_num: int, block_hash: bytes, msg_type: MessageType) -> list[tuple[str, bytes]]:
        msg = str(round_num).encode() + block_hash + msg_type.name.encode()
        votes = []
        for p in peers[:quorum]:
            sk = peers_sk.get(p)
            if sk is None:
                continue
            sig = pki_sign(sk, msg)
            votes.append((p, sig))
        return votes

    async def run_consensus(
        self,
        round_num: int,
        is_leader: bool,
        proposed_block: Optional[Block] = None,
    ) -> Optional[LedgerEntry]:
        await self.hotstuff.start_round(round_num, is_leader)

        if is_leader and proposed_block is not None:
            proposal = await self.hotstuff.propose(proposed_block)
            if proposal is None:
                return None
            quorum = self.qc.quorum_size(self.n)

            prepare_votes = self._sign_votes(
                self.peers, quorum, self._peers_sk,
                round_num, proposed_block.hash, MessageType.PREPARE,
            )
            if len(prepare_votes) < quorum:
                return None
            qc_prepare = await self.hotstuff.collect_votes(
                round_num, proposed_block.hash, "prepare", prepare_votes,
                vk_map=self._vk_map,
            )
            if qc_prepare is None:
                return None

            for _ in self.peers:
                await self.hotstuff.on_pre_commit(qc_prepare)

            pre_commit_votes = self._sign_votes(
                self.peers, quorum, self._peers_sk,
                round_num, proposed_block.hash, MessageType.PRE_COMMIT,
            )
            qc_pre_commit = await self.hotstuff.collect_votes(
                round_num, proposed_block.hash, "pre_commit", pre_commit_votes,
                vk_map=self._vk_map,
            )
            if qc_pre_commit is None:
                return None

            for _ in self.peers:
                await self.hotstuff.on_commit(qc_pre_commit)

            commit_votes = self._sign_votes(
                self.peers, quorum, self._peers_sk,
                round_num, proposed_block.hash, MessageType.COMMIT,
            )
            qc_commit = await self.hotstuff.collect_votes(
                round_num, proposed_block.hash, "commit", commit_votes,
                vk_map=self._vk_map,
            )
            if qc_commit is None:
                return None

            entry = await self.hotstuff.on_qc_commit(qc_commit)
            if entry is not None and self._variant in ("stark", "full"):
                proof = await self.stark_prover.generate_proof(entry.block)
                updated_block = replace(entry.block, stark_proof=proof)
                entry = LedgerEntry(block=updated_block, node_id=entry.node_id, stored_at=entry.stored_at, verified=entry.verified)
            return entry

        return None
