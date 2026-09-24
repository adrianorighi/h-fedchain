import time
import numpy as np
from dataclasses import replace
from typing import Optional
from hfc_types.messages import Gradient, GradientWithProof, VRFMessage, AggregateGradient
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
from core.pki.certificate import Certificate
from core.pki.verifier import VerificationPipeline
from core.pki.ca import CertificateAuthority
from core.audit.logger import AuditLogger
from monitoring.tracer import Tracer


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
        self.view_change = ViewChangeHandler(n, f)
        self.hotstuff = HotStuffEngine(node_id, sk, vk, peers, n, f,
                                        view_change_handler=self.view_change)
        self.vrf = VRFLeaderElection()
        self.multikrum = MultiKrum()
        self.ledger = LedgerStore()
        self.qc = QuorumCertifier()
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
        self.tracer: Optional[Tracer] = None

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
        verify_snark: bool = True,
    ) -> Optional[AggregateGradient]:
        tracer = self.tracer
        if tracer is not None:
            ctx = tracer.span("process_round", "fog", self.node_id, round_num)
            ctx.__enter__()
        else:
            ctx = None
        try:
            return await self._process_round_impl(
                gradients_or_proofs, seed, round_num, model_hash,
                verify_snark=verify_snark,
            )
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    async def _process_round_impl(
        self,
        gradients_or_proofs: list,
        seed: bytes,
        round_num: int,
        model_hash: Optional[bytes] = None,
        verify_snark: bool = True,
    ) -> Optional[AggregateGradient]:
        valid_grads: list[Gradient] = []
        total_adversarial = 0
        rejected_adversarial = 0
        rejected_honest = 0
        snark_attempted = 0
        snark_passed = 0
        snark_failed = 0
        self.stage_times = {}
        auditor = self.audit_logger

        use_pipeline = self.ca is not None

        t_verify_start = time.perf_counter()

        if use_pipeline:
            pipeline = VerificationPipeline(
                self.ca, self.ca_vk, self._cert_map,
                use_snark=(self._variant in ("snark", "full") and verify_snark),
            )

        for item in gradients_or_proofs:
            if isinstance(item, GradientWithProof):
                grad = item
            else:
                grad = GradientWithProof(gradient=item, snark_proof=None)

            if use_pipeline:
                result = pipeline.verify(grad, model_hash=model_hash)
                had_snark = (
                    grad.snark_proof is not None
                    and self._variant in ("snark", "full")
                    and verify_snark
                )
                if had_snark:
                    snark_attempted += 1
                    if result.accepted and result.snark_ok:
                        snark_passed += 1
                    else:
                        snark_failed += 1
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
                if auditor is not None:
                    event_type = "REJ_PKI" if "PKI" in (result.reason or "").upper() else "REJ_ZKP"
                    auditor.log(event_type, result.node_id, round_num,
                                {"reason": result.reason, "is_adversarial": is_adv})
                continue

            is_adv = grad.gradient.node_id.startswith("adv_")
            if is_adv:
                total_adversarial += 1

            had_snark = (
                grad.snark_proof is not None
                and self._variant in ("snark", "full")
                and verify_snark
            )
            if had_snark:
                snark_attempted += 1
                if verify_snark:
                    vk = self._vk_map.get(grad.gradient.node_id)
                    if vk is None or not await self.snark_verifier.verify(
                        grad.snark_proof, model_hash or b"", vk
                    ):
                        snark_failed += 1
                        if is_adv:
                            rejected_adversarial += 1
                        else:
                            rejected_honest += 1
                        if auditor is not None:
                            auditor.log("REJ_ZKP", grad.gradient.node_id, round_num,
                                        {"reason": "snark_verify_failed"})
                        continue
                snark_passed += 1

            if is_adv:
                rejected_adversarial += 1
                continue

            valid_grads.append(grad.gradient)

        self.stage_times["verify"] = (time.perf_counter() - t_verify_start) * 1000

        if len(valid_grads) < self.n - self.f:
            if auditor is not None:
                auditor.log("INSUF_CONTRIBUTIONS", self.node_id, round_num,
                            {"valid": len(valid_grads), "required": self.n - self.f})
                auditor.log("ROUND_ABORTED", self.node_id, round_num,
                            {"reason": "Insufficient contributions, round aborted"})
            return None

        t_krum_start = time.perf_counter()
        np_grads = [np.array(g.data) for g in valid_grads]
        selected = self.multikrum.select(np_grads, self.f)
        self.stage_times["multikrum"] = (time.perf_counter() - t_krum_start) * 1000

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
            if auditor is not None:
                auditor.log("REJ_MULTIKRUM", gid, round_num, {})

        agg = AggregateGradient(
            node_id=self.node_id,
            round=round_num,
            gradient=valid_grads[selected[0]],
            accepted_devices=accepted,
            rejected_devices=rejected,
            total_adversarial=total_adversarial,
            rejected_adversarial=rejected_adversarial,
            rejected_honest=rejected_honest,
            snark_attempted=snark_attempted,
            snark_passed=snark_passed,
            snark_failed=snark_failed,
        )
        return agg

    async def finalize_commit(self, qc_commit: QuorumCertificate, block: Block) -> Optional[LedgerEntry]:
        t_stark_start = time.perf_counter()
        self.hotstuff._last_proposal = block
        entry = await self.hotstuff.on_qc_commit(qc_commit)
        if entry is not None and self._variant in ("stark", "full"):
            proof = await self.stark_prover.generate_proof(entry.block)
            updated_block = replace(entry.block, stark_proof=proof)
            entry = LedgerEntry(block=updated_block, node_id=entry.node_id,
                                stored_at=entry.stored_at, verified=entry.verified)
        self.stage_times["stark_gen"] = (time.perf_counter() - t_stark_start) * 1000
        return entry
