import asyncio
import pickle
import random
import statistics
import time
import numpy as np
from hashlib import sha256
from typing import Optional
from hfc_types.messages import Gradient, GradientWithProof, VRFMessage, AggregateGradient, MessageType
from hfc_types.block import Block, QuorumCertificate
from simulator.network import EmulatedNetwork
from simulator.fog_node import FogNode
from monitoring.tracer import Tracer
from monitoring.system_metrics import SystemMetrics
from core.pki import generate_keypair, CertificateAuthority
from core.pki.certificate import Certificate
from core.multikrum.aggregator import MultiKrum
from core.audit.logger import AuditLogger
from dataset.loader import PTBXLLoader
from dataset.partitioner import DirichletPartitioner
from dataset.edge_worker import EdgeWorker


def select_adversarial_ids(total: int, ratio: float, seed: int) -> set[int]:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"adversarial ratio must be in [0, 1], got {ratio}")
    n = int(total * ratio)
    if n <= 0:
        return set()
    return set(random.Random(seed).sample(range(total), n))


class ExperimentResult:
    def __init__(self):
        self.round_metrics: list[dict] = []


class Orchestrator:
    def __init__(
        self,
        num_clusters: int = 1,
        nodes_per_cluster: int = 5,
        devices_per_cluster: int = 20,
        f: int = 1,
        latency_ms: float = 10.0,
        cloud_latency_ms: float = 50.0,
        adversarial_ratio: float = 0.0,
        adversarial_seed: int = 42,
        use_dataset: bool = False,
        dataset_max_records: int = 500,
        dirichlet_alpha: float = 0.5,
        variant: str = "no_zkp",
        tracer: Optional["Tracer"] = None,
        snark_prove: bool = False,
        snark_sample_verify: bool = True,
    ):
        self.num_clusters = num_clusters
        self.nodes_per_cluster = nodes_per_cluster
        self.devices_per_cluster = devices_per_cluster
        self.f = f
        self.latency_ms = latency_ms
        self.cloud_latency_ms = cloud_latency_ms
        self.adversarial_ratio = adversarial_ratio
        self.adversarial_seed = adversarial_seed
        total_devices = devices_per_cluster * num_clusters
        self._adv_ids = select_adversarial_ids(
            total_devices, adversarial_ratio, adversarial_seed
        )
        self.use_dataset = use_dataset
        self.dataset_max_records = dataset_max_records
        self.dirichlet_alpha = dirichlet_alpha
        self.variant = variant
        self.snark_prove = snark_prove
        self.snark_sample_verify = snark_sample_verify
        self.tracer = tracer
        self.nodes: list[FogNode] = []
        self.network = EmulatedNetwork(latency_ms, cloud_latency_ms)
        self.result = ExperimentResult()
        self.edge_workers: list[EdgeWorker] = []
        self._global_weights: dict = {}
        self._last_delta_w_reg: list[float] | None = None
        self._certificates: dict[str, Certificate] = {}
        self._vrf_candidates: list[VRFMessage] = []
        self._snark_sample_result: Optional[bool] = True
        self._pending_snark_verify = None
        self._snark_verify_fired = False
        self.audit_logger = AuditLogger()
        self._sys = SystemMetrics()

    def setup(self):
        ca_sk, ca_vk = generate_keypair()
        self.ca = CertificateAuthority("ca_root", ca_sk, ca_vk)
        self.ca_vk = ca_vk

        for i in range(self.nodes_per_cluster):
            nid = f"n{i}"
            sk, vk = generate_keypair()
            peers = [f"n{j}" for j in range(self.nodes_per_cluster)]
            self.network.add_node(nid)
            node = FogNode(
                node_id=nid,
                sk=sk,
                vk=vk,
                peers=peers,
                n=self.nodes_per_cluster,
                f=self.f,
                network=self.network,
            )
            node.set_variant(self.variant)
            cert = self.ca.issue_certificate(nid, vk)
            node.certificate = cert
            self._certificates[nid] = cert
            self.nodes.append(node)
        for node in self.nodes:
            for other in self.nodes:
                node._set_vk(other.node_id, other.vk)
                node._set_peer_sk(other.node_id, other.sk)
            node.ca = self.ca
            node.ca_vk = self.ca_vk
            node._cert_map = self._certificates
            node.audit_logger = self.audit_logger
            if self.tracer is not None:
                node.tracer = self.tracer

        if self.use_dataset:
            self._init_dataset()

    def _init_dataset(self):
        loader = PTBXLLoader(max_records=self.dataset_max_records)
        if loader.is_available():
            data, labels = loader.load()
        else:
            data, labels = loader.generate_synthetic(self.dataset_max_records)

        num_devices = self.devices_per_cluster * self.num_clusters
        partitioner = DirichletPartitioner(alpha=self.dirichlet_alpha, seed=42)
        assignments = partitioner.assign(
            num_devices=num_devices,
            labels=labels,
            num_classes=5,
        )

        use_snark = self.variant in ("snark", "full")
        self.edge_workers = []
        for i, indices in enumerate(assignments):
            if not indices:
                continue
            is_adv = i in self._adv_ids
            worker = EdgeWorker(
                device_id=f"d{i}",
                indices=indices,
                all_data=data,
                all_labels=labels,
                is_adversarial=is_adv,
                attack_type="label_flip",
                use_snark=use_snark,
            )
            self.edge_workers.append(worker)
            if use_snark:
                for node in self.nodes:
                    node._set_vk(worker.device_id, worker.vk)

        for worker in self.edge_workers:
            vk = getattr(worker, 'vk', None)
            if vk:
                cert = self.ca.issue_certificate(worker.device_id, vk)
                self._certificates[worker.device_id] = cert
                worker.certificate = cert
                worker.ca_vk = self.ca.vk

        self._global_weights = self.edge_workers[0].model.get_weights()

    async def _generate_gradients(
        self, round_num: int
    ) -> list[GradientWithProof]:
        if self.use_dataset:
            return self._generate_real_gradients(round_num)

        from core.pki import generate_keypair, sign as pki_sign

        grads: list[GradientWithProof] = []
        sk_map: dict[int, bytes] = {}
        for d in range(self.devices_per_cluster * self.num_clusters):
            is_adv = d in self._adv_ids
            data = (
                np.random.randn(10).tolist()
                if not is_adv
                else [100.0 * float(np.random.randn()) for _ in range(10)]
            )
            gid = f"adv_{d}" if is_adv else f"d{d}"
            if is_adv:
                fake_sk, _ = generate_keypair()
                sig = pki_sign(fake_sk, f"{gid}:{round_num}".encode())
            else:
                device_sk, device_vk = generate_keypair()
                cert = self.ca.issue_certificate(gid, device_vk)
                self._certificates[gid] = cert
                sig = pki_sign(device_sk, f"{gid}:{round_num}".encode())
                sk_map[d] = device_sk
            grads.append(GradientWithProof(
                gradient=Gradient(node_id=gid, round=round_num, data=data, signature=sig),
            ))
        if self._want_snark_proofs():
            await self._attach_snark_proofs(grads, sk_map)
        return grads

    def _want_snark_proofs(self) -> bool:
        return self.snark_prove and self.variant in ("snark", "full")

    async def _attach_snark_proofs(
        self, grads: list[GradientWithProof], sk_map: dict[int, bytes]
    ):
        from simulator.snark_worker import get_pool, prove_timed_sync

        loop = asyncio.get_event_loop()
        model_hash = b""
        tasks = {
            d: loop.run_in_executor(
                get_pool(), prove_timed_sync, grads[d].gradient, model_hash, sk,
            )
            for d, sk in sk_map.items()
        }
        for d, fut in tasks.items():
            proof, prove_cpu_ms = await fut
            grads[d].snark_proof = proof
            grads[d].prove_cpu_ms = prove_cpu_ms
            grads[d].snark_proof_size = len(proof.proof_bytes)

    def _generate_real_gradients(self, round_num: int) -> list[GradientWithProof]:
        grads = []
        for worker in self.edge_workers:
            gwp = worker.train_round(
                self._global_weights,
                round_num=round_num,
            )
            if worker.is_adversarial:
                dev_idx = int(worker.device_id[1:])
                gwp.gradient.node_id = f"adv_{dev_idx}"
            grads.append(gwp)
        return grads

    def _elect_leader(self, seed: bytes) -> tuple[str, list[VRFMessage]]:
        vrf = self.nodes[0].vrf
        vk_map = {n.node_id: n.vk for n in self.nodes}
        candidates = []
        for n in self.nodes:
            y, proof = vrf.evaluate(n.sk, seed)
            candidates.append(VRFMessage(
                node_id=n.node_id,
                round=n.hotstuff.round or 0,
                y=y,
                proof=proof,
            ))
        valid: list[VRFMessage] = []
        for c in candidates:
            vk = vk_map.get(c.node_id)
            if vk is not None and vrf.verify(vk, seed, c.y, c.proof):
                valid.append(c)
        valid.sort(key=lambda c: c.y)
        if not valid:
            raise ValueError("No valid VRF candidates")
        self._vrf_candidates = valid
        return valid[0].node_id, valid

    def _build_metrics(self, round_num, leader_id, t_start, t_end, block, grads, cluster_results,
                       snark_sample_ok: Optional[bool] = None,
                       snark_sampled_verified: bool = False) -> dict:
        total_dev = self.devices_per_cluster * self.num_clusters
        if self.use_dataset and self.edge_workers:
            # Devices with empty partitions are skipped in _init_dataset,
            # so ground truth must count actual workers, not nominal total.
            num_adv = sum(1 for w in self.edge_workers if w.is_adversarial)
            num_hon = len(self.edge_workers) - num_adv
        else:
            num_adv = len(self._adv_ids)
            num_hon = total_dev - num_adv
        # all fog nodes process the same grads; take max to avoid 5× sum
        rej_adv = max((r.rejected_adversarial for r in cluster_results), default=0)
        rej_hon = max((r.rejected_honest for r in cluster_results), default=0)

        leader_node = next(n for n in self.nodes if n.node_id == leader_id)

        vc_count = 0
        vc_latency = 0.0
        if leader_node.audit_logger:
            vc_count = leader_node.audit_logger.count_by_type("VIEW_CHANGE")
            if leader_node.view_change._vc_durations:
                vc_latency = statistics.mean(leader_node.view_change._vc_durations) * 1000

        snark_proofs_total = sum(1 for g in grads if g.snark_proof is not None)
        if snark_sampled_verified:
            snark_attempted = snark_proofs_total if self.variant in ("snark", "full") else 0
            snark_passed = snark_attempted if snark_sample_ok and snark_attempted > 0 else 0
        else:
            snark_attempted = sum(r.snark_attempted for r in cluster_results)
            snark_passed = sum(r.snark_passed for r in cluster_results)
        snark_failed = snark_attempted - snark_passed
        from simulator.snark_worker import SNARK_VERIFY_MS
        snark_verify_projected_ms = snark_proofs_total * SNARK_VERIFY_MS

        state_divergence = False
        if len(self.nodes) > 1:
            h0 = self.nodes[0].ledger.get_height()
            for n in self.nodes[1:]:
                if n.ledger.get_height() != h0:
                    state_divergence = True
                    break
            if not state_divergence and h0 > 0:
                for n in self.nodes[1:]:
                    for i in range(h0):
                        if n.ledger._entries[i].block.hash != self.nodes[0].ledger._entries[i].block.hash:
                            state_divergence = True
                            break
                    if state_divergence:
                        break

        ledger_integrity = all(n.ledger.verify_chain() for n in self.nodes)

        # Communication overhead (bytes) — Edge->Fog + inter-node + block propagation.
        # comm_overhead_bytes é mode-dependent (single=total completo,
        # multi=parcela cloud); network_bytes é o total unificado.
        edge_fog_bytes = sum(len(pickle.dumps(g)) for g in grads)
        inter_node_bytes = sum(len(pickle.dumps(r)) for r in cluster_results)
        block_bytes = len(pickle.dumps(block))
        comm_overhead = edge_fog_bytes + inter_node_bytes + block_bytes

        # Compute loss proxy = avg grad norm across valid gradients
        valid_grads_list = [
            np.array(g.data) for cr in cluster_results
            for g in [cr.gradient] if g.data is not None
        ]
        loss = float(np.mean([np.linalg.norm(g) for g in valid_grads_list])) if valid_grads_list else 0.0

        # Proof CPU / size. SNARK prove é medido com process_time na Edge
        # (prove_cpu_ms); stark_gen reutiliza a stage existente medida com
        # perf_counter (wall-proxy, CPU-bound perto de wall — escolha
        # documentada para não re-medir o estágio já instrumentado).
        snark_gen_ms = sum(getattr(g, "prove_cpu_ms", 0.0) for g in grads)
        stark_gen_ms = leader_node.stage_times.get("stark_gen", 0.0)
        proof_gen_cpu_ms = snark_gen_ms + stark_gen_ms

        # Rodadas single-cluster reportam 0.0 em proof_verify_cpu_ms porque
        # FogNode.verify_block não é invocado no fluxo normal da rodada.
        proof_verify_cpu_ms = leader_node.stage_times.get("stark_verify", 0.0)

        snark_size = sum(getattr(g, "snark_proof_size", 0) for g in grads)
        stark_size = (
            len(block.stark_proof.proof_bytes)
            if block is not None and block.stark_proof else 0
        )
        proof_size_bytes = snark_size + stark_size

        sys_snap = self._sys.snapshot()

        return {
            "round": round_num,
            "leader": leader_id,
            "latency": t_end - t_start,
            "consensus_time_ms": (t_end - t_start) * 1000,
            "block_size_bytes": len(pickle.dumps(block)),
            "comm_overhead_bytes": comm_overhead,
            "bytes_edge_fog": edge_fog_bytes,
            "bytes_fog_intra": inter_node_bytes + block_bytes,
            "bytes_fog_inter": 0,
            "bytes_fog_cloud": 0,
            "network_bytes": edge_fog_bytes + inter_node_bytes + block_bytes,
            "num_accepted": len(block.accepted_devices),
            "num_rejected": len(block.rejected_devices),
            "ledger_height": self.nodes[0].ledger.get_height(),
            "qc_emitted": True,
            "num_adversarial": num_adv,
            "num_honest": num_hon,
            "rejected_adversarial": rej_adv,
            "falsely_rejected": rej_hon,
            "variant": self.variant,
            "view_change_count": vc_count,
            "view_change_latency_ms": vc_latency,
            "state_divergence": state_divergence,
            "ledger_integrity": ledger_integrity,
            "stage_time_ms": leader_node.stage_times,
            "snark_required": self.variant in ("snark", "full"),
            "snark_attempted": snark_attempted,
            "snark_passed": snark_passed,
            "snark_failed": snark_failed,
            "snark_proofs_total": snark_proofs_total,
            "snark_sampled_verified": 1 if snark_sampled_verified else 0,
            "snark_sampled_passed": int(bool(snark_sample_ok)) if snark_sampled_verified else 0,
            "snark_verify_projected_ms": snark_verify_projected_ms,
            "proof_gen_cpu_ms": proof_gen_cpu_ms,
            "proof_verify_cpu_ms": proof_verify_cpu_ms,
            "proof_size_bytes": proof_size_bytes,
            "stark_proof_generated": block.stark_proof is not None,
            "cpu_percent": sys_snap["cpu_percent"],
            "memory_rss_bytes": sys_snap["memory_rss_bytes"],
            "global_weights": self._global_weights.copy() if self._global_weights else {},
            "loss": loss,
            "vrf_candidates": [(c.node_id, c.y.hex()[:8]) for c in self._vrf_candidates] if hasattr(self, '_vrf_candidates') else [],
        }

    async def run_round(self, round_num: int, fire_snark_verify: bool = False) -> dict:
        tracer = self.tracer
        if tracer is not None:
            ctx = tracer.span("run_round", "orchestrator", round_num=round_num)
            ctx.__enter__()
        else:
            ctx = None
        try:
            return await self._run_round_impl(round_num, fire_snark_verify)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    async def _run_round_impl(self, round_num: int, fire_snark_verify: bool = False) -> dict:
        prev_hash = b"\x00" * 32
        if round_num > 0 and self.nodes[0].ledger.get_height() > 0:
            prev_hash = self.nodes[0].ledger._entries[-1].block.hash
        seed = prev_hash

        leader_id, self._vrf_candidates = self._elect_leader(seed)
        grads = await self._generate_gradients(round_num)

        model_hash = None
        if self.use_dataset and self._global_weights:
            model_hash = sha256(
                str(sorted(self._global_weights.items())).encode()
            ).digest()

        snark_sample_ok: Optional[bool] = self._snark_sample_result
        if self._pending_snark_verify is not None:
            fut = self._pending_snark_verify
            if fut.done():
                try:
                    self._snark_sample_result = bool(fut.result())
                except Exception:
                    self._snark_sample_result = False
                self._pending_snark_verify = None

        if (
            self._want_snark_proofs()
            and self.snark_sample_verify
            and not self._snark_verify_fired
            and fire_snark_verify
        ):
            from simulator.snark_worker import get_pool, verify_sync
            self._snark_verify_fired = True
            snark_proofs_total = sum(
                1 for g in grads if g.snark_proof is not None
            )
            if snark_proofs_total > 0:
                sample = next(g for g in grads if g.snark_proof is not None)
                loop = asyncio.get_event_loop()
                self._pending_snark_verify = loop.run_in_executor(
                    get_pool(), verify_sync,
                    sample.snark_proof, model_hash or b"", b"",
                )

        t_start = time.time()
        cluster_results: list[AggregateGradient] = []
        for node in self.nodes:
            result = await node.process_round(
                grads, seed, round_num, model_hash=model_hash,
                verify_snark=False,
            )
            if result:
                cluster_results.append(result)

        if not cluster_results:
            return {"round": round_num, "latency": 0, "qc_emitted": False,
                    "bytes_edge_fog": 0, "bytes_fog_intra": 0,
                    "bytes_fog_inter": 0, "bytes_fog_cloud": 0,
                    "network_bytes": 0,
                    "proof_gen_cpu_ms": 0.0, "proof_verify_cpu_ms": 0.0,
                    "proof_size_bytes": 0,
                    **self._sys.snapshot()}

        inter_node_selected: list[int] | None = None
        delta_w_reg: list[float] | None = None
        gradient_hash: bytes | None = None

        if len(cluster_results) >= 2:
            node_grads = [np.array(r.gradient.data) for r in cluster_results]
            if len(node_grads) > 3 * self.f:
                krum = MultiKrum()
                inter_node_selected = krum.select(node_grads, self.f)
                selected_grads = [node_grads[i] for i in inter_node_selected]
                delta_w_reg = np.mean(selected_grads, axis=0).tolist()
                gradient_hash = sha256(str(delta_w_reg).encode()).digest()
            else:
                delta_w_reg = np.mean(node_grads, axis=0).tolist()
                gradient_hash = sha256(str(delta_w_reg).encode()).digest()

        if gradient_hash is None:
            gradient_hash = seed

        # Aggregate accepted/rejected from actual Fog node Multi-Krum results
        accepted_devices = list(set(
            d for r in cluster_results for d in r.accepted_devices
        ))
        rejected_devices = list(set(
            d for r in cluster_results for d in r.rejected_devices
        ))

        proposed_block = Block(
            round=round_num, gradient_hash=gradient_hash,
            qc_commit=None, stark_proof=None,
            accepted_devices=accepted_devices,
            rejected_devices=rejected_devices,
            timestamp=t_start, prev_hash=prev_hash,
            n=self.nodes_per_cluster, f=self.f,
        )

        leader_node = next(n for n in self.nodes if n.node_id == leader_id)
        for node in self.nodes:
            await node.hotstuff.start_round(round_num, node.node_id == leader_id)

        proposal = await leader_node.hotstuff.propose(proposed_block)
        if proposal is None:
            return {"round": round_num, "latency": 0, "qc_emitted": False,
                    "bytes_edge_fog": 0, "bytes_fog_intra": 0,
                    "bytes_fog_inter": 0, "bytes_fog_cloud": 0,
                    "network_bytes": 0,
                    "proof_gen_cpu_ms": 0.0, "proof_verify_cpu_ms": 0.0,
                    "proof_size_bytes": 0,
                    **self._sys.snapshot()}

        vk_map = {n.node_id: n.vk for n in self.nodes}
        CONSENSUS_TIMEOUT = 5.0

        async def _try_phase(phase_name: str, phase_fn):
            try:
                return await asyncio.wait_for(phase_fn(), timeout=CONSENSUS_TIMEOUT)
            except asyncio.TimeoutError:
                return None

        async def _prepare_phase():
            votes = []
            for node in self.nodes:
                vote = await node.hotstuff.on_prepare(proposed_block)
                if vote:
                    votes.append((vote.node_id, vote.signature))
            return await leader_node.hotstuff.collect_votes(
                round_num, proposed_block.hash, "prepare", votes, vk_map,
            )

        async def _pre_commit_phase(qc_p):
            votes = []
            for node in self.nodes:
                vote = await node.hotstuff.on_pre_commit(qc_p)
                if vote:
                    votes.append((vote.node_id, vote.signature))
            return await leader_node.hotstuff.collect_votes(
                round_num, proposed_block.hash, "pre_commit", votes, vk_map,
            )

        async def _commit_phase(qc_pc):
            votes = []
            for node in self.nodes:
                vote = await node.hotstuff.on_commit(qc_pc)
                if vote:
                    votes.append((vote.node_id, vote.signature))
            return await leader_node.hotstuff.collect_votes(
                round_num, proposed_block.hash, "commit", votes, vk_map,
            )

        qc_prepare = await _try_phase("prepare", _prepare_phase)
        if qc_prepare is None:
            for node in self.nodes:
                if node.audit_logger:
                    node.audit_logger.log("VIEW_CHANGE", node.node_id, round_num,
                                          {"reason": "prepare_timeout", "leader": leader_id})
                    node.view_change.should_change_view(timeout=True)
            return {"round": round_num, "latency": 0, "qc_emitted": False,
                    "bytes_edge_fog": 0, "bytes_fog_intra": 0,
                    "bytes_fog_inter": 0, "bytes_fog_cloud": 0,
                    "network_bytes": 0,
                    "proof_gen_cpu_ms": 0.0, "proof_verify_cpu_ms": 0.0,
                    "proof_size_bytes": 0,
                    **self._sys.snapshot()}

        qc_pre_commit = await _try_phase("pre_commit", lambda: _pre_commit_phase(qc_prepare))
        if qc_pre_commit is None:
            for node in self.nodes:
                if node.audit_logger:
                    node.audit_logger.log("VIEW_CHANGE", node.node_id, round_num,
                                          {"reason": "pre_commit_timeout", "leader": leader_id})
                    node.view_change.should_change_view(timeout=True)
            return {"round": round_num, "latency": 0, "qc_emitted": False,
                    "bytes_edge_fog": 0, "bytes_fog_intra": 0,
                    "bytes_fog_inter": 0, "bytes_fog_cloud": 0,
                    "network_bytes": 0,
                    "proof_gen_cpu_ms": 0.0, "proof_verify_cpu_ms": 0.0,
                    "proof_size_bytes": 0,
                    **self._sys.snapshot()}

        qc_commit = await _try_phase("commit", lambda: _commit_phase(qc_pre_commit))
        if qc_commit is None:
            for node in self.nodes:
                if node.audit_logger:
                    node.audit_logger.log("VIEW_CHANGE", node.node_id, round_num,
                                          {"reason": "commit_timeout", "leader": leader_id})
                    node.view_change.should_change_view(timeout=True)
            return {"round": round_num, "latency": 0, "qc_emitted": False,
                    "bytes_edge_fog": 0, "bytes_fog_intra": 0,
                    "bytes_fog_inter": 0, "bytes_fog_cloud": 0,
                    "network_bytes": 0,
                    "proof_gen_cpu_ms": 0.0, "proof_verify_cpu_ms": 0.0,
                    "proof_size_bytes": 0,
                    **self._sys.snapshot()}

        entry = await leader_node.finalize_commit(qc_commit, proposed_block)
        if entry is None:
            return {"round": round_num, "latency": 0, "qc_emitted": False,
                    "bytes_edge_fog": 0, "bytes_fog_intra": 0,
                    "bytes_fog_inter": 0, "bytes_fog_cloud": 0,
                    "network_bytes": 0,
                    "proof_gen_cpu_ms": 0.0, "proof_verify_cpu_ms": 0.0,
                    "proof_size_bytes": 0,
                    **self._sys.snapshot()}

        block = entry.block

        for node in self.nodes:
            node.ledger.append(block)

        if self.use_dataset and delta_w_reg is not None:
            avg_grad = np.array(delta_w_reg)
        elif self.use_dataset and cluster_results:
            avg_grad = np.mean(
                [np.array(r.gradient.data) for r in cluster_results], axis=0
            )
        else:
            avg_grad = None

        self.audit_logger.log("GLOBAL_AGGREGATION", "cloud", round_num,
                              {"num_cluster_results": len(cluster_results)})

        if avg_grad is not None:
            lr = 0.01
            w_old = self._global_weights.copy()
            flat_w = np.concatenate([v.ravel() for v in self._global_weights.values()])
            flat_w -= lr * avg_grad

            shapes = [(12000, 64), (64,), (64, 5), (5,)]
            keys = ["W1", "b1", "W2", "b2"]
            new_w = {}
            start = 0
            for key, shape in zip(keys, shapes):
                size = np.prod(shape)
                new_w[key] = flat_w[start:start + size].reshape(shape)
                start += size
            self._global_weights = new_w
            self.audit_logger.log("MODEL_UPDATE", "cloud", round_num,
                                  {"lr": lr, "avg_grad_norm": float(np.linalg.norm(avg_grad))})

            # Model Validation Gate (camada Cloud)
            from core.cloud.model_validation import ModelValidationGate
            gate = ModelValidationGate()
            val_result = gate.validate(w_old, self._global_weights, loss=None)
            self.audit_logger.log("MODEL_VALIDATION", "cloud", round_num, {
                "accepted": val_result.accepted,
                "loss": val_result.loss,
                "delta_norm": val_result.delta_norm,
                "reason": val_result.reason,
            })

        self._last_delta_w_reg = delta_w_reg
        t_end = time.time()

        return self._build_metrics(
            round_num, leader_id, t_start, t_end, block, grads,
            cluster_results, snark_sample_ok,
            self._pending_snark_verify is not None,
        )

    async def run_experiment(
        self, num_rounds: int, warmup: int = 10
    ) -> ExperimentResult:
        self.setup()
        self._sys.cpu_percent()  # arm baseline so round 1 measures since experiment start
        for r in range(num_rounds + warmup):
            metrics = await self.run_round(
                r, fire_snark_verify=(r == num_rounds + warmup - 1)
            )
            if r >= warmup:
                self.result.round_metrics.append(metrics)
        if self._pending_snark_verify is not None:
            try:
                self._snark_sample_result = bool(await self._pending_snark_verify)
            except Exception:
                self._snark_sample_result = False
            self._pending_snark_verify = None
            if self.result.round_metrics:
                last = self.result.round_metrics[-1]
                last["snark_sampled_verified"] = 1
                last["snark_sampled_passed"] = int(bool(self._snark_sample_result))
        return self.result
