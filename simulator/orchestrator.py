import asyncio
import time
import numpy as np
from hashlib import sha256
from typing import Optional
from hfc_types.messages import Gradient, GradientWithProof, VRFMessage, AggregateGradient, MessageType
from hfc_types.block import Block, QuorumCertificate
from simulator.network import EmulatedNetwork
from simulator.fog_node import FogNode
from core.pki import generate_keypair
from dataset.loader import PTBXLLoader
from dataset.partitioner import DirichletPartitioner
from dataset.edge_worker import EdgeWorker


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
        adversarial_ratio: float = 0.0,
        use_dataset: bool = False,
        dataset_max_records: int = 500,
        dirichlet_alpha: float = 0.5,
        variant: str = "no_zkp",
    ):
        self.num_clusters = num_clusters
        self.nodes_per_cluster = nodes_per_cluster
        self.devices_per_cluster = devices_per_cluster
        self.f = f
        self.latency_ms = latency_ms
        self.adversarial_ratio = adversarial_ratio
        self.use_dataset = use_dataset
        self.dataset_max_records = dataset_max_records
        self.dirichlet_alpha = dirichlet_alpha
        self.variant = variant
        self.nodes: list[FogNode] = []
        self.network = EmulatedNetwork(latency_ms)
        self.result = ExperimentResult()
        self.edge_workers: list[EdgeWorker] = []
        self._global_weights: dict = {}

    def setup(self):
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
            self.nodes.append(node)
        for node in self.nodes:
            for other in self.nodes:
                node._set_vk(other.node_id, other.vk)

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

        num_adv = int(num_devices * self.adversarial_ratio)
        use_snark = self.variant in ("snark", "full")
        self.edge_workers = []
        for i, indices in enumerate(assignments):
            if not indices:
                continue
            is_adv = i < num_adv
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

        self._global_weights = self.edge_workers[0].model.get_weights()

    def _generate_gradients(
        self, round_num: int
    ) -> list[GradientWithProof]:
        if self.use_dataset:
            return self._generate_real_gradients(round_num)

        grads: list[GradientWithProof] = []
        for d in range(self.devices_per_cluster * self.num_clusters):
            is_adv = (
                self.adversarial_ratio > 0.0
                and d < int(self.devices_per_cluster * self.adversarial_ratio)
            )
            data = (
                np.random.randn(10).tolist()
                if not is_adv
                else [100.0 * float(np.random.randn()) for _ in range(10)]
            )
            gid = f"adv_{d}" if is_adv else f"d{d}"
            grads.append(GradientWithProof(
                gradient=Gradient(node_id=gid, round=round_num, data=data),
            ))
        return grads

    def _generate_real_gradients(self, round_num: int) -> list[GradientWithProof]:
        grads = []
        for worker in self.edge_workers:
            grad = worker.train_round(
                self._global_weights,
                round_num=round_num,
            )
            grads.append(grad)
        return grads

    async def run_round(self, round_num: int) -> dict:
        t_start = time.time()
        seed = sha256(f"round_{round_num}".encode()).digest()
        grads = self._generate_gradients(round_num)

        model_hash = None
        if self.use_dataset and self._global_weights:
            model_hash = sha256(
                str(sorted(self._global_weights.items())).encode()
            ).digest()

        cluster_results: list[AggregateGradient] = []
        for node in self.nodes:
            result = await node.process_round(grads, seed, round_num, model_hash=model_hash)
            if result:
                cluster_results.append(result)

        if not cluster_results:
            return {"round": round_num, "latency": 0, "qc_emitted": False}

        # VRF election
        vrf = self.nodes[0].vrf
        vk_map = {n.node_id: n.vk for n in self.nodes}
        candidates = []
        for n in self.nodes:
            y, proof = vrf.evaluate(n.sk, seed)
            candidates.append(VRFMessage(
                node_id=n.node_id,
                round=round_num,
                y=y,
                proof=proof,
            ))
        leader_id = vrf.elect(candidates, seed, vk_map)

        # Build block
        quorum_size = self.nodes[0].qc.quorum_size(self.nodes_per_cluster)
        qc = QuorumCertificate(
            round=round_num,
            block_hash=seed,
            signatures=[(n.node_id, b"sig") for n in self.nodes[:quorum_size]],
            msg_type=MessageType.COMMIT,
        )
        prev_hash = b"\x00" * 32
        if round_num > 0 and self.nodes[0].ledger.get_height() > 0:
            prev_hash = self.nodes[0].ledger._entries[-1].block.hash

        block = Block(
            round=round_num,
            gradient_hash=seed,
            qc_commit=qc,
            stark_proof=None,
            accepted_devices=[g.gradient.node_id for g in grads if not g.gradient.node_id.startswith("adv_")],
            rejected_devices=[g.gradient.node_id for g in grads if g.gradient.node_id.startswith("adv_")],
            timestamp=t_start,
            prev_hash=prev_hash,
        )

        for node in self.nodes:
            node.ledger.append(block)

        if self.use_dataset and cluster_results:
            accepted_grads = [np.array(cluster_results[0].gradient.data)]
            for r in cluster_results[1:]:
                accepted_grads.append(np.array(r.gradient.data))
            avg_grad = np.mean(accepted_grads, axis=0)

            lr = 0.01
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

        t_end = time.time()
        total_adv = sum(r.total_adversarial for r in cluster_results)
        total_rej_adv = sum(r.rejected_adversarial for r in cluster_results)
        total_rej_honest = sum(r.rejected_honest for r in cluster_results)
        return {
            "round": round_num,
            "leader": leader_id,
            "latency": t_end - t_start,
            "num_accepted": len(block.accepted_devices),
            "num_rejected": len(block.rejected_devices),
            "ledger_height": self.nodes[0].ledger.get_height(),
            "qc_emitted": True,
            "num_adversarial": total_adv,
            "num_honest": (self.devices_per_cluster * self.num_clusters) - total_adv,
            "rejected_adversarial": total_rej_adv,
            "falsely_rejected": total_rej_honest,
            "variant": self.variant,
        }

    async def run_experiment(
        self, num_rounds: int, warmup: int = 10
    ) -> ExperimentResult:
        self.setup()
        for r in range(num_rounds + warmup):
            metrics = await self.run_round(r)
            if r >= warmup:
                self.result.round_metrics.append(metrics)
        return self.result
