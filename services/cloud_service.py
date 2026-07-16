"""Cloud service — top-tier gRPC aggregation node for H-FedChain."""

import asyncio
import time
from typing import Optional

import numpy as np

from core.cloud.model_validation import ModelValidationGate
from core.ledger.worm_store import WormStore
from hfc_types.block import GlobalOutput
from hfc_types.crypto import StarkProof
from zkp.stark import StarkVerifier
from services.identity_service import GlobalIdentityService


class CloudService:
    def __init__(
        self,
        n_expected_clusters: int = 1,
        grpc_port: int = 50052,
        learning_rate: float = 0.01,
    ):
        self.n_expected_clusters = n_expected_clusters
        self.grpc_port = grpc_port
        self.learning_rate = learning_rate

        self.validation_gate = ModelValidationGate()
        self.worm = WormStore()
        self.stark_verifier = StarkVerifier()
        self.identity_service = GlobalIdentityService()
        self.ca_vk = self.identity_service.ca_vk
        self.converged = False
        self.global_weights: Optional[dict] = None

        self._grpc_server = None
        self._round_outputs: dict[int, dict[str, dict]] = {}

    async def start(self):
        import grpc
        from proto import hfedchain_pb2_grpc

        self._grpc_server = grpc.aio.server()
        hfedchain_pb2_grpc.add_CloudAggregationServicer_to_server(
            CloudAggregationServicer(self), self._grpc_server,
        )
        self._grpc_server.add_insecure_port(f"0.0.0.0:{self.grpc_port}")
        await self._grpc_server.start()
        print(f"[cloud] gRPC server listening on :{self.grpc_port}")
        await self._grpc_server.wait_for_termination()

    async def stop(self):
        if self._grpc_server:
            await self._grpc_server.stop(None)

    async def submit_cluster_output(self, cluster_output_proto) -> "GlobalOutputMessage":
        from proto import hfedchain_pb2

        round_num = cluster_output_proto.round
        cluster_id = cluster_output_proto.cluster_id

        # Verify STARK proof if present
        if cluster_output_proto.stark_proof:
            proof = StarkProof(
                proof_bytes=cluster_output_proto.stark_proof,
                public_inputs={"round": round_num, "cluster_id": cluster_id},
            )
            if not await self.stark_verifier.verify(proof, proof.public_inputs):
                return hfedchain_pb2.GlobalOutputMessage(
                    delta_w_inter=b"", n_active_clusters=0, round=round_num,
                    converged=False,
                )

        # Collect outputs per round, keyed by cluster_id
        if round_num not in self._round_outputs:
            self._round_outputs[round_num] = {}
        self._round_outputs[round_num][cluster_id] = {
            "delta_w": cluster_output_proto.delta_w,
            "n_devices": cluster_output_proto.n_devices,
        }

        # Aggregate once all clusters have reported
        if len(self._round_outputs[round_num]) >= self.n_expected_clusters:
            return await self._aggregate_round(round_num)

        return hfedchain_pb2.GlobalOutputMessage(
            delta_w_inter=b"", n_active_clusters=0, round=round_num,
            converged=False,
        )

    async def _aggregate_round(self, round_num: int):
        from proto import hfedchain_pb2

        outputs = list(self._round_outputs.pop(round_num).values())
        total_devices = sum(o["n_devices"] for o in outputs)
        n_active = len(outputs)

        # Weighted FedAvg over the gradient deltas
        arrays = []
        weights = []
        for o in outputs:
            if o["delta_w"]:
                arr = np.frombuffer(o["delta_w"], dtype=np.float32)
                arrays.append(arr)
                weight = o["n_devices"] / total_devices if total_devices > 0 else 1.0 / n_active
                weights.append(weight)

        if arrays:
            weighted = sum(w * a for w, a in zip(weights, arrays))
            delta_w_inter = weighted.astype(np.float32).tobytes()
        else:
            weighted = np.zeros(1)
            delta_w_inter = b""

        # Compute proposed new weights then validate
        if self.global_weights is not None:
            w_new = {}
            for key in self.global_weights:
                w_new[key] = self.global_weights[key] - self.learning_rate * weighted
        else:
            w_new = {"W1": weighted}

        result = self.validation_gate.validate(self.global_weights, w_new)
        if result.accepted:
            self.global_weights = w_new

        # Persist to WORM ledger
        global_output = GlobalOutput(
            delta_w_inter=delta_w_inter,
            pi_inter=None,
            n_active_clusters=n_active,
            round_num=round_num,
        )
        self.worm.append(global_output)

        return hfedchain_pb2.GlobalOutputMessage(
            delta_w_inter=delta_w_inter,
            pi_inter=None,
            n_active_clusters=n_active,
            round=round_num,
            converged=self.converged,
        )


class CloudAggregationServicer:
    def __init__(self, cloud: CloudService):
        self.cloud = cloud

    async def SubmitClusterOutput(self, request, context):
        return await self.cloud.submit_cluster_output(request)
