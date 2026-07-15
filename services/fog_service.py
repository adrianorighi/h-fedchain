"""Fog service — standalone gRPC + MQTT node for H-FedChain."""

import asyncio
import hashlib
import pickle
import time
from dataclasses import replace
from typing import Optional

import numpy as np

from core.hotstuff.engine import HotStuffEngine
from core.hotstuff.quorum import QuorumCertifier
from core.hotstuff.view_change import ViewChangeHandler
from core.ledger.store import LedgerStore
from core.multikrum.aggregator import MultiKrum
from core.network.mqtt import MqttClient
from core.vrf.election import VRFLeaderElection
from hfc_types.block import Block, LedgerEntry
from hfc_types.messages import (
    AggregateGradient,
    GradientWithProof,
    VRFMessage,
)
from zkp.snark import SnarkVerifier
from zkp.stark import StarkProver


class FogConsensusServicer:
    """gRPC servicer for inter-Fog HotStuff consensus messages."""

    def __init__(self, fog_service: "FogService"):
        self.fog = fog_service

    async def ConsensusStream(self, request_iterator, context):
        async for msg in request_iterator:
            pass


class FogService:
    """Standalone Fog node with gRPC + MQTT connectivity."""

    def __init__(
        self,
        node_id: str,
        sk: bytes,
        vk: bytes,
        peers: list[str],
        n: int,
        f: int,
        variant: str = "no_zkp",
        grpc_port: int = 50051,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
    ):
        self.node_id = node_id
        self.peers = peers
        self.n = n
        self.f = f
        self.variant = variant
        self.grpc_port = grpc_port

        self.engine = HotStuffEngine(node_id, sk, vk, peers, n, f)
        self.vrf = VRFLeaderElection()
        self.multikrum = MultiKrum()
        self.quorum_certifier = QuorumCertifier()
        self.view_change = ViewChangeHandler(n, f)
        self.ledger = LedgerStore()
        self.snark_verifier = SnarkVerifier()
        self.stark_prover = StarkProver()

        self.mqtt = MqttClient(f"fog_{node_id}", mqtt_broker, mqtt_port)
        self._grpc_server = None
        self._peer_stubs: dict[str, any] = {}

        self._pending_gradients: list[GradientWithProof] = []
        self._vk_map: dict[str, bytes] = {}
        self._current_leader: Optional[str] = None
        self._last_aggregate: Optional[AggregateGradient] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        await self.mqtt.start()
        self.mqtt.subscribe(
            f"gradients/{self.node_id}", self._on_mqtt_gradient
        )
        await self._connect_peers()
        asyncio.create_task(self._serve_grpc())
        self._running = True

    async def stop(self):
        self._running = False
        if self._grpc_server is not None:
            await self._grpc_server.stop(5)
        await self.mqtt.stop()

    def set_vk(self, node_id: str, vk: bytes):
        self._vk_map[node_id] = vk

    # ------------------------------------------------------------------
    # MQTT gradient ingestion
    # ------------------------------------------------------------------

    def _on_mqtt_gradient(self, payload: bytes):
        try:
            grad = pickle.loads(payload)
        except Exception:
            return
        try:
            asyncio.get_running_loop().create_task(
                self.on_gradient_received(grad)
            )
        except RuntimeError:
            pass

    async def on_gradient_received(self, gradient_with_proof: GradientWithProof):
        self._pending_gradients.append(gradient_with_proof)
        if len(self._pending_gradients) >= self.n - self.f:
            await self._run_round()

    # ------------------------------------------------------------------
    # Peer gRPC connections
    # ------------------------------------------------------------------

    async def _connect_peers(self):
        import grpc
        from proto import hfedchain_pb2_grpc

        for peer_id in self.peers:
            addr = f"{peer_id}:{self.grpc_port}"
            try:
                channel = grpc.aio.insecure_channel(addr)
                stub = hfedchain_pb2_grpc.FogConsensusStub(channel)
                self._peer_stubs[peer_id] = stub
            except Exception:
                pass

    async def _serve_grpc(self):
        import grpc
        from proto import hfedchain_pb2_grpc

        self._grpc_server = grpc.aio.server()
        hfedchain_pb2_grpc.add_FogConsensusServicer_to_server(
            FogConsensusServicer(self), self._grpc_server
        )
        self._grpc_server.add_insecure_port(f"0.0.0.0:{self.grpc_port}")
        await self._grpc_server.start()
        await self._grpc_server.wait_for_termination()

    # ------------------------------------------------------------------
    # Consensus round
    # ------------------------------------------------------------------

    async def _run_round(self):
        round_num = self.engine.round + 1

        # Step 1: SNARK verify
        valid_grads = []
        for item in self._pending_gradients:
            if (
                isinstance(item, GradientWithProof)
                and item.snark_proof is not None
                and self.variant in ("snark", "full")
            ):
                vk = self._vk_map.get(item.gradient.node_id)
                if vk is None or not await self.snark_verifier.verify(
                    item.snark_proof, b"", vk
                ):
                    continue
            valid_grads.append(item.gradient)

        self._pending_gradients.clear()

        if len(valid_grads) < self.n - self.f:
            return

        # Step 2: MultiKrum
        np_grads = [np.array(g.data) for g in valid_grads]
        selected = self.multikrum.select(np_grads, self.f)
        agg = AggregateGradient(
            node_id=self.node_id,
            round=round_num,
            gradient=valid_grads[selected[0]],
            accepted_devices=[g.node_id for g in valid_grads],
            rejected_devices=[],
            total_adversarial=0,
            rejected_adversarial=0,
            rejected_honest=0,
        )
        self._last_aggregate = agg

        # Step 3: VRF elect leader
        seed = hashlib.sha256(f"round_{round_num}".encode()).digest()
        y, proof = self.vrf.evaluate(self.engine.sk, seed)
        candidates = [VRFMessage(self.node_id, round_num, y, proof)]
        leader_idx = round_num % len([self.node_id] + self.peers)
        is_leader = leader_idx == 0
        self._current_leader = (
            self.node_id
            if is_leader
            else [self.node_id] + self.peers[leader_idx]
        )

        # Step 4: HotStuff consensus
        if is_leader:
            block = Block(
                round=round_num,
                gradient_hash=seed,
                qc_commit=None,
                stark_proof=None,
                accepted_devices=[g.node_id for g in valid_grads],
                rejected_devices=[],
                timestamp=time.time(),
                prev_hash=b"\x00" * 32,
            )
            if self.ledger.get_height() > 0:
                block = replace(
                    block, prev_hash=self.ledger._entries[-1].block.hash
                )

            await self.engine.start_round(round_num, is_leader=True)
            proposal = await self.engine.propose(block)
            if proposal is not None:
                for peer_id in self.peers:
                    stub = self._peer_stubs.get(peer_id)
                    if stub:
                        self._send_prepare(stub, peer_id, round_num, block)

                quorum = self.quorum_certifier.quorum_size(self.n)
                votes = [(p, b"sim_sig") for p in self.peers[:quorum]]
                qc_prepare = await self.engine.collect_votes(
                    round_num,
                    block.hash,
                    "prepare",
                    votes,
                    vk_map=self._vk_map,
                )
                if qc_prepare:
                    for _ in self.peers:
                        await self.engine.on_pre_commit(qc_prepare)
                    qc_pre_commit = await self.engine.collect_votes(
                        round_num, block.hash, "pre_commit", votes
                    )
                    if qc_pre_commit:
                        for _ in self.peers:
                            await self.engine.on_commit(qc_pre_commit)
                        qc_commit = await self.engine.collect_votes(
                            round_num, block.hash, "commit", votes
                        )
                        if qc_commit:
                            entry = await self.engine.on_qc_commit(qc_commit)
                            if entry:
                                if self.variant in ("stark", "full"):
                                    proof = (
                                        await self.stark_prover.generate_proof(
                                            entry.block
                                        )
                                    )
                                    entry = LedgerEntry(
                                        block=replace(
                                            entry.block,
                                            stark_proof=proof,
                                        ),
                                        node_id=entry.node_id,
                                        stored_at=entry.stored_at,
                                    )
                                self.ledger.append(entry.block)
                                await self._send_to_cloud(entry.block)
        else:
            await self.engine.start_round(round_num, is_leader=False)

    def _send_prepare(self, stub, peer_id, round_num, block):
        try:
            from proto import hfedchain_pb2

            msg = hfedchain_pb2.ConsensusMessage(
                prepare=hfedchain_pb2.PrepareProposal(
                    leader_id=self.node_id,
                    round=round_num,
                    block_data=pickle.dumps(block),
                    signature=bytes(0),
                )
            )
            stub.ConsensusStream(msg)
        except Exception:
            pass

    async def _send_to_cloud(self, block: Block):
        try:
            import grpc
            from proto import hfedchain_pb2, hfedchain_pb2_grpc

            channel = grpc.aio.insecure_channel("cloud:50052")
            stub = hfedchain_pb2_grpc.CloudAggregationStub(channel)
            output = hfedchain_pb2.ClusterOutput(
                cluster_id=self.node_id,
                round=block.round,
                delta_w=block.gradient_hash,
                stark_proof=block.stark_proof.proof_bytes
                if block.stark_proof
                else None,
                qc_commit=pickle.dumps(block.qc_commit)
                if block.qc_commit
                else None,
                n_devices=len(self.peers) + 1,
            )
            await stub.SubmitClusterOutput(output)
        except Exception as e:
            print(f"[{self.node_id}] Cloud submission failed: {e}")
