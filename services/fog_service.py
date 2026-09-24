"""Fog service — standalone gRPC + MQTT node for H-FedChain."""

import asyncio
import hashlib
import logging
import pickle
import time
from dataclasses import replace
from typing import Optional

logger = logging.getLogger(__name__)

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
    MessageType,
    VRFMessage,
)
from zkp.snark import SnarkVerifier
from zkp.stark import StarkProver


class FogConsensusServicer:
    """gRPC servicer for inter-Fog HotStuff consensus messages."""

    def __init__(self, fog_service: "FogService"):
        self.fog = fog_service

    async def ConsensusStream(self, request_iterator, context):
        peer_id = context.peer()
        async for msg in request_iterator:
            await self.fog._on_consensus_message(msg, peer_id)


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
        self._mqtt_enabled = True
        self._grpc_server = None
        self._peer_queues: dict[str, asyncio.Queue] = {}
        self._peer_channels: dict[str, any] = {}

        self._pending_gradients: list[GradientWithProof] = []
        self._vk_map: dict[str, bytes] = {}
        self._current_leader: Optional[str] = None
        self._last_aggregate: Optional[AggregateGradient] = None
        self._running = False

        # Consensus state
        self._pending_votes: list = []
        self._vote_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if self._mqtt_enabled:
            try:
                await self.mqtt.start()
                self.mqtt.subscribe(
                    f"gradients/{self.node_id}", self._on_mqtt_gradient
                )
            except Exception as e:
                logger.error("MQTT connection failed for node %s: %s", self.node_id, e)
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
        except RuntimeError as e:
            logger.warning("No running event loop for gradient callback: %s", e)

    async def on_gradient_received(self, gradient_with_proof: GradientWithProof):
        self._pending_gradients.append(gradient_with_proof)
        if len(self._pending_gradients) >= self.n - self.f:
            await self._run_round()

    # ------------------------------------------------------------------
    # Peer gRPC connections (bidirectional streaming)
    # ------------------------------------------------------------------

    async def _connect_peers(self):
        import grpc
        from proto import hfedchain_pb2_grpc

        for peer_id in self.peers:
            queue: asyncio.Queue = asyncio.Queue()
            self._peer_queues[peer_id] = queue
            addr = f"{peer_id}:{self.grpc_port}"

            async def request_generator(q=queue, p=peer_id):
                while self._running:
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=1.0)
                        yield msg
                    except asyncio.TimeoutError:
                        continue

            asyncio.create_task(self._connect_one_peer(peer_id, addr, request_generator))

    async def _connect_one_peer(self, peer_id, addr, req_gen):
        import grpc
        from proto import hfedchain_pb2_grpc

        for attempt in range(30):
            try:
                channel = grpc.aio.insecure_channel(addr)
                stub = hfedchain_pb2_grpc.FogConsensusStub(channel)
                self._peer_channels[peer_id] = channel
                stream = stub.ConsensusStream(req_gen())
                await self._read_stream(stream, peer_id)
                return
            except Exception:
                pass

    async def _read_stream(self, stream, peer_id: str):
        async for response in stream:
            await self._on_consensus_message(response, peer_id)

    async def _send_to(self, peer_id: str, msg):
        queue = self._peer_queues.get(peer_id)
        if queue is not None:
            await queue.put(msg)

    async def _broadcast(self, msg, exclude: Optional[list[str]] = None):
        targets = [p for p in self.peers if not exclude or p not in exclude]
        for peer_id in targets:
            await self._send_to(peer_id, msg)

    async def _on_consensus_message(self, msg, sender_id: str):
        which = msg.WhichOneof("msg")
        if which == "prepare":
            block = pickle.loads(msg.prepare.block_data)
            vote = await self.engine.on_prepare(block)
            if vote is not None:
                from proto import hfedchain_pb2
                vote_msg = hfedchain_pb2.ConsensusMessage(
                    vote=hfedchain_pb2.VoteMessage(
                        node_id=self.node_id,
                        round=vote.round,
                        phase=vote.phase,
                        block_hash=vote.block_hash,
                        signature=vote.signature,
                    )
                )
                leader_id = msg.prepare.leader_id
                await self._send_to(leader_id, vote_msg)

        elif which == "vote":
            self._pending_votes.append(msg.vote)
            self._vote_event.set()

        elif which == "qc":
            qc = pickle.loads(msg.qc.quorum_certificate)
            phase = msg.qc.phase
            if phase == "pre_commit":
                vote = await self.engine.on_pre_commit(qc)
            elif phase == "commit":
                vote = await self.engine.on_commit(qc)
            else:
                return
            if vote is not None:
                from proto import hfedchain_pb2
                vote_msg = hfedchain_pb2.ConsensusMessage(
                    vote=hfedchain_pb2.VoteMessage(
                        node_id=self.node_id,
                        round=vote.round,
                        phase=vote.phase,
                        block_hash=vote.block_hash,
                        signature=vote.signature,
                    )
                )
                leader_id = msg.qc.leader_id
                await self._send_to(leader_id, vote_msg)

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

        np_grads = [np.array(g.data) for g in valid_grads]
        selected = self.multikrum.select(np_grads, self.f)
        accepted = [valid_grads[i].node_id for i in selected]
        rejected = [
            valid_grads[i].node_id for i in range(len(valid_grads))
            if i not in selected
        ]
        total_adv = sum(1 for g in valid_grads if g.node_id.startswith("adv_"))
        rej_adv = sum(1 for i in selected if valid_grads[i].node_id.startswith("adv_"))
        rej_honest = len(rejected) - (total_adv - rej_adv)
        agg = AggregateGradient(
            node_id=self.node_id,
            round=round_num,
            gradient=valid_grads[selected[0]],
            accepted_devices=accepted,
            rejected_devices=rejected,
            total_adversarial=total_adv,
            rejected_adversarial=total_adv - rej_adv,
            rejected_honest=rej_honest,
        )
        self._last_aggregate = agg

        seed = hashlib.sha256(f"round_{round_num}".encode()).digest()
        y, proof = self.vrf.evaluate(self.engine.sk, seed)
        candidates = [VRFMessage(self.node_id, round_num, y, proof)]
        all_nodes = [self.node_id] + self.peers
        leader_idx = round_num % len(all_nodes)
        is_leader = leader_idx == 0
        self._current_leader = all_nodes[leader_idx]

        if is_leader:
            await self._run_leader_consensus(round_num, seed, valid_grads)
        else:
            await self.engine.start_round(round_num, is_leader=False)

    async def _run_leader_consensus(self, round_num, seed, valid_grads):
        from proto import hfedchain_pb2

        block = Block(
            round=round_num,
            gradient_hash=seed,
            qc_commit=None,
            stark_proof=None,
            accepted_devices=[g.node_id for g in valid_grads],
            rejected_devices=[],
            timestamp=time.time(),
            prev_hash=b"\x00" * 32,
            n=self.n, f=self.f,
        )
        if self.ledger.get_height() > 0:
            block = replace(
                block, prev_hash=self.ledger._entries[-1].block.hash
            )

        await self.engine.start_round(round_num, is_leader=True)
        proposal = await self.engine.propose(block)
        if proposal is None:
            return

        # Single-node: commit directly
        if not self.peers:
            entry = await self.engine.on_qc_commit(
                self.quorum_certifier.collect(
                    round=round_num, block_hash=block.hash,
                    msg_type=MessageType.COMMIT,
                    signatures=[(self.node_id, bytes(0))],
                    quorum_size=1,
                )
            )
            if entry:
                self.ledger.append(entry.block)
            return

        prepare_msg = hfedchain_pb2.ConsensusMessage(
            prepare=hfedchain_pb2.PrepareProposal(
                leader_id=self.node_id,
                round=round_num,
                block_data=pickle.dumps(block),
                signature=bytes(0),
            )
        )
        await self._broadcast(prepare_msg)

        quorum = self.quorum_certifier.quorum_size(self.n)

        qc_prepare = await self._collect_quorum_votes(round_num, block.hash, "prepare", quorum)
        if qc_prepare is None:
            return

        qc_msg = hfedchain_pb2.ConsensusMessage(
            qc=hfedchain_pb2.QcBroadcast(
                leader_id=self.node_id,
                round=round_num,
                phase="pre_commit",
                quorum_certificate=pickle.dumps(qc_prepare),
            )
        )
        await self._broadcast(qc_msg)

        qc_pre_commit = await self._collect_quorum_votes(round_num, block.hash, "pre_commit", quorum)
        if qc_pre_commit is None:
            return

        qc_msg = hfedchain_pb2.ConsensusMessage(
            qc=hfedchain_pb2.QcBroadcast(
                leader_id=self.node_id,
                round=round_num,
                phase="commit",
                quorum_certificate=pickle.dumps(qc_pre_commit),
            )
        )
        await self._broadcast(qc_msg)

        qc_commit = await self._collect_quorum_votes(round_num, block.hash, "commit", quorum)
        if qc_commit is None:
            return

        entry = await self.engine.on_qc_commit(qc_commit)
        if entry is None:
            return

        if self.variant in ("stark", "full"):
            proof = await self.stark_prover.generate_proof(entry.block)
            entry = LedgerEntry(
                block=replace(entry.block, stark_proof=proof),
                node_id=entry.node_id,
                stored_at=entry.stored_at,
            )

        self.ledger.append(entry.block)
        await self._send_to_cloud(entry.block)

    async def _collect_quorum_votes(self, round_num, block_hash, phase, quorum):
        self._pending_votes = []
        self._vote_event.clear()

        deadline = time.time() + 5.0
        while len(self._pending_votes) < quorum and time.time() < deadline:
            try:
                await asyncio.wait_for(self._vote_event.wait(), timeout=0.5)
                self._vote_event.clear()
            except asyncio.TimeoutError:
                pass

        if len(self._pending_votes) < quorum:
            return None

        phase_map = {
            "prepare": MessageType.PREPARE,
            "pre_commit": MessageType.PRE_COMMIT,
            "commit": MessageType.COMMIT,
        }
        msg_type = phase_map.get(phase)
        if msg_type is None:
            return None

        signatures = [(v.node_id, v.signature) for v in self._pending_votes[:quorum]]
        try:
            return self.quorum_certifier.collect(
                round=round_num,
                block_hash=block_hash,
                msg_type=msg_type,
                signatures=signatures,
                quorum_size=quorum,
                vk_map=self._vk_map,
            )
        except ValueError:
            return None

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
