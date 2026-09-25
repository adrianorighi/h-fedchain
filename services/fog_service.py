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
from core.pki import gradient_signed_message, sign as pki_sign, verify as pki_verify
from core.pki.gradient import gradient_to_f32
from core.vrf.election import VRFLeaderElection
from hfc_types.block import Block, LedgerEntry
from hfc_types.messages import (
    AggregateGradient,
    Gradient,
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
        cluster_id: str = "default",
        gradient_threshold: int = 0,
        cloud_address: str = "cloud:50052",
        peer_addrs: Optional[dict[str, str]] = None,
        vote_timeout_s: float = 5.0,
        mqtt_enabled: bool = True,
    ):
        self.node_id = node_id
        self.peers = peers
        self.n = n
        self.f = f
        self.variant = variant
        self.grpc_port = grpc_port
        self.cluster_id = cluster_id
        self.gradient_threshold = gradient_threshold or (n - f)
        self.cloud_address = cloud_address
        self.peer_addrs = peer_addrs or {}
        self.vote_timeout_s = vote_timeout_s

        self.engine = HotStuffEngine(node_id, sk, vk, peers, n, f)
        self.vrf = VRFLeaderElection()
        self.multikrum = MultiKrum()
        self.quorum_certifier = QuorumCertifier()
        self.view_change = ViewChangeHandler(n, f)
        self.ledger = LedgerStore()
        self.snark_verifier = SnarkVerifier()
        self.stark_prover = StarkProver()

        self.mqtt = MqttClient(f"fog_{node_id}", mqtt_broker, mqtt_port)
        self._mqtt_enabled = mqtt_enabled
        self._grpc_server = None
        self._peer_queues: dict[str, asyncio.Queue] = {}
        self._peer_channels: dict[str, any] = {}

        self._pending_gradients: list[GradientWithProof] = []
        self._round_in_flight = False
        self._vk_map: dict[str, bytes] = {}
        self._kx_sent: set[str] = set()
        self._current_leader: Optional[str] = None
        self._last_aggregate: Optional[AggregateGradient] = None
        self._last_delta_w: Optional[bytes] = None
        self._last_n_devices: int = 0
        # Per-round delta (delta_bytes, n_devices) so a re-drive always
        # ships the round's own delta, even after later rounds overwrote
        # the shared instance state. Window-pruned in _run_round_once.
        self._delta_by_round: dict[int, tuple[Optional[bytes], int]] = {}
        self._running = False

        # Consensus state
        self._pending_votes: list = []
        self._vote_event: asyncio.Event = asyncio.Event()
        self._vc_sent: set[int] = set()
        self._pending_vc: dict[int, set[str]] = {}
        self._vc_applied: set[int] = set()
        self._completed_rounds: set[int] = set()
        self._round_completed: dict[int, asyncio.Event] = {}
        self._qc_watchers: dict[int, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if self._mqtt_enabled:
            try:
                await self.mqtt.start()
                self.mqtt.subscribe(
                    f"gradients/{self.cluster_id}", self._on_mqtt_gradient
                )
                self.mqtt.subscribe(
                    f"register/{self.cluster_id}/#", self._on_mqtt_registration
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

    def _key_exchange_msg(self):
        from proto import hfedchain_pb2
        return hfedchain_pb2.ConsensusMessage(
            key_exchange=hfedchain_pb2.KeyExchange(
                node_id=self.node_id, vk=self.engine.vk
            )
        )

    def _on_mqtt_registration(self, payload: bytes):
        try:
            reg = pickle.loads(payload)
        except Exception:
            return
        if not isinstance(reg, dict):
            logger.warning("Ignoring malformed registration (not a dict)")
            return
        node_id = reg.get("node_id")
        vk = reg.get("vk")
        if not isinstance(node_id, str) or not isinstance(vk, bytes) \
                or len(vk) != 32:
            logger.warning(
                "Ignoring malformed registration from %r "
                "(node_id must be str, vk must be 32 bytes)",
                node_id,
            )
            return
        if node_id in self.peers:
            # Peer vks are established exclusively via key exchange over the
            # authenticated gRPC channel; the MQTT registration topic is
            # device-facing and must not be able to re-key (spoof) a peer.
            logger.warning(
                "Ignoring MQTT registration for peer %s "
                "(peer vks come only from key exchange)",
                node_id,
            )
            return
        if node_id in self._vk_map:
            if self._vk_map[node_id] != vk:
                logger.warning(
                    "Registration for %s attempted to replace pinned vk; "
                    "keeping first vk", node_id,
                )
            return
        self._vk_map[node_id] = vk

    async def on_gradient_received(self, gradient_with_proof: GradientWithProof):
        g = gradient_with_proof.gradient
        if g.round <= self.engine.round:
            logger.warning(
                "Rejected stale gradient from %s (round %d <= current round %d)",
                g.node_id, g.round, self.engine.round,
            )
            return
        # Data shape/type validated BEFORE signature verification so that
        # malformed payloads are rejected regardless of signature validity.
        try:
            arr = gradient_to_f32(g.data)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Rejected gradient from %s (invalid data: %s)", g.node_id, e
            )
            return
        if arr.ndim != 1 or arr.size == 0:
            logger.warning(
                "Rejected gradient from %s (invalid data: expected a non-empty "
                "flat numeric list, got ndim=%d size=%d)",
                g.node_id, arr.ndim, arr.size,
            )
            return
        for item in self._pending_gradients:
            if item.gradient.round != g.round:
                continue
            pending_size = gradient_to_f32(item.gradient.data).size
            if pending_size != arr.size:
                logger.warning(
                    "Rejected gradient from %s (size mismatch for round %d: "
                    "got %d values, pending gradients have %d)",
                    g.node_id, g.round, arr.size, pending_size,
                )
                return
        if any(item.gradient.node_id == g.node_id and item.gradient.round == g.round
               for item in self._pending_gradients):
            logger.warning(
                "Rejected duplicate gradient from %s for round %d "
                "(already have a gradient from this device for this round)",
                g.node_id, g.round,
            )
            return
        vk = self._vk_map.get(g.node_id)
        signed = gradient_signed_message(g.node_id, g.round, g.data)
        if vk is None or g.signature is None:
            logger.warning("Rejected gradient from %s (pki)", g.node_id)
            return
        try:
            signature_ok = pki_verify(vk, signed, g.signature)
        except Exception as e:
            logger.warning(
                "Rejected gradient from %s (pki error: %s)", g.node_id, e
            )
            return
        if not signature_ok:
            logger.warning("Rejected gradient from %s (pki)", g.node_id)
            return
        self._pending_gradients.append(gradient_with_proof)
        if len(self._pending_gradients) >= self.gradient_threshold:
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
            addr = self.peer_addrs.get(peer_id) or f"{peer_id}:{self.grpc_port}"

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

        for attempt in range(300):
            if not self._running:
                return
            try:
                channel = grpc.aio.insecure_channel(addr)
                stub = hfedchain_pb2_grpc.FogConsensusStub(channel)
                self._peer_channels[peer_id] = channel
                stream = stub.ConsensusStream(req_gen())
                self._kx_sent.add(peer_id)
                await self._peer_queues[peer_id].put(self._key_exchange_msg())
                await self._read_stream(stream, peer_id)
                return
            except Exception as e:
                if not self._running:
                    return
                if attempt % 30 == 0:
                    logger.warning(
                        "connect to %s failed (attempt %d): %s", peer_id, attempt, e
                    )
                await asyncio.sleep(1.0)

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
        if which == "key_exchange":
            kx = msg.key_exchange
            if kx.node_id not in self.peers:
                logger.warning(
                    "Ignoring key exchange from non-peer %s", kx.node_id
                )
                return
            if not isinstance(kx.vk, bytes) or len(kx.vk) != 32:
                logger.warning(
                    "Ignoring key exchange with malformed vk from %s",
                    kx.node_id,
                )
                return
            if kx.node_id in self._vk_map:
                if self._vk_map[kx.node_id] != kx.vk:
                    logger.warning(
                        "Key exchange for %s attempted to replace pinned vk; "
                        "keeping first vk", kx.node_id,
                    )
                return
            self._vk_map[kx.node_id] = kx.vk
            if kx.node_id not in self._kx_sent and kx.node_id in self._peer_queues:
                self._kx_sent.add(kx.node_id)
                await self._send_to(kx.node_id, self._key_exchange_msg())
            return

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
                if (
                    vote.round not in self._qc_watchers
                    and vote.round not in self._completed_rounds
                ):
                    self._qc_watchers[vote.round] = asyncio.create_task(
                        self._qc_watch(vote.round)
                    )

        elif which == "vote":
            self._pending_votes.append(msg.vote)
            self._vote_event.set()

        elif which == "qc":
            qc = pickle.loads(msg.qc.quorum_certificate)
            phase = msg.qc.phase
            if phase == "pre_commit":
                vote = await self.engine.on_pre_commit(
                    qc, vk_map=self._vk_map_with_self()
                )
            elif phase == "commit":
                vote = await self.engine.on_commit(
                    qc, vk_map=self._vk_map_with_self()
                )
            elif phase == "qc_commit":
                await self._commit_from_qc(qc)
                return
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

        elif which == "view_change":
            await self._on_view_change(msg.view_change)

    async def _commit_from_qc(self, qc):
        if qc.round in self._completed_rounds:
            return
        proposal = self.engine._last_proposal
        if proposal is None or proposal.round != qc.round:
            logger.warning(
                "Ignoring commit QC for round %d: no prepared proposal",
                qc.round,
            )
            return
        entry = await self.engine.on_qc_commit(
            qc, vk_map=self._vk_map_with_self()
        )
        if entry is None:
            return
        if not self.ledger.append(entry.block):
            logger.warning(
                "Commit QC for round %d does not extend the local ledger",
                qc.round,
            )
            return
        self._mark_round_completed(qc.round)

    async def _on_view_change(self, vc):
        if vc.node_id not in self.peers:
            logger.warning(
                "Ignoring view change from non-peer %s", vc.node_id
            )
            return
        vk = self._vk_map.get(vc.node_id)
        if vk is None:
            logger.warning(
                "Ignoring view change from %s (unknown vk)", vc.node_id
            )
            return
        try:
            valid = pki_verify(
                vk, str(vc.round).encode() + vc.highest_qc, vc.signature
            )
        except Exception as e:
            logger.warning(
                "View change from %s failed verification: %s", vc.node_id, e
            )
            return
        if not valid:
            logger.warning(
                "Ignoring view change with invalid signature from %s",
                vc.node_id,
            )
            return
        if vc.round in self._completed_rounds:
            return
        self._pending_vc.setdefault(vc.round, set()).add(vc.node_id)
        if vc.round not in self._vc_sent:
            await self._send_view_change(vc.round)
        if (
            len(self._pending_vc[vc.round]) >= self.n - self.f
            and vc.round not in self._vc_applied
        ):
            self._vc_applied.add(vc.round)
            new_leader = self._vc_leader(vc.round)
            if new_leader != self.node_id:
                logger.info(
                    "View change for round %d: new leader is %s",
                    vc.round, new_leader,
                )
                return
            proposal = self.engine._last_proposal
            if proposal is None or proposal.round != vc.round:
                logger.warning(
                    "Selected as leader for round %d view change "
                    "but have no proposal for it",
                    vc.round,
                )
                return
            asyncio.create_task(
                self._re_drive_after_view_change(vc.round)
            )

    async def _re_drive_after_view_change(self, round_num: int):
        try:
            committed = await self.re_drive_existing_round()
            logger.info(
                "Re-drove round %d as view-change leader (committed=%s)",
                round_num, committed,
            )
        except Exception as e:
            logger.warning(
                "Re-drive of round %d failed: %s", round_num, e
            )

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

    def _canonical_nodes(self) -> list[str]:
        return sorted({self.node_id, *self.peers})

    def _vc_leader(self, round_num: int) -> str:
        """Deterministic view-change leader: successor of the round-robin
        leader for ``round_num`` in the canonical node list. Every node
        computes the same value from (round, canonical list) alone — no
        shared mutable view state, so no divergent elections."""
        nodes = self._canonical_nodes()
        orig = nodes[round_num % len(nodes)]
        idx = nodes.index(orig)
        return nodes[(idx + 1) % len(nodes)]

    def _vk_map_with_self(self) -> dict[str, bytes]:
        return {**self._vk_map, self.node_id: self.engine.vk}

    def _mark_round_completed(self, round_num: int):
        if round_num in self._completed_rounds:
            return
        self._completed_rounds.add(round_num)
        print(
            f"[{self.node_id}] committed round {round_num} "
            f"(ledger height={self.ledger.get_height()})",
            flush=True,
        )
        event = self._round_completed.get(round_num)
        if event is not None:
            event.set()
        watcher = self._qc_watchers.pop(round_num, None)
        if watcher is not None:
            watcher.cancel()

    async def _qc_watch(self, round_num: int):
        try:
            event = self._round_completed.setdefault(round_num, asyncio.Event())
            if round_num in self._completed_rounds:
                return
            await asyncio.wait_for(
                event.wait(), timeout=2 * self.vote_timeout_s
            )
        except asyncio.TimeoutError:
            if round_num not in self._completed_rounds and self._running:
                try:
                    await self._send_view_change(round_num)
                except Exception as e:
                    logger.warning(
                        "Failed to send view change for round %d: %s",
                        round_num, e,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("QC watcher for round %d failed: %s", round_num, e)

    async def _send_view_change(self, round_num: int):
        if round_num in self._vc_sent:
            return
        self._vc_sent.add(round_num)
        from proto import hfedchain_pb2

        # Single signing path: the handler builds and signs the message;
        # receivers verify str(round) + highest_qc bytes.
        vc = self.view_change.create_view_change(
            node_id=self.node_id,
            new_view=self.view_change.current_view + 1,
            sk=self.engine.sk,
            round=round_num,
        )
        highest_qc_bytes = vc.highest_qc[1] if vc.highest_qc else b""
        msg = hfedchain_pb2.ConsensusMessage(
            view_change=hfedchain_pb2.ViewChangeMessage(
                node_id=vc.node_id,
                new_view=vc.new_view,
                highest_qc=highest_qc_bytes,
                signature=vc.signature,
                round=vc.round,
            )
        )
        self._pending_vc.setdefault(round_num, set()).add(self.node_id)
        await self._broadcast(msg)

    async def re_drive_existing_round(self) -> bool:
        if self._round_in_flight:
            logger.warning(
                "Skipping re-drive: a round is already in flight"
            )
            return False
        block = self.engine._last_proposal
        if block is None:
            return False
        if block.round in self._completed_rounds:
            return True
        delta = self._delta_by_round.get(block.round)
        if delta is None:
            # Should not happen: deltas are recorded when the round first
            # ran. Fall back to instance state rather than aborting.
            logger.warning(
                "Re-drive for round %d has no recorded delta; "
                "falling back to instance state", block.round,
            )
            delta = (self._last_delta_w, self._last_n_devices)
        delta_bytes, n_devices = delta
        if delta_bytes is None:
            # Neither the round's own delta nor the instance fallback
            # exists: driving would ship a None payload to the cloud.
            logger.warning(
                "No delta available for round %d; cannot re-drive",
                block.round,
            )
            return False
        self._round_in_flight = True
        try:
            await self.engine.start_round(block.round, is_leader=True)
            return await self._drive_consensus(
                block.round, block, delta_bytes, n_devices
            )
        finally:
            self._round_in_flight = False
            if (
                self._pending_gradients
                and len(self._pending_gradients) >= self.gradient_threshold
            ):
                # Gradients accumulated while this re-drive held the guard
                # hit _run_round's in-flight check and returned; re-trigger
                # the backlog now that the guard is released.
                asyncio.create_task(self._run_round())

    async def _run_round(self):
        if self._round_in_flight:
            return
        self._round_in_flight = True
        outcome = "stalled"
        try:
            # Backlog loop: triggers that arrived while this run was awaiting
            # (votes, STARK) hit the guard above and returned; re-check after
            # each attempt so they are not left unprocessed. Looping in-place
            # instead of asyncio.create_task keeps a single task and lets us
            # stop on a failed attempt (retrying would spin on the same
            # deterministic error).
            while True:
                outcome = await self._run_round_once()
                if outcome != "ran":
                    break
        finally:
            self._round_in_flight = False
        if (
            outcome == "stalled"
            and len(self._pending_gradients) >= self.gradient_threshold
        ):
            # A trigger arrived during this attempt's awaits, but the attempt
            # had already snapshotted pending and stalled below threshold.
            # The trigger's own _run_round() returned on the guard above, so
            # schedule the leftover backlog here. Never done after "failed":
            # failed attempts restore their gradients and would respawn
            # forever.
            asyncio.create_task(self._run_round())

    async def _run_round_once(self) -> str:
        """Consume pending gradients for one round attempt.

        Returns "ran" if a round ran (a backlog may still remain),
        "stalled" when there is nothing to run or the batch is below
        threshold (valid gradients stay pending for a later trigger), or
        "failed" when aggregation raised (valid gradients stay pending, no
        retry is scheduled to avoid spinning on a deterministic error).
        """
        if not self._pending_gradients:
            return "stalled"
        snapshot = list(self._pending_gradients)
        round_num = max(item.gradient.round for item in snapshot)

        valid_items: list[GradientWithProof] = []
        discarded: set[int] = set()
        for item in snapshot:
            if item.gradient.round != round_num:
                discarded.add(id(item))  # stale: superseded by round_num
                continue
            if (
                isinstance(item, GradientWithProof)
                and item.snark_proof is not None
                and self.variant in ("snark", "full")
            ):
                vk = self._vk_map.get(item.gradient.node_id)
                if vk is None or not await self.snark_verifier.verify(
                    item.snark_proof, b"", vk
                ):
                    discarded.add(id(item))
                    continue
            valid_items.append(item)

        # Drop stale + invalid (identity-based so gradients appended to
        # _pending_gradients during the awaits above are not clobbered).
        if discarded:
            self._pending_gradients = [
                it for it in self._pending_gradients if id(it) not in discarded
            ]

        if len(valid_items) < self.gradient_threshold:
            return "stalled"  # keep valid items for a later trigger

        valid_grads = [item.gradient for item in valid_items]
        try:
            np_grads = [gradient_to_f32(g.data) for g in valid_grads]
            selected = self.multikrum.select(np_grads, self.f)
            selected_grads = [valid_grads[i] for i in selected]
            delta = np.mean(
                np.stack([
                    gradient_to_f32(g.data)
                    for g in selected_grads
                ]),
                axis=0,
            )
            delta_bytes = delta.astype(np.float32).tobytes()
        except Exception as e:
            logger.error(
                "Aggregation failed for round %d (%d gradients, f=%d): %s",
                round_num, len(valid_grads), self.f, e,
            )
            return "failed"  # keep valid gradients pending for a later trigger

        # Delta computed: the valid gradients are consumed.
        consumed = {id(item) for item in valid_items}
        self._pending_gradients = [
            it for it in self._pending_gradients if id(it) not in consumed
        ]

        self._last_delta_w = delta_bytes
        self._last_n_devices = len(selected)
        # Bind the delta to its round so later rounds (or instance-state
        # corruption) cannot change what a re-drive of THIS round ships.
        self._delta_by_round[round_num] = (delta_bytes, len(selected))
        self._prune_delta_by_round(max(self.engine.round, round_num))
        accepted = [g.node_id for g in selected_grads]
        rejected = [
            valid_grads[i].node_id for i in range(len(valid_grads))
            if i not in selected
        ]
        total_adv = sum(1 for g in valid_grads if g.node_id.startswith("adv_"))
        rej_adv = sum(1 for g in selected_grads if g.node_id.startswith("adv_"))
        rej_honest = len(rejected) - (total_adv - rej_adv)
        agg = AggregateGradient(
            node_id=self.node_id,
            round=round_num,
            gradient=Gradient(
                node_id=self.node_id, round=round_num, data=delta.tolist()
            ),
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
        all_nodes = self._canonical_nodes()
        leader_idx = round_num % len(all_nodes)
        self._current_leader = all_nodes[leader_idx]
        is_leader = self._current_leader == self.node_id
        print(
            f"[{self.node_id}] round {round_num}: cluster={self.cluster_id} "
            f"leader={self._current_leader} "
            f"role={'leader' if is_leader else 'follower'}",
            flush=True,
        )

        if is_leader:
            await self._run_leader_consensus(round_num, accepted, rejected)
        else:
            await self.engine.start_round(round_num, is_leader=False)
        return "ran"

    def _prune_delta_by_round(self, current_round: int) -> None:
        """Keep only a small window of round deltas (current and previous)
        so _delta_by_round stays bounded."""
        cutoff = current_round - 1
        for stale in [r for r in self._delta_by_round if r < cutoff]:
            del self._delta_by_round[stale]

    async def _run_leader_consensus(self, round_num, accepted, rejected):
        # The round's OWN delta, recorded in _run_round_once before any
        # await here — later rounds may already have overwritten
        # _last_delta_w/_last_n_devices.
        delta = self._delta_by_round.get(round_num)
        if delta is None:
            logger.warning(
                "No recorded delta for round %d; falling back to instance "
                "state", round_num,
            )
            delta = (self._last_delta_w, self._last_n_devices)
        delta_bytes, n_devices = delta

        block = Block(
            round=round_num,
            gradient_hash=hashlib.sha256(delta_bytes).digest(),
            qc_commit=None,
            stark_proof=None,
            accepted_devices=list(accepted),
            rejected_devices=list(rejected),
            timestamp=time.time(),
            prev_hash=b"\x00" * 32,
            n=self.n, f=self.f,
        )
        if self.ledger.get_height() > 0:
            block = replace(
                block, prev_hash=self.ledger._entries[-1].block.hash
            )

        return await self._drive_consensus(
            round_num, block, delta_bytes, n_devices
        )

    async def _drive_consensus(
        self,
        round_num: int,
        block: Block,
        delta_bytes: Optional[bytes],
        n_devices: int,
    ) -> bool:
        # delta_bytes/n_devices are REQUIRED parameters: the caller binds
        # them to this round (from _delta_by_round), so no instance state
        # read here can ship another round's payload under this block.
        from proto import hfedchain_pb2

        await self.engine.start_round(round_num, is_leader=True)
        proposal = await self.engine.propose(block)
        if proposal is None:
            return False

        # Single-node: commit directly
        if not self.peers:
            signature = pki_sign(
                self.engine.sk,
                str(round_num).encode()
                + block.hash
                + MessageType.COMMIT.name.encode(),
            )
            qc = self.quorum_certifier.collect(
                round=round_num, block_hash=block.hash,
                msg_type=MessageType.COMMIT,
                signatures=[(self.node_id, signature)],
                quorum_size=1,
            )
            entry = await self.engine.on_qc_commit(
                qc, vk_map=self._vk_map_with_self()
            )
            if entry is None:
                return False
            self.ledger.append(entry.block)
            self._mark_round_completed(round_num)
            await self._send_to_cloud(entry.block, delta_bytes, n_devices)
            return True

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

        # The leader counts its own vote (quorum = self + followers); without
        # it, quorum 2 for n=3 would need 2/2 inbound peer votes (zero
        # fault tolerance). Converted to the proto shape _pending_votes uses.
        self_vote = await self.engine.on_prepare(block)
        qc_prepare = await self._collect_quorum_votes(
            round_num, block.hash, "prepare", quorum,
            self_vote=self._proto_vote(self_vote),
        )
        if qc_prepare is None:
            await self._send_view_change(round_num)
            return False

        qc_msg = hfedchain_pb2.ConsensusMessage(
            qc=hfedchain_pb2.QcBroadcast(
                leader_id=self.node_id,
                round=round_num,
                phase="pre_commit",
                quorum_certificate=pickle.dumps(qc_prepare),
            )
        )
        await self._broadcast(qc_msg)

        self_vote = await self.engine.on_pre_commit(qc_prepare)
        qc_pre_commit = await self._collect_quorum_votes(
            round_num, block.hash, "pre_commit", quorum,
            self_vote=self._proto_vote(self_vote),
        )
        if qc_pre_commit is None:
            await self._send_view_change(round_num)
            return False

        qc_msg = hfedchain_pb2.ConsensusMessage(
            qc=hfedchain_pb2.QcBroadcast(
                leader_id=self.node_id,
                round=round_num,
                phase="commit",
                quorum_certificate=pickle.dumps(qc_pre_commit),
            )
        )
        await self._broadcast(qc_msg)

        self_vote = await self.engine.on_commit(qc_pre_commit)
        qc_commit = await self._collect_quorum_votes(
            round_num, block.hash, "commit", quorum,
            self_vote=self._proto_vote(self_vote),
        )
        if qc_commit is None:
            await self._send_view_change(round_num)
            return False

        entry = await self.engine.on_qc_commit(
            qc_commit, vk_map=self._vk_map_with_self()
        )
        if entry is None:
            return False

        if self.variant in ("stark", "full"):
            proof = await self.stark_prover.generate_proof(entry.block)
            entry = LedgerEntry(
                block=replace(entry.block, stark_proof=proof),
                node_id=entry.node_id,
                stored_at=entry.stored_at,
            )

        self.ledger.append(entry.block)
        self._mark_round_completed(round_num)

        commit_msg = hfedchain_pb2.ConsensusMessage(
            qc=hfedchain_pb2.QcBroadcast(
                leader_id=self.node_id,
                round=round_num,
                phase="qc_commit",
                quorum_certificate=pickle.dumps(qc_commit),
            )
        )
        await self._broadcast(commit_msg)
        await self._send_to_cloud(entry.block, delta_bytes, n_devices)
        return True

    def _matching_votes(self, round_num, block_hash, phase) -> list:
        """Votes for this (round, block_hash, phase), deduped by node_id.

        Only the collector's peers (the voters) and the collector's own
        injected vote count: arbitrary ids registered via MQTT must not be
        able to form a quorum.
        """
        matching = []
        seen: set[str] = set()
        for v in self._pending_votes:
            if (v.round != round_num or v.block_hash != block_hash
                    or v.phase != phase):
                continue
            if v.node_id != self.node_id and v.node_id not in self.peers:
                continue
            if v.node_id in seen:
                continue
            seen.add(v.node_id)
            matching.append(v)
        return matching

    @staticmethod
    def _proto_vote(vote):
        """Convert a core VoteMessage (or None) to the proto shape used in
        _pending_votes."""
        if vote is None:
            return None
        from proto import hfedchain_pb2

        return hfedchain_pb2.VoteMessage(
            node_id=vote.node_id,
            round=vote.round,
            phase=vote.phase,
            block_hash=vote.block_hash,
            signature=vote.signature,
        )

    async def _collect_quorum_votes(self, round_num, block_hash, phase, quorum,
                                    self_vote=None):
        self._pending_votes = []
        self._vote_event.clear()
        if self_vote is not None:
            # Leader's own vote for this phase, injected after the reset so
            # it survives into this collection.
            self._pending_votes.append(self_vote)
            self._vote_event.set()

        phase_map = {
            "prepare": MessageType.PREPARE,
            "pre_commit": MessageType.PRE_COMMIT,
            "commit": MessageType.COMMIT,
        }
        msg_type = phase_map.get(phase)
        if msg_type is None:
            return None

        deadline = time.monotonic() + self.vote_timeout_s
        while True:
            matching = self._matching_votes(round_num, block_hash, phase)
            if len(matching) >= quorum:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "Quorum not reached for round %d phase %s: %d distinct "
                    "votes < quorum %d", round_num, phase, len(matching), quorum,
                )
                return None
            try:
                await asyncio.wait_for(
                    self._vote_event.wait(), timeout=min(0.5, remaining)
                )
                self._vote_event.clear()
            except asyncio.TimeoutError:
                pass

        signatures = [(v.node_id, v.signature) for v in matching[:quorum]]
        try:
            return self.quorum_certifier.collect(
                round=round_num,
                block_hash=block_hash,
                msg_type=msg_type,
                signatures=signatures,
                quorum_size=quorum,
                vk_map=self._vk_map_with_self(),
            )
        except Exception as e:
            logger.warning(
                "Quorum certificate failed for round %d phase %s "
                "(%d distinct votes, quorum %d): %s",
                round_num, phase, len(matching), quorum, e,
            )
            return None

    def _build_cluster_output(self, block: Block, delta_bytes: bytes,
                              n_devices: int):
        from proto import hfedchain_pb2

        return hfedchain_pb2.ClusterOutput(
            cluster_id=self.cluster_id,
            round=block.round,
            delta_w=delta_bytes,
            stark_proof=block.stark_proof.proof_bytes
            if block.stark_proof
            else None,
            qc_commit=pickle.dumps(block.qc_commit)
            if block.qc_commit
            else None,
            n_devices=n_devices,
            gradient_hash=block.gradient_hash,
        )

    async def _send_to_cloud(self, block: Block, delta_bytes: bytes,
                             n_devices: int):
        try:
            import grpc
            from proto import hfedchain_pb2_grpc

            async with grpc.aio.insecure_channel(self.cloud_address) as channel:
                stub = hfedchain_pb2_grpc.CloudAggregationStub(channel)
                output = self._build_cluster_output(block, delta_bytes, n_devices)
                await stub.SubmitClusterOutput(
                    output, timeout=self.vote_timeout_s + 5
                )
                print(
                    f"[{self.node_id}] submitted cluster={self.cluster_id} "
                    f"round={block.round} to cloud OK",
                    flush=True,
                )
        except Exception as e:
            print(f"[{self.node_id}] Cloud submission failed: {e}", flush=True)


if __name__ == "__main__":
    # `python -m services.fog_service` — same environment-driven startup as
    # `python -m services.entrypoint` with SERVICE=fog.
    import os

    from services.entrypoint import main as entrypoint_main

    os.environ.setdefault("SERVICE", "fog")
    asyncio.run(entrypoint_main())
