import asyncio
import time
from typing import Optional
from simulator.cluster import Cluster
from simulator.interregional import InterRegionalManager
from simulator.cloud import CloudComponent
from monitoring.tracer import Tracer
from monitoring.system_metrics import SystemMetrics


class ExperimentRunner:
    """
    Multi-cluster runner.
    """
    def __init__(self, clusters: list[Cluster],
                 cloud: CloudComponent,
                 f: int = 1,
                 interregional_latency_ms: float = 50.0,
                 jitter_ms: float = 0.0,
                 tracer: Optional[Tracer] = None):
        self.clusters = clusters
        self.interregional = InterRegionalManager(
            clusters, n=len(clusters), f=f,
            interregional_latency_ms=interregional_latency_ms,
            jitter_ms=jitter_ms,
        )
        self.cloud = cloud
        self.tracer = tracer
        self.metrics: list[dict] = []
        self._sys = SystemMetrics()

    async def run_round(self, round_num: int) -> dict:
        tracer = self.tracer
        if tracer is not None:
            ctx = tracer.span("experiment_run_round", "experiment_runner", round_num=round_num)
            ctx.__enter__()
        else:
            ctx = None
        try:
            return await self._run_round_impl(round_num)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    async def _run_round_impl(self, round_num: int) -> dict:
        t_start = time.time()
        global_output = await self.interregional.run_round(round_num)
        result = await self.cloud.process(global_output)
        elapsed = time.time() - t_start
        # L_rodada inter-regional: soma das latências de pipeline dos
        # clusters (excluem a geração local de gradientes/provas SNARK na
        # Edge) + estágios inter-regionais medidos.
        cluster_ms = sum(
            r.pipeline_latency_ms for r in getattr(global_output, "regionals", [])
        )
        inter_ms = sum(self.interregional.stage_times.values())
        result["latency"] = (cluster_ms + inter_ms) / 1000
        result["wall_clock_s"] = elapsed
        result["consensus_time_ms"] = cluster_ms + inter_ms
        result["block_size_bytes"] = len(global_output.delta_w_inter) if global_output.delta_w_inter else 0
        result["qc_emitted"] = global_output.n_active_clusters > 0
        result["stage_time_ms"] = self.interregional.stage_times
        result["leader"] = self.interregional.last_leader_id
        proof_bytes = len(global_output.pi_inter.proof_bytes) if global_output.pi_inter else 0
        regionals = getattr(global_output, "regionals", None) or []
        # Proof CPU/size: soma das rodadas regionais (SNARK prove + STARK gen
        # da fog) + STARK inter-regional; verificação = regional
        # (verify_cpu_ms do InterRegionalManager, fora de stage_times p/ não
        # vazar na latência) + cloud (pi_inter).
        result["proof_gen_cpu_ms"] = (
            sum(getattr(r, "proof_gen_cpu_ms", 0.0) for r in regionals)
            + self.interregional.stage_times.get("stark_gen_inter", 0.0)
        )
        result["proof_verify_cpu_ms"] = (
            self.interregional.verify_cpu_ms.get("stark_verify_regional", 0.0)
            + result.get("stark_verify_cloud_ms", 0.0)
        )
        result["proof_size_bytes"] = (
            sum(getattr(r, "proof_size_bytes", 0) for r in regionals)
            + proof_bytes
        )
        result["bytes_edge_fog"] = sum(
            getattr(r, "bytes_edge_fog", 0) for r in regionals
        )
        result["bytes_fog_intra"] = sum(
            getattr(r, "bytes_fog_intra", 0) for r in regionals
        )
        result["bytes_fog_inter"] = getattr(global_output, "bytes_fog_inter", 0)
        result["bytes_fog_cloud"] = (
            result["block_size_bytes"] + proof_bytes
            + global_output.n_active_clusters * 64
        )
        result["network_bytes"] = (
            result["bytes_edge_fog"] + result["bytes_fog_intra"]
            + result["bytes_fog_inter"] + result["bytes_fog_cloud"]
        )
        # comm_overhead_bytes: mode-dependent (multi=parcela cloud;
        # single=total completo). Total unificado = network_bytes.
        result["comm_overhead_bytes"] = result["bytes_fog_cloud"]
        result["loss"] = self.interregional.last_loss
        result["model_accuracy"] = 0.0  # inter-regional has no ground truth
        result["snark_proofs_total"] = sum(
            r.snark_proofs_total for r in global_output.regionals
        ) if getattr(global_output, "regionals", None) else 0
        result["snark_verify_projected_ms"] = sum(
            r.snark_verify_projected_ms for r in global_output.regionals
        ) if getattr(global_output, "regionals", None) else 0.0
        result["snark_sampled_verified"] = 1 if getattr(
            global_output, "regionals", None
        ) and any(r.snark_proofs_total > 0 for r in global_output.regionals) else 0
        result["snark_sampled_passed"] = min(
            (1 if r.snark_sampled_passed else 0 for r in global_output.regionals),
            default=1,
        ) if getattr(global_output, "regionals", None) else 1
        result["snark_attempted"] = result["snark_proofs_total"]
        result["snark_passed"] = result["snark_proofs_total"] if result["snark_sampled_passed"] else 0
        sys_snap = self._sys.snapshot()
        result["cpu_percent"] = sys_snap["cpu_percent"]
        result["memory_rss_bytes"] = sys_snap["memory_rss_bytes"]
        self.metrics.append(result)
        return result

    async def run_experiment(self, num_rounds: int) -> list[dict]:
        self._sys.cpu_percent()  # arm baseline so round 1 measures since experiment start
        for rnd in range(1, num_rounds + 1):
            await self.run_round(rnd)
        return self.metrics
