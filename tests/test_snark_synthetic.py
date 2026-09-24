"""Testes da Estratégia B2 — SNARK real na Edge (prove) + verificação
amostrada na Fog (verify), no modo sintético.

Evita executar a verificação de referência de ~18,2 s (configurada via
`snark_sample_verify=False`); a correção da verificação é coberta pelo
microbenchmark e pelos testes de integração com amostragem real.
"""
import asyncio

import pytest

from simulator.orchestrator import Orchestrator
from simulator.snark_worker import SNARK_VERIFY_MS


def _orch(variant: str, devices: int = 4, **kw):
    return Orchestrator(
        num_clusters=1,
        nodes_per_cluster=5,
        devices_per_cluster=devices,
        f=1,
        latency_ms=10.0,
        variant=variant,
        snark_prove=True,
        snark_sample_verify=False,
        **kw,
    )


@pytest.mark.asyncio
async def test_synthetic_full_generates_real_proofs():
    orch = _orch("full")
    result = await orch.run_experiment(num_rounds=1, warmup=0)
    m = result.round_metrics[-1]
    assert m["snark_proofs_total"] == 4
    assert m["snark_proofs_total"] > 0
    assert m["snark_required"] is True
    assert m["snark_verify_projected_ms"] == pytest.approx(
        4 * SNARK_VERIFY_MS, rel=1e-6
    )


@pytest.mark.asyncio
async def test_synthetic_no_zkp_has_no_proofs():
    orch = _orch("no_zkp")
    result = await orch.run_experiment(num_rounds=1, warmup=0)
    m = result.round_metrics[-1]
    assert m["snark_proofs_total"] == 0
    assert m["snark_verify_projected_ms"] == 0.0
    assert m["snark_required"] is False


@pytest.mark.asyncio
async def test_synthetic_full_proofs_are_pickleable():
    """Provas anexadas não devem quebrar a serialização (comm_overhead)."""
    orch = _orch("full")
    orch.setup()
    grads = await orch._generate_gradients(0)
    assert all(g.snark_proof is not None for g in grads)
    import pickle
    for g in grads:
        assert len(pickle.dumps(g)) > 900


@pytest.mark.asyncio
async def test_metrics_collector_exposes_snark_projection():
    from experiments.metrics import MetricsCollector
    orch = _orch("full")
    result = await orch.run_experiment(num_rounds=2, warmup=0)
    mc = MetricsCollector()
    for m in result.round_metrics:
        mc.add_round(m)
    all_m = mc.all_metrics()
    assert all_m["snark_proofs_total"] == 4
    assert all_m["snark_verify_projected_ms"] > 0
