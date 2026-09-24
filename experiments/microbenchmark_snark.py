"""Microbenchmark da camada zk-SNARK (Edge -> Fog).

Mede o custo real de geracao (prove, no dispositivo Edge) e verificacao
(verify, no Fog Node) de provas zk-SNARK sobre gradientes sinteticos
representativos, reportando media e desvio padrao.

O custo das operacoes e dominado pelas operacoes de grupo em BN254
(bn128), independente da dimensao do gradiente. Rodada sob demanda;
nao integra o batch de 105 execucoes nem a suíte de testes.
"""
import asyncio
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hfc_types.messages import Gradient
from zkp.snark import SnarkProver, SnarkVerifier

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "microbenchmark"

REFERENCE_TOPOLOGY = {
    "clusters": 4,
    "nodes_per_cluster": 5,
    "devices_per_cluster": 50,
    "total_devices": 200,
}


def _rep_gradient(seed: int = 42) -> Gradient:
    rng = np.random.default_rng(seed)
    return Gradient(node_id="d0", round=0, data=rng.standard_normal(10).tolist())


async def _bench_prove(prover: SnarkProver, grad: Gradient, n: int) -> list[float]:
    model_hash = b"\x00" * 32
    sk = b"\x01" * 32
    await prover.generate_proof(grad, model_hash, sk)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        proof = await prover.generate_proof(grad, model_hash, sk)
        times.append(time.perf_counter() - t0)
    return times, len(proof.proof_bytes)


async def _bench_verify(verifier: SnarkVerifier, grad: Gradient, n: int) -> list[float]:
    prover = SnarkProver()
    model_hash = b"\x00" * 32
    proof = await prover.generate_proof(grad, model_hash, b"\x01" * 32)
    ok = await verifier.verify(proof, model_hash, b"")
    if not ok:
        raise RuntimeError("prova de referencia rejeitada pelo verificador")
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        await verifier.verify(proof, model_hash, b"")
        times.append(time.perf_counter() - t0)
    return times


def _proj(mean: float, count: int) -> dict:
    return {
        "per_unit_ms": mean * 1000,
        "serial_total_s": mean * count,
        "parallel_total_s": mean,
    }


async def run(n_prove: int = 10, n_verify: int = 3) -> dict:
    grad = _rep_gradient()

    prove_times, proof_bytes = await _bench_prove(SnarkProver(), grad, n_prove)
    verify_times = await _bench_verify(SnarkVerifier(), grad, n_verify)

    prove_mean = statistics.mean(prove_times)
    prove_std = statistics.stdev(prove_times) if len(prove_times) > 1 else 0.0
    verify_mean = statistics.mean(verify_times)
    verify_std = statistics.stdev(verify_times) if len(verify_times) > 1 else 0.0

    devices = REFERENCE_TOPOLOGY["total_devices"]
    per_cluster = REFERENCE_TOPOLOGY["devices_per_cluster"]

    report = {
        "gradient_dim": len(grad.data),
        "n_prove": n_prove,
        "n_verify": n_verify,
        "prove_ms": {"mean": prove_mean * 1000, "std": prove_std * 1000},
        "verify_ms": {"mean": verify_mean * 1000, "std": verify_std * 1000},
        "proof_bytes": proof_bytes,
        "projection_reference_topology": {
            **REFERENCE_TOPOLOGY,
            "edge_prove_per_round_s": {
                "serial_200_devices": _proj(prove_mean, devices)["serial_total_s"],
                "parallel_200_devices": _proj(prove_mean, devices)["parallel_total_s"],
            },
            "fog_verify_per_node_round_s": _proj(verify_mean, per_cluster)["serial_total_s"],
        },
    }
    return report


def main(n_prove: int, n_verify: int):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(run(n_prove=n_prove, n_verify=n_verify))

    p = report["prove_ms"]
    v = report["verify_ms"]
    print("=== Microbenchmark zk-SNARK (implementacao de referencia) ===")
    print(f"Gradiente representativo: {report['gradient_dim']} dimensoes")
    print(f"prove  (Edge): {p['mean']:.1f} ms +/- {p['std']:.1f} ms  "
          f"(n={report['n_prove']}, prova de {report['proof_bytes']} B)")
    print(f"verify (Fog) : {v['mean']:.1f} ms +/- {v['std']:.1f} ms  "
          f"(n={report['n_verify']})")
    print("\nProjecao na topologia de referencia do Cenario 5 "
          f"({report['projection_reference_topology']['clusters']} clusters x "
          f"{report['projection_reference_topology']['devices_per_cluster']} "
          f"dispositivos, {report['projection_reference_topology']['total_devices']} "
          f"dispositivos no total):")
    proj = report["projection_reference_topology"]
    print(f"  Edge  — geracao de provas/rodada: "
          f"{proj['edge_prove_per_round_s']['serial_200_devices']:.1f} s "
          f"(serial) / {proj['edge_prove_per_round_s']['parallel_200_devices']*1000:.0f} ms "
          f"(paralelo)")
    print(f"  Fog   — verificacao por no/rodada: "
          f"{proj['fog_verify_per_node_round_s']:.1f} s (serial, "
          f"{report['projection_reference_topology']['devices_per_cluster']} "
          f"provas por no)")

    out = RESULTS_DIR / "snark.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n-> Resultado: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microbenchmark zk-SNARK")
    parser.add_argument("--n-prove", type=int, default=10)
    parser.add_argument("--n-verify", type=int, default=3)
    args = parser.parse_args()
    main(n_prove=args.n_prove, n_verify=args.n_verify)
