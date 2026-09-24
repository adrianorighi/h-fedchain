"""Pool compartilhado de processos para a camada zk-SNARK.

Gera provas (Edge) e verifica (Fog) fora do loop asyncio, aproveitando
multiples nucleos via ProcessPoolExecutor. O custo das operacoes em BN254
(py_ecc) e CPU-bound e dominado pelo GIL no processo principal, por isso a
execucao paralela ocorre em subprocessos.

As constantes SNARK_PROVE_MS / SNARK_VERIFY_MS sao os valores medidos pelo
microbenchmark (results/microbenchmark/snark.json, implementacao de
referencia) e sao usadas para projetar o custo integral da verificacao SNARK
por rodada sem executar a verificacao completa (estrategia B2 do artigo).
"""
import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from hfc_types.crypto import SnarkProof
from hfc_types.messages import Gradient
from zkp.snark import SnarkProver, SnarkVerifier

SNARK_PROVE_MS = 227.904  # microbenchmark: geracao de prova (Edge)
SNARK_VERIFY_MS = 18190.892  # microbenchmark: verificacao de prova (Fog)

_pool: Optional[ProcessPoolExecutor] = None
_pool_workers = max(2, min(12, os.cpu_count() or 1))


def get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=_pool_workers)
    return _pool


def shutdown_pool():
    global _pool
    if _pool is not None:
        _pool.shutdown(cancel_futures=False)
        _pool = None


def prove_sync(gradient: Gradient, model_hash: bytes, sk: bytes) -> SnarkProof:
    """Gera a prova SNARK para um gradiente (executa em subprocesso)."""
    return asyncio.run(SnarkProver().generate_proof(gradient, model_hash, sk))


def verify_sync(proof: SnarkProof, model_hash: bytes, vk_bytes: bytes = b"") -> bool:
    """Verifica a prova SNARK (executa em subprocesso)."""
    return asyncio.run(SnarkVerifier().verify(proof, model_hash, vk_bytes))
