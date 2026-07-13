# H-FedChain Prototype

Hierarchical Edge–Fog–Cloud architecture with HotStuff BFT consensus for
secure federated learning in IoHT.

## Project Structure

- `hfc_types/` — shared dataclasses (messages, blocks, crypto types)
- `core/` — portable algorithm packages (HotStuff, Multi-Krum, VRF, Ledger)
- `zkp/` — wrapper interfaces for PySNARK (zk-SNARK) and ethSTARK (zk-STARK)
- `simulator/` — asyncio actor-based simulator for Fog intra-cluster
- `experiments/` — 5 evaluation scenarios from the paper
- `tests/` — unit and integration tests

## Quick Start

```bash
./scripts/setup.sh
source .venv/bin/activate
pytest -v
```

## Running Experiments

```bash
mkdir -p results
python -m experiments.scenario_1_nominal
python -m experiments.scenario_2_scalability
python -m experiments.scenario_3_adversarial
python -m experiments.scenario_4_interregional
python -m experiments.scenario_5_zkp_overhead
```

Results are written to `results/*.json`.

## Architecture

See `docs/superpowers/specs/2026-07-13-h-fedchain-prototype-design.md`.
