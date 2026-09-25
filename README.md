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

# Architecture

## 3-Tier Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│  EDGE TIER  (notebooks, Raspberry Pi, dispositivos IoHT) │
│                                                         │
│  Função: treinamento local do modelo ML, geração de     │
│          gradientes, prova SNARK opcional                │
│                                                         │
│  Protocolo: MQTT publish para Fog cluster               │
│  Tópico:   hfedchain/cluster/{cluster_id}/gradients     │
├─────────────────────────────────────────────────────────┤
│  FOG TIER  (VMs / servidores)                            │
│                                                         │
│  Função: verificação SNARK, agregação Multi-Krum,       │
│          consenso HotStuff BFT (3 fases), prova STARK,   │
│          ledger blockchain                               │
│                                                         │
│  Protocolo: gRPC bidirecional streaming (P2P),           │
│             gRPC unary (Fog→Cloud),                      │
│             MQTT subscribe (Edge→Fog)                    │
├─────────────────────────────────────────────────────────┤
│  CLOUD TIER  (VM / servidor)                             │
│                                                         │
│  Função: verificação STARK, FedAvg ponderado global,     │
│          atualização do modelo, ledger WORM,              │
│          detecção de convergência                        │
│                                                         │
│  Protocolo: gRPC unary (recebe de Fog),                  │
│             gRPC unary (distribui modelo)                │
└─────────────────────────────────────────────────────────┘
```

## Responsabilidades por Camada

### Edge
- Executar `MLP.forward()` e `MLP.backward()` com dados locais
- Opcionalmente gerar prova SNARK via `SnarkProver.generate_proof()`
- Assinar gradiente com chave Ed25519
- Publicar `GradientWithProof` serializado via MQTT

### Fog (cada nó no cluster)
- Subscrever tópico MQTT para receber gradientes dos Edges
- Verificar provas SNARK (variante `snark` ou `full`)
- Executar Multi-Krum para agregação robusta (`core/multikrum/`)
- Eleger líder via VRF (`core/vrf/`)
- Executar HotStuff 3-phase (`core/hotstuff/engine.py`):
  - Líder propõe bloco → seguidores votam → QC
  - Repetir para Prepare, PreCommit, Commit
- Opcionalmente gerar prova STARK do bloco (variante `stark` ou `full`)
- Armazenar bloco no ledger local (`core/ledger/store.py`)
- Enviar resultado agregado para Cloud via gRPC

### Cloud
- Receber `RegionalOutput` de cada Fog cluster
- Verificar prova STARK (se presente)
- Executar FedAvg ponderado por número de dispositivos
- Atualizar modelo global
- Armazenar em WORM ledger (`core/ledger/worm_store.py`)
- Detectar convergência
- Distribuir modelo atualizado para todos os clusters

## Fluxo de Dados por Rodada

```
ROUND r:

  Cloud ──broadcast model──► Fog ──broadcast model──► Edge
                                                         │
                                              Edge treina MLP
                                              (forward + backward)
                                                         │
                                              Opcional: SNARK proof
                                                         │
                                              Edge publica gradiente
                                              via MQTT ────────► Fog
                                                                  │
                                                    Fog: SNARK verify
                                                    Fog: Multi-Krum
                                                    Fog: VRF election
                                                    Fog: HotStuff (3 fases)
                                                    Fog: STARK proof (opcional)
                                                    Fog: ledger.append()
                                                                  │
                                              Fog envia resultado
                                              via gRPC ──────────► Cloud
                                                                      │
                                                        Cloud: STARK verify
                                                        Cloud: FedAvg
                                                        Cloud: WORM store
                                                        Cloud: convergência?
                                                        Cloud: novo modelo
```

## Componentes Internos do Fog

```
┌───────────────────────────────────────────────┐
│  FogService                                    │
│                                                 │
│  ┌─────────────┐  ┌───────────────────┐       │
│  │ MQTT Client  │  │ gRPC Server       │       │
│  │ (subscriber) │  │ (port 50051)      │       │
│  └──────┬──────┘  └────────┬──────────┘       │
│         │                  │                    │
│         ▼                  ▼                    │
│  ┌──────────────────────────────────────┐      │
│  │  Consensus Engine                    │      │
│  │  ┌──────────┐ ┌────────┐ ┌───────┐  │      │
│  │  │ Multi-   │ │ VRF    │ │HotStuff│  │      │
│  │  │ Krum     │ │Election│ │Engine  │  │      │
│  │  └──────────┘ └────────┘ └───────┘  │      │
│  │  ┌──────────┐ ┌────────┐ ┌───────┐  │      │
│  │  │ SNARK    │ │ STARK  │ │Ledger │  │      │
│  │  │ Verifier │ │ Prover │ │Store  │  │      │
│  │  └──────────┘ └────────┘ └───────┘  │      │
│  └──────────────────────────────────────┘      │
│                                                 │
│  ┌────────────────┐  ┌───────────────────┐      │
│  │ gRPC Client    │  │ gRPC Client       │      │
│  │ (P2P peers)    │  │ (→ Cloud)         │      │
│  └────────────────┘  └───────────────────┘      │
└───────────────────────────────────────────────┘
```

