#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d .venv ]; then
    echo "Virtual env not found. Run scripts/setup.sh first."
    exit 1
fi

source .venv/bin/activate

echo "=========================================="
echo "  Validação de Experimentos — H-FedChain"
echo "=========================================="
echo ""

echo "=== Testes (pytest) ==="
python -m pytest -v --tb=short
echo ""

echo "=== Cenário 1: Nominal ==="
timeout 180 python -m experiments.scenario_1_nominal || echo "FAIL"
echo ""

echo "=== Cenário 2: Escalabilidade ==="
timeout 180 python -m experiments.scenario_2_scalability || echo "FAIL"
echo ""

echo "=== Cenário 3: Adversarial ==="
timeout 300 python -m experiments.scenario_3_adversarial || echo "FAIL"
echo ""

echo "=== Cenário 4: Inter-Regional ==="
timeout 300 python -m experiments.scenario_4_interregional || echo "FAIL"
echo ""

echo "=== Cenário 5: ZKP Overhead ==="
timeout 120 python -m experiments.scenario_5_zkp_overhead || echo "FAIL"
echo ""

echo "=== Tabela LaTeX ==="
python -m experiments.result_extractor
cat experiments/results/resultados_tabela.tex
echo ""

echo "=== FIM — Todos os cenários executados ==="
