#!/bin/bash
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/experiments/results"
QUICK=${QUICK:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
INCLUDE_REAL=${INCLUDE_REAL:-0}
INCLUDE_COMPARISON=${INCLUDE_COMPARISON:-0}
INCLUDE_LATEX=${INCLUDE_LATEX:-0}

cd "$SCRIPT_DIR"

if [ ! -d .venv ]; then
    echo "ERRO: .venv não encontrado. Execute scripts/setup.sh primeiro."
    exit 1
fi
source .venv/bin/activate
mkdir -p "$RESULTS_DIR"

run() {
    local label="$1" module="$2" timeout_sec="$3"
    shift 3
    local flags="$@"
    local logfile="$RESULTS_DIR/${label// /_}.log"
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  [$label]"
    echo "  Comando: python -m $module $flags"
    echo "  Timeout: ${timeout_sec}s"
    echo "════════════════════════════════════════════════"
    local start end elapsed
    start=$(date +%s)
    timeout "$timeout_sec" python -m "$module" $flags > "$logfile" 2>&1
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))
    if [ $rc -eq 0 ]; then
        echo "  ✓ PASS (${elapsed}s)  Log: $logfile"
        RUN_PASSED+=("$label")
    else
        echo "  ✗ FAIL (${elapsed}s)  Log: $logfile"
        RUN_FAILED+=("$label")
        [ "$CONTINUE_ON_ERROR" -eq 0 ] && { echo "Abortando."; exit 1; }
    fi
}

run_test() {
    local label="$1"
    local logfile="$RESULTS_DIR/${label// /_}.log"
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  [$label]"
    echo "════════════════════════════════════════════════"
    local start end elapsed
    start=$(date +%s)
    python -m pytest -v --tb=short > "$logfile" 2>&1
    local rc=$?
    end=$(date +%s)
    elapsed=$((end - start))
    if [ $rc -eq 0 ]; then
        echo "  ✓ PASS (${elapsed}s)  Log: $logfile"
        RUN_PASSED+=("$label")
    else
        echo "  ✗ FAIL (${elapsed}s)  Log: $logfile"
        RUN_FAILED+=("$label")
        [ "$CONTINUE_ON_ERROR" -eq 0 ] && { echo "Abortando."; exit 1; }
    fi
}

summary() {
    local end elapsed
    end=$(date +%s)
    elapsed=$((end - START_GLOBAL))
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  RESUMO  (${elapsed}s total)"
    echo "════════════════════════════════════════════════"
    echo "  PASS: ${#RUN_PASSED[@]}  FAIL: ${#RUN_FAILED[@]}"
    for p in "${RUN_PASSED[@]}"; do echo "    ✓ $p"; done
    for f in "${RUN_FAILED[@]}"; do echo "    ✗ $f"; done
    [ ${#RUN_FAILED[@]} -eq 0 ] && echo "  Todos passaram." \
        || echo "  ${#RUN_FAILED[@]} falha(s). Logs em $RESULTS_DIR"
    echo "════════════════════════════════════════════════"
}

toggle() { local v="$1"; [ "$v" -eq 0 ] && echo 1 || echo 0; }

menu() {
    local rl cl ll xl ql
    [ "$INCLUDE_REAL" -eq 1 ] && rl="SIM" || rl="NÃO"
    [ "$INCLUDE_COMPARISON" -eq 1 ] && cl="SIM" || cl="NÃO"
    [ "$INCLUDE_LATEX" -eq 1 ] && ll="SIM" || ll="NÃO"
    [ "$CONTINUE_ON_ERROR" -eq 1 ] && xl="SIM" || xl="NÃO"
    [ "$QUICK" -eq 1 ] && ql="SIM" || ql="NÃO"
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║         H-FedChain — Execução de Cenários           ║"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  1)  Cenário 1 — Nominal                            ║"
    echo "║  2)  Cenário 1 — Nominal (--real)                   ║"
    echo "║  3)  Cenário 2 — Escalabilidade                     ║"
    echo "║  4)  Cenário 3 — Adversarial                        ║"
    echo "║  5)  Cenário 3 — Adversarial (--real)               ║"
    echo "║  6)  Cenário 4 — Inter-Regional                     ║"
    echo "║  7)  Cenário 5 — ZKP Overhead                       ║"
    echo "║  8)  Comparison — Nominal                           ║"
    echo "║  9)  Comparison — Escalabilidade                    ║"
    echo "║ 10)  Tabela LaTeX (result_extractor)                ║"
    echo "║ 11)  Rodar testes (pytest)                          ║"
    echo "║ 12)  TODOS os cenários (sequencial)                 ║"
    echo "║ 13)  Validação rápida (QUICK=1)                     ║"
    echo "║  0)  Sair                                           ║"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  r)  --real             [${rl}]                        ║"
    echo "║  c)  --comparison       [${cl}]                        ║"
    echo "║  l)  --latex            [${ll}]                        ║"
    echo "║  x)  continuar em erro  [${xl}]                        ║"
    echo "║  q)  QUICK              [${ql}]                        ║"
    echo "╚══════════════════════════════════════════════════════╝"
}

reset_run() {
    START_GLOBAL=$(date +%s)
    RUN_PASSED=()
    RUN_FAILED=()
}

while true; do
    menu
    read -rp "Opção: " choice || break
    case "$choice" in
        0|q|Q) echo ""; exit 0 ;;
        r|R) INCLUDE_REAL=$(toggle "$INCLUDE_REAL") ;;
        c|C) INCLUDE_COMPARISON=$(toggle "$INCLUDE_COMPARISON") ;;
        l|L) INCLUDE_LATEX=$(toggle "$INCLUDE_LATEX") ;;
        x|X) CONTINUE_ON_ERROR=$(toggle "$CONTINUE_ON_ERROR") ;;
        qq|QQ) QUICK=$(toggle "$QUICK") ;;
        1) reset_run
            run "Cenario1_Nominal" \
                "experiments.scenario_1_nominal" 180 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        2) reset_run
            run "Cenario1_Nominal_real" \
                "experiments.scenario_1_nominal" 300 --real $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        3) reset_run
            run "Cenario2_Escalabilidade" \
                "experiments.scenario_2_scalability" 180 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        4) reset_run
            run "Cenario3_Adversarial" \
                "experiments.scenario_3_adversarial" 300 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        5) reset_run
            run "Cenario3_Adversarial_real" \
                "experiments.scenario_3_adversarial" 600 --real $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        6) reset_run
            run "Cenario4_InterRegional" \
                "experiments.scenario_4_interregional" 300 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        7) reset_run
            run "Cenario5_ZKP_Overhead" \
                "experiments.scenario_5_zkp_overhead" 120 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        8) reset_run
            run "Comparison_Nominal" \
                "experiments.comparison" 300 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        9) reset_run
            run "Comparison_Escalabilidade" \
                "experiments.comparison" 300 --scenario scalability $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            summary ;;
        10) reset_run
            run "Tabela_LaTeX" \
                "experiments.result_extractor" 30
            summary ;;
        11) reset_run
            run_test "Testes_pytest"
            summary ;;
        12) reset_run
            echo ""
            echo "════════════════════════════════════════════════"
            echo "  Executando TODOS os cenários"
            echo "════════════════════════════════════════════════"
            run_test "Testes_pytest"
            run "Cenario1_Nominal" \
                "experiments.scenario_1_nominal" 180 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            run "Cenario2_Escalabilidade" \
                "experiments.scenario_2_scalability" 180 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            run "Cenario3_Adversarial" \
                "experiments.scenario_3_adversarial" 300 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            run "Cenario4_InterRegional" \
                "experiments.scenario_4_interregional" 300 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            run "Cenario5_ZKP_Overhead" \
                "experiments.scenario_5_zkp_overhead" 120 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            [ "$INCLUDE_REAL" -eq 1 ] && {
                run "Cenario1_Nominal_real" \
                    "experiments.scenario_1_nominal" 300 --real $( [ "$QUICK" -eq 1 ] && echo "--quick" )
                run "Cenario3_Adversarial_real" \
                    "experiments.scenario_3_adversarial" 600 --real $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            }
            [ "$INCLUDE_COMPARISON" -eq 1 ] && {
                run "Comparison_Nominal" \
                    "experiments.comparison" 300 $( [ "$QUICK" -eq 1 ] && echo "--quick" )
                run "Comparison_Escalabilidade" \
                    "experiments.comparison" 300 --scenario scalability $( [ "$QUICK" -eq 1 ] && echo "--quick" )
            }
            [ "$INCLUDE_LATEX" -eq 1 ] &&
                run "Tabela_LaTeX" \
                    "experiments.result_extractor" 30
            summary ;;
        13) reset_run
            echo ""
            echo "════════════════════════════════════════════════"
            echo "  Validação rápida"
            echo "════════════════════════════════════════════════"
            run "Cenario1_Nominal" \
                "experiments.scenario_1_nominal" 60 --quick
            run "Cenario2_Escalabilidade" \
                "experiments.scenario_2_scalability" 60 --quick
            run "Cenario3_Adversarial" \
                "experiments.scenario_3_adversarial" 120 --quick
            run "Cenario4_InterRegional" \
                "experiments.scenario_4_interregional" 120 --quick
            run "Cenario5_ZKP_Overhead" \
                "experiments.scenario_5_zkp_overhead" 60 --quick
            summary ;;
        *) echo "Opção inválida." ;;
    esac
done
