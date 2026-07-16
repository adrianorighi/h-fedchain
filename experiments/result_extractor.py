import asyncio
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from experiments.metrics import MetricsCollector

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

class ResultExtractor:
    def __init__(self):
        self.all_metrics = {}

    def run_all(self):
        from experiments.scenario_1_nominal import run as run_1
        from experiments.scenario_2_scalability import run as run_2
        from experiments.scenario_3_adversarial import run as run_3
        from experiments.scenario_4_interregional import run as run_4
        from experiments.scenario_5_zkp_overhead import run as run_5

        scenarios = [
            ("1_nominal", run_1),
            ("2_scalability", run_2),
            ("3_adversarial", run_3),
            ("4_interregional", run_4),
            ("5_zkp_overhead", run_5),
        ]
        for name, run_fn in scenarios:
            print(f"\n=== Executing scenario {name} ===")
            collector = asyncio.run(run_fn())
            self.all_metrics[name] = collector.all_metrics() if hasattr(collector, 'all_metrics') else {}
            self._save_json(name, self.all_metrics[name])
        self._generate_latex_table()

    def _save_json(self, name: str, data: dict):
        path = RESULTS_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  -> Saved: {path}")

    def _generate_latex_table(self):
        headers = ["Cenário", "Latência (s)", "Sucesso Cons.",
                   "Consenso (ms)", "Bloco (B)", "Detecção", "FPR",
                   "Uniform. VRF", "Audit."]
        rows = []
        for name, metrics in self.all_metrics.items():
            row = [
                name.replace("_", " ").title(),
                f"{metrics.get('avg_latency', 0):.3f}",
                f"{metrics.get('consensus_success', 0):.0%}",
                f"{metrics.get('consensus_time_ms', 0):.1f}",
                f"{metrics.get('block_size_bytes', 0):.0f}",
                f"{metrics.get('detection_rate', 0):.0%}",
                f"{metrics.get('false_positive_rate', 0):.1%}",
                f"{metrics.get('vrf_uniformity', 0):.3f}",
                f"{metrics.get('compliance', 0):.0%}",
            ]
            rows.append(" & ".join(row) + r" \\")
        latex = r"""%% Tabela gerada automaticamente por experiments/result_extractor.py
\begin{table}[htbp]
\caption{Resultados experimentais da arquitetura H-FedChain.}
\label{tab:resultados}
\centering
\begin{tabular}{lcccccccc}
\toprule
""" + " & ".join(headers) + r""" \\ \midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
        path = RESULTS_DIR / "resultados_tabela.tex"
        with open(path, "w") as f:
            f.write(latex)
        print(f"\n-> LaTeX table: {path}")


if __name__ == "__main__":
    extractor = ResultExtractor()
    extractor.run_all()
