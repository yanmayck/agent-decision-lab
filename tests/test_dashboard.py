"""Unit tests for Experiment 06 Interactive Dashboard generator."""

import json
from pathlib import Path
import unittest

from experiments.generate_dashboard import generate_dashboard


class TestInteractiveDashboard(unittest.TestCase):
    """Validates the structure, integrity, and offline functionality of dashboard.html."""

    @classmethod
    def setUpClass(cls):
        # Run dashboard generation to ensure output is up to date
        generate_dashboard()
        cls.dashboard_path = Path("experiments/results/dashboard.html")
        cls.data_path = Path("experiments/results/experiment_06_data.json")

    def test_dashboard_file_exists_and_is_standalone(self):
        self.assertTrue(self.dashboard_path.exists(), "dashboard.html não foi encontrado.")
        content = self.dashboard_path.read_text(encoding="utf-8")
        self.assertGreater(len(content), 1_000_000, "dashboard.html deve ter mais de 1MB devido ao Plotly embutido.")
        # Ensure Plotly JS is embedded inline
        self.assertIn("Plotly.newPlot", content, "O script do dashboard deve inicializar gráficos Plotly.")
        # Ensure offline capability: no external script tags
        self.assertNotIn('<script src="http', content, "Não deve carregar scripts externos via HTTP para garantir 100% de funcionamento offline.")
        self.assertNotIn('<script src="https', content, "Não deve carregar scripts externos via HTTPS para garantir 100% de funcionamento offline.")

    def test_all_runs_present_in_dashboard(self):
        content = self.dashboard_path.read_text(encoding="utf-8")
        with open(self.data_path, encoding="utf-8") as f:
            runs = json.load(f)

        self.assertEqual(len(runs), 15, "Experimento 06 deve conter exatamente 15 execuções.")

        # Validate that all conditions and sizes are present in text/scripts
        for cond in ["Sem Descrição", "Descrição Curta", "Descrição Detalhada"]:
            self.assertIn(cond, content, f"Condição {cond} deve estar presente no dashboard.")

        for size in [5, 10, 20, 50, 100]:
            self.assertIn(str(size), content, f"Tamanho {size} deve estar presente no dashboard.")

    def test_token_differentiation_and_metrics(self):
        with open(self.data_path, encoding="utf-8") as f:
            runs = json.load(f)

        content = self.dashboard_path.read_text(encoding="utf-8")

        # Check token category headers in dashboard
        self.assertIn("Tokens Entrada", content)
        self.assertIn("Tokens Saída", content)
        self.assertIn("Tokens Raciocínio", content)
        self.assertIn("Tokens Ferramentas (Est.)", content)

        # Check each run's tool_result_tokens is populated and consistent
        for r in runs:
            self.assertIn("tool_result_tokens", r)
            self.assertGreater(r["tool_result_tokens"], 0)
            self.assertIn("tool_result_bytes", r)
            self.assertGreater(r["tool_result_bytes"], 0)

    def test_filter_and_table_elements_present(self):
        content = self.dashboard_path.read_text(encoding="utf-8")
        self.assertIn('id="filterCond"', content)
        self.assertIn('id="filterSize"', content)
        self.assertIn('id="filterLimit"', content)
        self.assertIn('id="filterSearch"', content)
        self.assertIn('id="resultsTable"', content)
        self.assertIn('id="tableBody"', content)


if __name__ == "__main__":
    unittest.main()
