"""Unit tests for Experiment 08 (Decision Theory & Expected Cost Optimization).

Validates:
1. Integrity of the 20 deterministic scenarios (bounds, types, distribution).
2. Independent evaluator mathematics against `treasuregraph/decision_theory.py`.
3. Strict absence of answer leakage in prompts.
4. Robustness of model choice parsing heuristics.
5. End-to-end execution with simulated LLM client.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from experiments.run_experiment_08 import (
    DEFAULT_SCENARIOS_PATH,
    build_scenario_prompt,
    evaluate_decision,
    parse_model_choice,
    run_experiment_08,
)
from treasuregraph.decision_theory import (
    expected_cost_open_first,
    expected_cost_report_first,
)


class TestExperiment08Scenarios(unittest.TestCase):
    """Validates the 20 deterministic scenarios."""

    def setUp(self) -> None:
        self.assertTrue(DEFAULT_SCENARIOS_PATH.exists(), "Scenarios file must exist.")
        self.scenarios = json.loads(DEFAULT_SCENARIOS_PATH.read_text(encoding="utf-8"))

    def test_scenario_count_and_keys(self) -> None:
        """Must contain exactly 20 scenarios with all required fields."""
        self.assertEqual(len(self.scenarios), 20, "Must contain exactly 20 scenarios.")
        seen_ids = set()
        for sc in self.scenarios:
            self.assertIn("scenario_id", sc)
            self.assertIn("probability", sc)
            self.assertIn("open_cost", sc)
            self.assertIn("report_cost", sc)
            self.assertIn("name", sc)
            self.assertIn("description", sc)

            sid = sc["scenario_id"]
            self.assertNotIn(sid, seen_ids, f"Duplicate scenario_id {sid}")
            seen_ids.add(sid)

            # Validate parameter ranges
            p = sc["probability"]
            c_o = sc["open_cost"]
            c_r = sc["report_cost"]

            self.assertIsInstance(p, (int, float))
            self.assertIsInstance(c_o, (int, float))
            self.assertIsInstance(c_r, (int, float))

            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)
            self.assertGreater(c_o, 0.0)
            self.assertGreater(c_r, 0.0)

        self.assertEqual(seen_ids, set(range(1, 21)))

    def test_scenario_distribution_balance(self) -> None:
        """Must have a balanced distribution: 10 A-optimal and 10 B-optimal."""
        a_count = 0
        b_count = 0

        for sc in self.scenarios:
            eval_res = evaluate_decision(sc, model_choice=None)
            opt = eval_res["optimal_choice"]
            if opt == "A":
                a_count += 1
            elif opt == "B":
                b_count += 1

        self.assertEqual(a_count, 10, f"Expected 10 A-optimal scenarios, got {a_count}")
        self.assertEqual(b_count, 10, f"Expected 10 B-optimal scenarios, got {b_count}")


class TestIndependentEvaluator(unittest.TestCase):
    """Validates the independent mathematical evaluator against decision_theory.py."""

    def test_canonical_known_values(self) -> None:
        # Case 1: p=0.9, c_o=1, c_r=10 -> EC(A)=2.1, EC(B)=11.0, Opt=A
        sc1 = {"probability": 0.9, "open_cost": 1.0, "report_cost": 10.0}
        res_a = evaluate_decision(sc1, model_choice="A")
        self.assertEqual(res_a["expected_cost_a"], 2.1)
        self.assertEqual(res_a["expected_cost_b"], 11.0)
        self.assertEqual(res_a["optimal_choice"], "A")
        self.assertEqual(res_a["optimal_expected_cost"], 2.1)
        self.assertEqual(res_a["choice_expected_cost"], 2.1)
        self.assertEqual(res_a["regret"], 0.0)
        self.assertTrue(res_a["is_optimal"])

        res_b = evaluate_decision(sc1, model_choice="B")
        self.assertEqual(res_b["choice_expected_cost"], 11.0)
        self.assertAlmostEqual(res_b["regret"], 8.9, places=6)
        self.assertFalse(res_b["is_optimal"])

        # Case 2: p=0.5, c_o=10, c_r=2 -> EC(A)=16.0, EC(B)=12.0, Opt=B
        sc2 = {"probability": 0.5, "open_cost": 10.0, "report_cost": 2.0}
        res2_b = evaluate_decision(sc2, model_choice="B")
        self.assertEqual(res2_b["expected_cost_a"], 16.0)
        self.assertEqual(res2_b["expected_cost_b"], 12.0)
        self.assertEqual(res2_b["optimal_choice"], "B")
        self.assertEqual(res2_b["optimal_expected_cost"], 12.0)
        self.assertEqual(res2_b["choice_expected_cost"], 12.0)
        self.assertEqual(res2_b["regret"], 0.0)
        self.assertTrue(res2_b["is_optimal"])

        res2_a = evaluate_decision(sc2, model_choice="A")
        self.assertEqual(res2_a["choice_expected_cost"], 16.0)
        self.assertAlmostEqual(res2_a["regret"], 4.0, places=6)
        self.assertFalse(res2_a["is_optimal"])

    def test_unparseable_choice_handling(self) -> None:
        sc = {"probability": 0.7, "open_cost": 5.0, "report_cost": 10.0}
        res = evaluate_decision(sc, model_choice=None)
        self.assertIsNone(res["model_choice"])
        self.assertIsNone(res["choice_expected_cost"])
        self.assertIsNone(res["regret"])
        self.assertFalse(res["is_optimal"])


class TestPromptAntiLeakage(unittest.TestCase):
    """Ensures prompts present all parameters without leaking solutions or evaluations."""

    def test_prompts_do_not_leak_solution(self) -> None:
        scenarios = json.loads(DEFAULT_SCENARIOS_PATH.read_text(encoding="utf-8"))

        for sc in scenarios:
            prompt = build_scenario_prompt(sc)
            eval_res = evaluate_decision(sc, model_choice=None)

            ec_a_str = f"{eval_res['expected_cost_a']:.2f}"
            ec_b_str = f"{eval_res['expected_cost_b']:.2f}"
            opt = eval_res["optimal_choice"]

            # Must contain probability and costs
            self.assertIn(f"{sc['open_cost']:.2f}", prompt)
            self.assertIn(f"{sc['report_cost']:.2f}", prompt)
            self.assertIn("Opção A", prompt)
            self.assertIn("Opção B", prompt)
            self.assertIn("DECISÃO:", prompt)

            # Must NOT state the calculated expected cost values
            self.assertNotIn(f"custo esperado é {ec_a_str}", prompt.lower())
            self.assertNotIn(f"custo esperado é {ec_b_str}", prompt.lower())
            self.assertNotIn(f"resposta correta é {opt}", prompt.lower())
            self.assertNotIn(f"estratégia ótima é {opt}", prompt.lower())
            self.assertNotIn("limiar de indiferença", prompt.lower())


class TestParseModelChoice(unittest.TestCase):
    """Tests the parsing heuristics for model responses."""

    def test_standard_formats(self) -> None:
        self.assertEqual(parse_model_choice("Cálculo realizado.\nDECISÃO: A"), "A")
        self.assertEqual(parse_model_choice("Portanto, DECISÃO: B"), "B")
        self.assertEqual(parse_model_choice("DECISAO: **A**"), "A")
        self.assertEqual(parse_model_choice("DECISION: 'B'"), "B")
        self.assertEqual(parse_model_choice("Opção escolhida: DECISÃO: A."), "A")

    def test_alternative_and_markdown_formats(self) -> None:
        self.assertEqual(parse_model_choice("Concluímos pela opção A.\nDECISÃO: A\n"), "A")
        self.assertEqual(
            parse_model_choice(
                "Analisando os custos, o valor esperado de B é menor.\n"
                "Portanto, a recomendação é a Opção B."
            ),
            "B",
        )
        self.assertEqual(parse_model_choice("ESCOLHA: A"), "A")
        self.assertEqual(parse_model_choice("OPÇÃO: B"), "B")

    def test_ambiguous_and_none(self) -> None:
        self.assertIsNone(parse_model_choice(None))
        self.assertIsNone(parse_model_choice(""))
        self.assertIsNone(parse_model_choice("Não tenho certeza de qual escolher."))


class TestExperiment08SimulatedRun(unittest.TestCase):
    """Tests end-to-end experiment execution with a simulated LLM."""

    def test_simulated_run_all_optimal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_file = tmp_path / "test_results.json"
            report_file = tmp_path / "test_report.json"

            def mock_optimal_llm(prompt: str) -> dict:
                # Mock LLM that perfectly solves every scenario
                p_match = re.search(r"p = (\d+\.\d+)", prompt)
                co_match = re.search(r"C_open = (\d+\.\d+)", prompt)
                cr_match = re.search(r"C_report = (\d+\.\d+)", prompt)

                p = float(p_match.group(1))
                co = float(co_match.group(1))
                cr = float(cr_match.group(1))

                eca = expected_cost_open_first(p, co, cr)
                ecb = expected_cost_report_first(co, cr)
                choice = "A" if eca < ecb else "B"

                return {
                    "content": f"Raciocínio analítico detalhado.\nDECISÃO: {choice}",
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 45,
                        "reasoning_tokens": 80,
                        "total_tokens": 245,
                    },
                    "estimated_cost_usd": 0.0001,
                    "latency_seconds": 0.05,
                }

            report = run_experiment_08(
                scenarios_path=DEFAULT_SCENARIOS_PATH,
                results_path=results_file,
                report_path=report_file,
                llm_callable=mock_optimal_llm,
            )

            self.assertTrue(results_file.exists())
            self.assertTrue(report_file.exists())

            summary = report["summary"]
            self.assertEqual(summary["total_scenarios"], 20)
            self.assertEqual(summary["optimal_choices_count"], 20)
            self.assertEqual(summary["suboptimal_choices_count"], 0)
            self.assertEqual(summary["optimal_rate_pct"], 100.0)
            self.assertEqual(summary["mean_regret"], 0.0)
            self.assertEqual(summary["max_regret"], 0.0)
            self.assertGreater(summary["total_estimated_cost_usd"], 0.0)

    def test_pricing_config_calculation_parameters(self) -> None:
        """Verifies calculate_cost uses correct keyword arguments."""
        from treasuregraph.instrumentation import PricingConfig
        pricing = PricingConfig(
            input_cost_per_million=0.075,
            output_cost_per_million=0.30,
            reasoning_cost_per_million=0.30,
        )
        cost = pricing.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            reasoning_tokens=200,
            reasoning_included_in_output=False,
        )
        self.assertIsNotNone(cost)
        self.assertGreater(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
