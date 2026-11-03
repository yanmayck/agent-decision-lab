"""Unit tests for TreasureGraph resource instrumentation and cost estimation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from treasuregraph.agent import BaselineAgent
from treasuregraph.generator import generate_world
from treasuregraph.instrumentation import (
    PricingConfig,
    ResourceTracker,
    ToolExecutionRecord,
    estimate_tokens,
    parse_api_usage,
)


class TestInstrumentation(unittest.TestCase):
    """Tests resource tracking, token heuristics, API usage parsing, and cost estimation."""

    def test_estimate_tokens_heuristic(self) -> None:
        """Verifies ~4 bytes per token heuristic, empty handling, and estimate labeling."""
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens(None), 0)  # type: ignore

        # 4 bytes ASCII
        self.assertEqual(estimate_tokens("abcd"), 1)
        # 8 bytes ASCII
        self.assertEqual(estimate_tokens("abcdefgh"), 2)
        # 10 bytes ASCII -> ceil(10/4) = 3
        self.assertEqual(estimate_tokens("1234567890"), 3)

        # UTF-8 multibyte characters (e.g. accented PT-BR characters)
        text_pt = "Operação de segurança"  # ~23 bytes UTF-8
        bytes_len = len(text_pt.encode("utf-8"))
        self.assertEqual(estimate_tokens(text_pt), -(-bytes_len // 4))

    def test_parse_api_usage_openai_format(self) -> None:
        """Verifies OpenAI reasoning semantics where completion_tokens includes reasoning."""
        usage_openai = {
            "prompt_tokens": 150,
            "completion_tokens": 60,
            "total_tokens": 210,
            "completion_tokens_details": {
                "reasoning_tokens": 40,
            },
        }
        parsed = parse_api_usage(usage_openai)
        self.assertEqual(parsed["prompt_tokens"], 150)
        self.assertEqual(parsed["completion_tokens"], 60)
        self.assertEqual(parsed["reasoning_tokens"], 40)
        # total == prompt + completion, so reasoning is already included in completion
        self.assertTrue(parsed["reasoning_included_in_output"])
        self.assertEqual(parsed["total_tokens"], 210)

    def test_parse_api_usage_gemini_format(self) -> None:
        """Verifies Gemini semantics where completion_tokens excludes reasoning (thinking)."""
        # Actual Gemini payload structure observed in experiments:
        # prompt: 8621, completion: 207, total: 9437 (thinking: 609)
        usage_gemini = {
            "prompt_tokens": 8621,
            "completion_tokens": 207,
            "total_tokens": 9437,
        }
        parsed = parse_api_usage(usage_gemini)
        self.assertEqual(parsed["prompt_tokens"], 8621)
        self.assertEqual(parsed["completion_tokens"], 207)
        self.assertEqual(parsed["reasoning_tokens"], 609)
        # total > prompt + completion, so reasoning was reported separately
        self.assertFalse(parsed["reasoning_included_in_output"])
        self.assertEqual(parsed["total_tokens"], 9437)

    def test_parse_api_usage_native_gemini_with_explicit_thoughts(self) -> None:
        """Verifies native Gemini usage fields like thoughtsTokenCount."""
        usage_native = {
            "promptTokenCount": 500,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 120,
            "totalTokenCount": 670,
        }
        parsed = parse_api_usage(usage_native)
        self.assertEqual(parsed["prompt_tokens"], 500)
        self.assertEqual(parsed["completion_tokens"], 50)
        self.assertEqual(parsed["reasoning_tokens"], 120)
        self.assertFalse(parsed["reasoning_included_in_output"])
        self.assertEqual(parsed["total_tokens"], 670)

    def test_parse_api_usage_standard_non_reasoning(self) -> None:
        """Verifies standard non-reasoning usage returns None for reasoning_tokens."""
        usage_standard = {
            "prompt_tokens": 300,
            "completion_tokens": 50,
            "total_tokens": 350,
        }
        parsed = parse_api_usage(usage_standard)
        self.assertEqual(parsed["prompt_tokens"], 300)
        self.assertEqual(parsed["completion_tokens"], 50)
        self.assertIsNone(parsed["reasoning_tokens"])
        self.assertTrue(parsed["reasoning_included_in_output"])
        self.assertEqual(parsed["total_tokens"], 350)

    def test_pricing_config_unconfigured_returns_null(self) -> None:
        """Verifies that unknown or unconfigured prices return None (null), never invented values."""
        # Unconfigured
        p_empty = PricingConfig()
        self.assertIsNone(p_empty.calculate_cost(input_tokens=1000, output_tokens=500))

        # Only input configured
        p_partial = PricingConfig(input_cost_per_million=1.5)
        self.assertIsNone(p_partial.calculate_cost(input_tokens=1000, output_tokens=500))

    def test_pricing_config_standard_rates(self) -> None:
        """Verifies standard input and output cost calculation per million tokens."""
        # $1.00 per 1M input, $2.00 per 1M output
        pricing = PricingConfig(input_cost_per_million=1.0, output_cost_per_million=2.0)

        # 1,000,000 input ($1.00) + 500,000 output ($1.00) = $2.00
        cost = pricing.calculate_cost(input_tokens=1_000_000, output_tokens=500_000)
        self.assertEqual(cost, 2.0)

    def test_pricing_config_openai_reasoning_no_double_counting(self) -> None:
        """Verifies reasoning tokens inside output_tokens are not billed twice."""
        # In OpenAI, 100k completion tokens already includes 40k reasoning tokens
        pricing = PricingConfig(
            input_cost_per_million=2.50,
            output_cost_per_million=10.00,
            reasoning_included_in_output=True,
        )

        cost = pricing.calculate_cost(
            input_tokens=1_000_000,  # $2.50
            output_tokens=100_000,   # $1.00 (contains 40k reasoning tokens)
            reasoning_tokens=40_000,
            reasoning_included_in_output=True,
        )
        self.assertEqual(cost, 3.50)

    def test_pricing_config_gemini_reasoning_separate_billed(self) -> None:
        """Verifies separate reasoning tokens are billed without omission or double counting."""
        # In Gemini, completion_tokens (20k) and reasoning_tokens (80k) are separate
        pricing = PricingConfig(
            input_cost_per_million=0.50,
            output_cost_per_million=1.50,
            reasoning_included_in_output=False,
        )

        cost = pricing.calculate_cost(
            input_tokens=1_000_000,  # $0.50
            output_tokens=20_000,    # $0.03
            reasoning_tokens=80_000,  # $0.12 (billed at output rate 1.50/M)
            reasoning_included_in_output=False,
        )
        # 0.50 + 0.03 + 0.12 = 0.65
        self.assertEqual(cost, 0.65)

    def test_pricing_config_differential_reasoning_rate(self) -> None:
        """Verifies differential pricing when reasoning has a different rate than regular output."""
        pricing = PricingConfig(
            input_cost_per_million=1.00,
            output_cost_per_million=5.00,
            reasoning_cost_per_million=15.00,  # Reasoning costs $10/M extra
            reasoning_included_in_output=True,
        )

        # 1M input ($1.00) + 100k output ($0.50) + 50k reasoning differential ($0.50) = $2.00
        cost = pricing.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=100_000,
            reasoning_tokens=50_000,
            reasoning_included_in_output=True,
        )
        self.assertEqual(cost, 2.00)

    def test_resource_tracker_metrics_and_tool_bytes(self) -> None:
        """Verifies ResourceTracker records all 8 metrics and exact tool result bytes."""
        pricing = PricingConfig(
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        )
        tracker = ResourceTracker(pricing=pricing)
        tracker.start()

        # Record 2 API calls (one standard, one with reasoning)
        tracker.record_api_call({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        })
        tracker.record_api_call({
            "prompt_tokens": 150,
            "completion_tokens": 30,
            "total_tokens": 220,  # 40 reasoning tokens separate
        })

        # Record 2 tool executions
        text_1 = "file1.txt (file)\nfile2.txt (file)"
        rec1 = tracker.record_tool_execution(
            step=1,
            action="list_directory",
            arguments={"path": "."},
            result_text=text_1,
            duration_seconds=0.005,
        )
        self.assertEqual(rec1.result_bytes, len(text_1.encode("utf-8")))
        self.assertGreater(rec1.estimated_tool_result_tokens, 0)

        text_2 = "SECRET_CODE = 'TREASURE-1234'"
        rec2 = tracker.record_tool_execution(
            step=2,
            action="read_file",
            arguments={"path": "file1.txt"},
            result_text=text_2,
            duration_seconds=0.008,
            cost_units=1,
            is_parallel=False,
        )
        self.assertEqual(rec2.result_bytes, len(text_2.encode("utf-8")))

        tracker.stop()
        metrics = tracker.get_metrics_summary()

        # Verify all 8 metrics are explicitly present
        self.assertIn("input_tokens", metrics)
        self.assertIn("output_tokens", metrics)
        self.assertIn("reasoning_tokens", metrics)
        self.assertIn("tool_result_tokens", metrics)
        self.assertIn("tool_calls", metrics)
        self.assertIn("api_calls", metrics)
        self.assertIn("estimated_cost_usd", metrics)
        self.assertIn("wall_clock_seconds", metrics)

        # Value checks
        self.assertEqual(metrics["input_tokens"], 250)
        self.assertEqual(metrics["output_tokens"], 50)
        self.assertEqual(metrics["reasoning_tokens"], 40)
        self.assertEqual(metrics["api_calls"], 2)
        self.assertEqual(metrics["tool_calls"], 2)
        self.assertTrue(metrics["tool_result_tokens_is_estimate"])
        self.assertGreater(metrics["tool_result_tokens"], 0)
        self.assertGreaterEqual(metrics["wall_clock_seconds"], 0.0)
        self.assertIsNotNone(metrics["estimated_cost_usd"])

        # Check tool execution byte sizes
        self.assertEqual(len(metrics["tool_executions"]), 2)
        self.assertEqual(metrics["tool_executions"][0]["result_bytes"], len(text_1.encode("utf-8")))
        self.assertEqual(metrics["tool_executions"][1]["result_bytes"], len(text_2.encode("utf-8")))

    def test_baseline_agent_integration_with_resource_metrics(self) -> None:
        """Verifies BaselineAgent collects decoupled resource metrics without real LLM calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            world_info = generate_world(
                output_dir=tmp_path / "world",
                seed=42,
                language="pt-BR",
            )
            correct_code = world_info["secret_code"]

            call_count = 0

            def mock_llm_client(messages: list[dict], tools_schema: list[dict]) -> dict:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "list_directory",
                                                "arguments": json.dumps({"path": "."}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 15,
                            "total_tokens": 135,
                        },
                    }
                else:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_2",
                                            "type": "function",
                                            "function": {
                                                "name": "submit_answer",
                                                "arguments": json.dumps({"code": correct_code}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 200,
                            "completion_tokens": 25,
                            "total_tokens": 225,
                        },
                    }

            pricing = PricingConfig(
                input_cost_per_million=1.00,
                output_cost_per_million=2.00,
            )

            agent = BaselineAgent(
                world_dir=tmp_path / "world",
                model="mock-model",
                api_key="mock-key",
                llm_client=mock_llm_client,
                pricing_config=pricing,
            )

            result = agent.run()

            # Verify completion
            self.assertEqual(result["status"], "submitted")
            self.assertEqual(result["submitted_code"], correct_code)

            # Verify all 8 resource metrics at top level
            self.assertEqual(result["input_tokens"], 320)
            self.assertEqual(result["output_tokens"], 40)
            self.assertIsNone(result["reasoning_tokens"])
            self.assertGreater(result["tool_result_tokens"], 0)
            self.assertEqual(result["tool_calls"], 2)
            self.assertEqual(result["api_calls"], 2)
            self.assertGreater(result["wall_clock_seconds"], 0.0)
            self.assertIsNotNone(result["estimated_cost_usd"])

            # Verify resource_metrics block
            res_meta = result["resource_metrics"]
            self.assertEqual(res_meta["input_tokens"], 320)
            self.assertEqual(res_meta["output_tokens"], 40)
            self.assertEqual(res_meta["tool_calls"], 2)

            # Verify each action in trajectory records result_bytes
            traj = result["trajectory"]
            self.assertEqual(len(traj), 2)
            self.assertIn("result_bytes", traj[0])
            self.assertGreater(traj[0]["result_bytes"], 0)
            self.assertIn("result_bytes", traj[1])


if __name__ == "__main__":
    unittest.main()
