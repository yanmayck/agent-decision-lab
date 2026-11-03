"""Runner script for TreasureGraph Scalability & Resource Consumption Experiments.

Evaluates how agent resource consumption scales across problem sizes:
- input_tokens (prompt tokens reported by API)
- output_tokens (completion tokens reported by API)
- reasoning_tokens (thinking tokens reported by API, null if unsupported)
- tool_result_tokens (heuristic estimate of tool output tokens, kept separate from billable)
- tool_calls (number of tool actions)
- api_calls (number of LLM requests)
- estimated_cost_usd (based on configurable pricing per million tokens; null when unknown)
- wall_clock_seconds (total elapsed wall-clock duration)
- result_bytes for each individual tool call

Usage:
    # Dry-run validation (does not call live LLM API):
    python experiments/run_scalability_experiment.py --dry-run

    # Live experiment (when authorized):
    python experiments/run_scalability_experiment.py --model gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treasuregraph.agent import DEFAULT_SYSTEM_PROMPT, BaselineAgent, load_dotenv
from treasuregraph.generator import generate_world
from treasuregraph.instrumentation import PricingConfig, ResourceTracker


def create_pricing_config_from_env() -> PricingConfig:
    """Reads token pricing from environment variables or returns unconfigured (null cost).

    Supported environment variables:
        TREASURE_INPUT_COST_PER_M: USD price per 1M input tokens.
        TREASURE_OUTPUT_COST_PER_M: USD price per 1M output tokens.
        TREASURE_REASONING_COST_PER_M: USD price per 1M reasoning tokens.
    """
    def _parse_float(name: str) -> float | None:
        val = os.getenv(name)
        if val is None or not val.strip():
            return None
        try:
            return float(val.strip())
        except ValueError:
            return None

    return PricingConfig(
        input_cost_per_million=_parse_float("TREASURE_INPUT_COST_PER_M"),
        output_cost_per_million=_parse_float("TREASURE_OUTPUT_COST_PER_M"),
        reasoning_cost_per_million=_parse_float("TREASURE_REASONING_COST_PER_M"),
    )


def run_scalability_dry_run(world_dir: Path, pricing: PricingConfig) -> dict[str, Any]:
    """Validates the scalability instrumentation without contacting external LLM APIs."""
    print("\n--- Modo Dry-Run de Escalabilidade (Sem chamadas reais ao LLM) ---")

    # Generate test world if missing
    if not world_dir.exists():
        generate_world(output_dir=world_dir, seed=42, language="pt-BR")

    answer_key_path = Path("experiments/answer_key.json")
    correct_code = "SIMULATED-KEY-999"
    if answer_key_path.exists():
        try:
            correct_code = json.loads(answer_key_path.read_text(encoding="utf-8")).get("secret_code", correct_code)
        except Exception:
            pass

    turn = 0

    def simulated_client(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal turn
        turn += 1
        if turn == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "dry_call_1",
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
                    "prompt_tokens": 250,
                    "completion_tokens": 20,
                    "total_tokens": 270,
                },
            }
        else:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "dry_call_2",
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
                    "prompt_tokens": 350,
                    "completion_tokens": 30,
                    "total_tokens": 380,
                },
            }

    agent = BaselineAgent(
        world_dir=world_dir,
        model="simulated-scalability-model",
        llm_client=simulated_client,
        pricing_config=pricing,
    )

    result = agent.run()
    return result


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="TreasureGraph Scalability Experiment Runner")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run with simulated responses (no API calls)")
    parser.add_argument("--live", action="store_true", help="Execute with live model API (requires authorization)")
    parser.add_argument("--world-dir", type=str, default="experiments/world_pt_001", help="Target environment directory")
    parser.add_argument("--input-price", type=float, default=None, help="Price USD per 1M input tokens")
    parser.add_argument("--output-price", type=float, default=None, help="Price USD per 1M output tokens")
    parser.add_argument("--reasoning-price", type=float, default=None, help="Price USD per 1M reasoning tokens")
    args = parser.parse_args()

    # Determine pricing config
    env_pricing = create_pricing_config_from_env()
    pricing = PricingConfig(
        input_cost_per_million=args.input_price if args.input_price is not None else env_pricing.input_cost_per_million,
        output_cost_per_million=args.output_price if args.output_price is not None else env_pricing.output_cost_per_million,
        reasoning_cost_per_million=args.reasoning_price if args.reasoning_price is not None else env_pricing.reasoning_cost_per_million,
    )

    world_dir = Path(args.world_dir)

    if not args.live or args.dry_run:
        result = run_scalability_dry_run(world_dir, pricing)
    else:
        # Live execution (reserved until authorized)
        agent = BaselineAgent(
            world_dir=world_dir,
            reasoning_effort="high",
            pricing_config=pricing,
        )
        result = agent.run()

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "scalability_dry_run_report.json"

    report_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados gravados em: {report_file}")
    print("\nResumo da Instrumentação de Recursos:")
    print(f" - input_tokens: {result.get('input_tokens')}")
    print(f" - output_tokens: {result.get('output_tokens')}")
    print(f" - reasoning_tokens: {result.get('reasoning_tokens')}")
    print(f" - tool_result_tokens (estimativa): {result.get('tool_result_tokens')}")
    print(f" - tool_calls: {result.get('tool_calls')}")
    print(f" - api_calls: {result.get('api_calls')}")
    print(f" - estimated_cost_usd: {result.get('estimated_cost_usd')}")
    print(f" - wall_clock_seconds: {result.get('wall_clock_seconds')} s")


if __name__ == "__main__":
    main()
