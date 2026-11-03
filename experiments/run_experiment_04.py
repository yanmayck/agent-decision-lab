"""Runner script for Experiment 04: Tool Cost Sensitivity.

Investigates whether presenting an explicit operational cost for read_file
influences agent behavior (e.g., reducing unnecessary reads or switching
from parallel eager batching to sequential clue following).

Three conditions evaluated on world_pt_001 with gemini-3.5-flash-lite (HIGH):
Condition A: No cost informed (baseline)
Condition B: 1 unit per read_file
Condition C: 10 units per read_file
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treasuregraph.agent import DEFAULT_SYSTEM_PROMPT, BaselineAgent, load_dotenv


def run_condition(
    condition_id: str,
    cost: int,
    world_dir: Path,
    model_name: str,
    api_key: str,
    api_base: str | None,
    correct_code: str,
    solution_path: list[str],
) -> dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"EXECUTANDO CONDIÇÃO {condition_id} (Custo de read_file: {cost} unidade(s))")
    print(f"{'='*70}")

    agent = BaselineAgent(
        world_dir=world_dir,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_steps=15,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        reasoning_effort="high",
        read_cost=cost,
    )

    t_start = time.perf_counter()
    agent_result = agent.run()
    wall_duration = time.perf_counter() - t_start
    agent_result["total_wall_clock_seconds"] = round(wall_duration, 4)

    traj = agent_result.get("trajectory", [])
    submitted = agent_result.get("submitted_code")
    is_correct = submitted == correct_code if submitted else False

    read_actions = [s for s in traj if s.get("action") == "read_file"]
    parallel_reads = [s for s in read_actions if s.get("is_parallel")]
    sequential_reads = [s for s in read_actions if not s.get("is_parallel")]

    files_consulted = [s.get("arguments", {}).get("path") for s in read_actions]
    unnecessary_files = [f for f in files_consulted if f not in solution_path]

    actions_sequence = [
        f"{s['step']}. {s['action']}({s['arguments']}) [paralelo={s.get('is_parallel')}, custo={s.get('cost', 0)}]"
        for s in traj
    ]

    # Check if reads were strictly sequential
    is_fully_sequential_reading = len(parallel_reads) == 0 and len(read_actions) > 0

    analysis = {
        "condition_id": condition_id,
        "cost_per_read": cost,
        "is_correct": is_correct,
        "submitted_code": submitted,
        "total_actions": len(traj),
        "api_calls_count": agent_result.get("api_calls_count", 0),
        "reads_requested_count": len(read_actions),
        "parallel_reads_count": len(parallel_reads),
        "sequential_reads_count": len(sequential_reads),
        "is_sequential_clue_following": is_fully_sequential_reading,
        "files_consulted": files_consulted,
        "unnecessary_files_consulted": unnecessary_files,
        "total_tool_cost": agent_result.get("total_tool_cost", 0),
        "tokens": agent_result.get("total_tokens", {}),
        "wall_clock_duration_seconds": agent_result.get("total_wall_clock_seconds"),
        "tool_duration_seconds": agent_result.get("total_duration_seconds"),
        "actions_sequence": actions_sequence,
        "full_result": agent_result,
    }

    print(f"Condição {condition_id} concluída:")
    print(f" - Sucesso: {'SIM' if is_correct else 'NÃO'} ({submitted})")
    print(f" - Chamadas de API: {agent_result.get('api_calls_count')}")
    print(f" - Ações de ferramentas: {len(traj)}")
    print(f" - Leituras solicitadas: {len(read_actions)} (Paralelas: {len(parallel_reads)}, Sequenciais: {len(sequential_reads)})")
    print(f" - Seguiu pistas sequencialmente: {'SIM' if is_fully_sequential_reading else 'NÃO'}")
    print(f" - Custo total das ferramentas: {agent_result.get('total_tool_cost')}")
    print(f" - Tokens totais: {agent_result.get('total_tokens', {}).get('total_tokens')}")
    print(f" - Duração Wall-clock: {agent_result.get('total_wall_clock_seconds')}s")

    return analysis


def main():
    load_dotenv()

    world_dir = Path("experiments/world_pt_001")
    answer_key_path = Path("experiments/answer_key.json")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    file_a = results_dir / "cost_cond_a_nocost.json"
    file_b = results_dir / "cost_cond_b_cost1.json"
    file_c = results_dir / "cost_cond_c_cost10.json"
    report_file = results_dir / "cost_comparison_report.json"

    model_name = os.getenv("TREASURE_MODEL") or "gemini-3.5-flash-lite"
    api_key = (
        os.getenv("TREASURE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    api_base = os.getenv("TREASURE_API_BASE") or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("ERROR: API key not found in environment or .env.")
        sys.exit(1)

    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8")) if answer_key_path.exists() else {}
    correct_code = answer_key.get("secret_code", "")
    solution_path = answer_key.get("solution_path", [])

    print("=" * 70)
    print("INICIANDO EXPERIMENTO 04: SENSIBILIDADE A CUSTO DE FERRAMENTAS")
    print("=" * 70)
    print(f"Ambiente: {world_dir}")
    print(f"Modelo: {model_name}")
    print(f"Nível de raciocínio: HIGH (reasoning_effort='high')")
    print(f"Limite de ações: 15")

    # Condição A: Custo = 0 (Sem custo informado)
    res_a = run_condition("A", 0, world_dir, model_name, api_key, api_base, correct_code, solution_path)
    file_a.write_text(json.dumps(res_a["full_result"], indent=2, ensure_ascii=False), encoding="utf-8")

    # Condição B: Custo = 1
    res_b = run_condition("B", 1, world_dir, model_name, api_key, api_base, correct_code, solution_path)
    file_b.write_text(json.dumps(res_b["full_result"], indent=2, ensure_ascii=False), encoding="utf-8")

    # Condição C: Custo = 10
    res_c = run_condition("C", 10, world_dir, model_name, api_key, api_base, correct_code, solution_path)
    file_c.write_text(json.dumps(res_c["full_result"], indent=2, ensure_ascii=False), encoding="utf-8")

    # Remove full_result from report structure to keep summary clean
    report_data = {
        "metadata": {
            "model": model_name,
            "reasoning_effort": "high",
            "environment": str(world_dir),
            "seed": 42,
            "solution_path": solution_path,
            "correct_code": correct_code,
        },
        "condition_a_nocost": {k: v for k, v in res_a.items() if k != "full_result"},
        "condition_b_cost1": {k: v for k, v in res_b.items() if k != "full_result"},
        "condition_c_cost10": {k: v for k, v in res_c.items() if k != "full_result"},
    }

    report_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório comparativo completo salvo em {report_file}")


if __name__ == "__main__":
    main()
