"""Controlled Experiment: Investigating Late Stopping (Parada Tardia).

Compares:
Condition A (Control): Original system prompt.
Condition B (Intervention): Original prompt + explicit early termination directive
when evidence of validity is sufficient.

Evaluated on: world_pt_001 with gemini-3.5-flash-lite and reasoning_effort='high'.
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


def analyze_trajectory(
    result: dict[str, Any],
    correct_code: str,
    terminal_file: str = "cofre/acesso_terminal.txt",
    solution_path: list[str] | None = None,
) -> dict[str, Any]:
    solution_path = solution_path or []
    traj = result.get("trajectory", [])
    submitted = result.get("submitted_code")
    is_correct = submitted == correct_code if submitted else False

    # Find step where code was revealed
    code_revealed_step = None
    for step in traj:
        if step.get("action") == "read_file" and step.get("arguments", {}).get("path") == terminal_file:
            code_revealed_step = step.get("step")
            break

    # Actions taken after the code was encountered (excluding the terminal read itself)
    actions_after_code = []
    if code_revealed_step is not None:
        for step in traj:
            if step.get("step") > code_revealed_step:
                actions_after_code.append(step)

    # Number of redundant exploratory actions after finding code (excluding submit_answer)
    exploratory_actions_after_code = [
        s for s in actions_after_code if s.get("action") != "submit_answer"
    ]

    files_consulted = [
        s.get("arguments", {}).get("path")
        for s in traj
        if s.get("action") == "read_file" and "path" in s.get("arguments", {})
    ]
    unnecessary_files = [f for f in files_consulted if f not in solution_path]

    actions_sequence = [
        f"{s['step']}. {s['action']}({s['arguments']})" for s in traj
    ]

    return {
        "is_correct": is_correct,
        "submitted_code": submitted,
        "total_actions": len(traj),
        "code_revealed_at_step": code_revealed_step,
        "actions_after_finding_code": len(actions_after_code),
        "exploratory_actions_after_finding_code": len(exploratory_actions_after_code),
        "post_code_action_details": [
            f"{s['step']}. {s['action']}({s['arguments']})" for s in actions_after_code
        ],
        "files_consulted": files_consulted,
        "unnecessary_files_consulted": unnecessary_files,
        "actions_sequence": actions_sequence,
        "tokens": result.get("total_tokens", {}),
        "wall_clock_duration_seconds": result.get("total_elapsed_seconds"),
        "tool_duration_seconds": result.get("total_duration_seconds"),
    }


def main():
    load_dotenv()

    world_dir = Path("experiments/world_pt_001")
    answer_key_path = Path("experiments/answer_key.json")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    cond_a_file = results_dir / "late_stopping_cond_a_control.json"
    cond_b_file = results_dir / "late_stopping_cond_b_intervention.json"
    report_file = results_dir / "late_stopping_comparison_report.json"

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

    intervention_directive = (
        "\nQuando encontrar o código secreto e tiver evidência suficiente de que ele é válido, "
        "submeta imediatamente a resposta, sem continuar explorando."
    )
    prompt_intervention = DEFAULT_SYSTEM_PROMPT + intervention_directive

    print("=" * 70)
    print("EXPERIMENTO CONTROLADO: INVESTIGAÇÃO DE PARADA TARDIA (LATE STOPPING)")
    print("=" * 70)
    print(f"Ambiente: {world_dir}")
    print(f"Modelo: {model_name}")
    print(f"Raciocínio: HIGH (reasoning_effort='high')")
    print(f"Limite de ações: 15")

    # -------------------------------------------------------------------------
    # CONDIÇÃO A: CONTROLE (Prompt original)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXECUTANDO CONDIÇÃO A — CONTROLE (Prompt Original)")
    print("-" * 70)

    agent_a = BaselineAgent(
        world_dir=world_dir,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_steps=15,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        reasoning_effort="high",
    )

    t0_a = time.perf_counter()
    result_a = agent_a.run()
    elapsed_a = time.perf_counter() - t0_a
    result_a["total_elapsed_seconds"] = round(elapsed_a, 4)

    cond_a_file.write_text(json.dumps(result_a, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Condição A concluída em {elapsed_a:.2f}s. Trajetória salva em {cond_a_file}")

    analysis_a = analyze_trajectory(result_a, correct_code, "cofre/acesso_terminal.txt", solution_path)

    # -------------------------------------------------------------------------
    # CONDIÇÃO B: INTERVENÇÃO (Prompt com diretiva de parada imediata)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXECUTANDO CONDIÇÃO B — INTERVENÇÃO (Com Diretiva de Parada Imediata)")
    print("-" * 70)

    agent_b = BaselineAgent(
        world_dir=world_dir,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_steps=15,
        system_prompt=prompt_intervention,
        reasoning_effort="high",
    )

    t0_b = time.perf_counter()
    result_b = agent_b.run()
    elapsed_b = time.perf_counter() - t0_b
    result_b["total_elapsed_seconds"] = round(elapsed_b, 4)

    cond_b_file.write_text(json.dumps(result_b, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Condição B concluída em {elapsed_b:.2f}s. Trajetória salva em {cond_b_file}")

    analysis_b = analyze_trajectory(result_b, correct_code, "cofre/acesso_terminal.txt", solution_path)

    # -------------------------------------------------------------------------
    # RELATÓRIO COMPARATIVO
    # -------------------------------------------------------------------------
    comparison = {
        "metadata": {
            "model": model_name,
            "reasoning_effort": "high",
            "environment": str(world_dir),
            "seed": 42,
            "intervention_text": intervention_directive.strip(),
        },
        "condition_a_control": analysis_a,
        "condition_b_intervention": analysis_b,
        "delta": {
            "actions_delta (B - A)": analysis_b["total_actions"] - analysis_a["total_actions"],
            "post_code_exploratory_actions_delta (B - A)": (
                analysis_b["exploratory_actions_after_finding_code"]
                - analysis_a["exploratory_actions_after_finding_code"]
            ),
            "tokens_delta (B - A)": (
                analysis_b["tokens"].get("total_tokens", 0) - analysis_a["tokens"].get("total_tokens", 0)
            ),
            "wall_clock_seconds_delta (B - A)": round(
                analysis_b["wall_clock_duration_seconds"] - analysis_a["wall_clock_duration_seconds"], 4
            ),
        },
    }

    report_file.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório comparativo salvo em {report_file}")

    print("\n" + "=" * 70)
    print("RESULTADOS DO EXPERIMENTO CONTROLADO")
    print("=" * 70)
    print(f"1. Sucesso na tarefa:")
    print(f"   - Controle (A):     {'SIM' if analysis_a['is_correct'] else 'NÃO'} ({analysis_a['submitted_code']})")
    print(f"   - Intervenção (B):  {'SIM' if analysis_b['is_correct'] else 'NÃO'} ({analysis_b['submitted_code']})")
    print(f"\n2. Ponto em que o código foi revelado (leitura de cofre/acesso_terminal.txt):")
    print(f"   - Controle (A):     Passo {analysis_a['code_revealed_at_step']}")
    print(f"   - Intervenção (B):  Passo {analysis_b['code_revealed_at_step']}")
    print(f"\n3. Ações executadas APÓS encontrar o código:")
    print(f"   - Controle (A):     {analysis_a['actions_after_finding_code']} ações totais ({analysis_a['exploratory_actions_after_finding_code']} exploratórias + 1 submissão)")
    print(f"     Detalhe: {analysis_a['post_code_action_details']}")
    print(f"   - Intervenção (B):  {analysis_b['actions_after_finding_code']} ações totais ({analysis_b['exploratory_actions_after_finding_code']} exploratórias + 1 submissão)")
    print(f"     Detalhe: {analysis_b['post_code_action_details']}")
    print(f"\n4. Total de ações e arquivos consultados:")
    print(f"   - Controle (A):     {analysis_a['total_actions']} ações | {len(analysis_a['files_consulted'])} arquivos ({analysis_a['unnecessary_files_consulted']} desnecessários)")
    print(f"   - Intervenção (B):  {analysis_b['total_actions']} ações | {len(analysis_b['files_consulted'])} arquivos ({analysis_b['unnecessary_files_consulted']} desnecessários)")
    print(f"\n5. Tokens consumidos:")
    print(f"   - Controle (A):     {analysis_a['tokens']}")
    print(f"   - Intervenção (B):  {analysis_b['tokens']}")
    print(f"\n6. Tempo total de execução (Wall-clock):")
    print(f"   - Controle (A):     {analysis_a['wall_clock_duration_seconds']}s")
    print(f"   - Intervenção (B):  {analysis_b['wall_clock_duration_seconds']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
