"""Runner script for Experiment 02: Reasoning Effort comparison (HIGH).

Evaluates gemini-3.5-flash-lite on world_pt_001 with reasoning_effort='high'.
Compares metrics and trajectory directly against Experiment 01 (baseline).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treasuregraph.agent import BaselineAgent, load_dotenv


def main():
    load_dotenv()

    world_dir = Path("experiments/world_pt_001")
    answer_key_path = Path("experiments/answer_key.json")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    exp1_file = results_dir / "baseline_seed42.json"
    exp2_output_file = results_dir / "baseline_seed42_high.json"
    exp2_report_file = results_dir / "baseline_seed42_high_report.json"

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

    print("=" * 60)
    print("INICIANDO EXPERIMENTO 02 (TREASUREGRAPH - REASONING EFFORT: HIGH)")
    print("=" * 60)
    print(f"Ambiente: {world_dir}")
    print(f"Modelo: {model_name}")
    print(f"Nível de raciocínio: HIGH (reasoning_effort='high')")
    print(f"Limite de ações: 15")

    # 1. VERIFICAÇÃO DE SUPORTE OFICIAL ANTES DA EXECUÇÃO
    print("\nVerificando suporte a reasoning_effort='high' junto à API...")
    probe_agent = BaselineAgent(
        world_dir=world_dir,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        reasoning_effort="high",
    )

    probe_payload = [
        {"role": "user", "content": "Teste de compatibilidade de raciocínio. Responda 'OK'."}
    ]
    try:
        probe_resp = probe_agent._call_llm(probe_payload)
        probe_usage = probe_resp.get("usage", {})
        print("Confirmação da API: Configuração 'reasoning_effort=high' ACEITA e ATIVA.")
        print(f"Métricas do teste prévio: {probe_usage}")
    except Exception as e:
        print(f"ERRO: O modelo {model_name} rejeitou reasoning_effort='high': {e}")
        print("Abortando experimento conforme requisitos.")
        sys.exit(1)

    # 2. EXECUÇÃO DO AGENTE
    print("\nExecutando agente com reasoning_effort='high' (uma única vez, sem intervenções)...")
    agent = BaselineAgent(
        world_dir=world_dir,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_steps=15,
        reasoning_effort="high",
    )

    t_wall_start = time.perf_counter()
    agent_result = agent.run()
    wall_duration = time.perf_counter() - t_wall_start
    agent_result["total_wall_clock_seconds"] = round(wall_duration, 4)

    # Salva trajetória completa
    exp2_output_file.write_text(
        json.dumps(agent_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Trajetória salva em {exp2_output_file}")

    # 3. AVALIAÇÃO EXTERNA E COMPARAÇÃO COM EXPERIMENTO 01
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8")) if answer_key_path.exists() else {}
    correct_code = answer_key.get("secret_code")
    solution_path = answer_key.get("solution_path", [])
    submitted_code = agent_result.get("submitted_code")
    is_correct = (submitted_code == correct_code) if submitted_code else False

    trajectory = agent_result.get("trajectory", [])
    actions_sequence = [f"{step['step']}. {step['action']}({step['arguments']})" for step in trajectory]
    files_consulted = [
        step["arguments"].get("path")
        for step in trajectory
        if step["action"] == "read_file" and "path" in step["arguments"]
    ]
    unnecessary_files = [f for f in files_consulted if f not in solution_path]
    errors = [step for step in trajectory if step["status"] == "error"]

    # Carrega dados do Experimento 01 para comparação direta
    exp1_data = {}
    if exp1_file.exists():
        exp1_data = json.loads(exp1_file.read_text(encoding="utf-8"))

    exp1_steps = exp1_data.get("total_steps", 0)
    exp1_traj = exp1_data.get("trajectory", [])
    exp1_files = [
        s["arguments"].get("path")
        for s in exp1_traj
        if s.get("action") == "read_file" and "path" in s.get("arguments", {})
    ]
    exp1_unnecessary = [f for f in exp1_files if f not in solution_path]
    exp1_tokens = exp1_data.get("total_tokens", {})
    exp1_submitted = exp1_data.get("submitted_code")

    comparison_report = {
        "experiment_02_high": {
            "model": model_name,
            "reasoning_effort_applied": "high",
            "is_correct": is_correct,
            "submitted_code": submitted_code,
            "expected_code": correct_code,
            "total_actions": len(trajectory),
            "actions_sequence": actions_sequence,
            "files_consulted": files_consulted,
            "unnecessary_files_consulted": unnecessary_files,
            "tokens": agent_result.get("total_tokens"),
            "tool_duration_seconds": agent_result.get("total_duration_seconds"),
            "wall_clock_duration_seconds": agent_result.get("total_wall_clock_seconds"),
            "errors_count": len(errors),
        },
        "experiment_01_baseline": {
            "model": model_name,
            "reasoning_effort_applied": "default (none)",
            "is_correct": exp1_submitted == correct_code,
            "submitted_code": exp1_submitted,
            "total_actions": exp1_steps,
            "files_consulted": exp1_files,
            "unnecessary_files_consulted": exp1_unnecessary,
            "tokens": exp1_tokens,
        },
        "comparison_analysis": {
            "action_count_delta": len(trajectory) - exp1_steps,
            "unnecessary_files_delta": len(unnecessary_files) - len(exp1_unnecessary),
            "tokens_delta": (agent_result.get("total_tokens", {}).get("total_tokens", 0) - exp1_tokens.get("total_tokens", 0)),
        },
    }

    exp2_report_file.write_text(
        json.dumps(comparison_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Relatório de comparação salvo em {exp2_report_file}")

    print("\n" + "=" * 60)
    print("RELATÓRIO COMPARATIVO: EXPERIMENTO 02 (HIGH) vs EXPERIMENTO 01 (DEFAULT)")
    print("=" * 60)
    print(f"1. Sucesso na tarefa:")
    print(f"   - Exp 01: {'SIM' if exp1_submitted == correct_code else 'NÃO'} ({exp1_submitted})")
    print(f"   - Exp 02: {'SIM' if is_correct else 'NÃO'} ({submitted_code})")
    print(f"\n2. Quantidade e sequência de ações:")
    print(f"   - Exp 01: {exp1_steps} ações")
    print(f"   - Exp 02: {len(trajectory)} ações")
    print("   Sequência Exp 02:")
    for a in actions_sequence:
        print(f"     {a}")
    print(f"\n3. Arquivos consultados ({len(files_consulted)}):")
    print(f"   - Consultados: {files_consulted}")
    print(f"   - Fora do caminho ideal (desnecessários): {unnecessary_files}")
    print(f"\n4. Tokens consumidos:")
    print(f"   - Exp 01: {exp1_tokens}")
    print(f"   - Exp 02: {agent_result.get('total_tokens')}")
    print(f"\n5. Tempo total de execução (incluindo chamadas de API):")
    print(f"   - Exp 02 Wall-clock: {agent_result.get('total_wall_clock_seconds')}s")
    print(f"   - Ferramentas I/O:   {agent_result.get('total_duration_seconds')}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
