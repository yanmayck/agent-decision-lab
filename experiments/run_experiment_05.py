"""Runner script for Experiment 05: Document Graph Maps and Agent Decision Making.

Investigates how the availability of topological document network maps influences
agent decision making, planning, exploration strategies, and tool efficiency.

Three conditions evaluated on world_pt_001 with gemini-3.5-flash-lite (HIGH):
Condition A: Agent without map (baseline, original 3 tools).
Condition B: Agent with complete map (includes get_map tool with 100% genuine edges).
Condition C: Agent with noisy map (includes get_map tool with genuine edges + fake decoy connections).

Validation:
Strict verification is executed before any agent run to ensure that the agent-facing
map contains ZERO leaks of the secret code, terminal flags, or edge validity.
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
from treasuregraph.instrumentation import PricingConfig
from treasuregraph.map_generator import MapMode, generate_document_map


def validate_map_security(map_obj: Any, secret_code: str, condition_name: str) -> None:
    """Strictly validates that agent-facing map contains zero leakage before execution."""
    agent_dict = map_obj.to_agent_dict() if hasattr(map_obj, "to_agent_dict") else map_obj
    agent_json = json.dumps(agent_dict)

    # 1. Must not contain the genuine secret code
    if secret_code in agent_json:
        raise ValueError(
            f"SECURITY BREACH in {condition_name}: Secret code '{secret_code}' was found in the agent map!"
        )

    # 2. Must not contain decoy fake codes
    for decoy_prefix in ("ISCA-", "DECOY-"):
        if decoy_prefix in agent_json:
            raise ValueError(
                f"SECURITY BREACH in {condition_name}: Decoy code prefix '{decoy_prefix}' was found in the agent map!"
            )

    # 3. Must not leak ground truth metadata fields or semantic indicators
    forbidden_terms = [
        "solution_path",
        "is_terminal",
        "has_treasure",
        "is_true_clue",
        "secret_code",
        "role",
    ]
    for term in forbidden_terms:
        if f'"{term}"' in agent_json:
            raise ValueError(
                f"SECURITY BREACH in {condition_name}: Ground-truth metadata field '{term}' found in agent map!"
            )

    for val in ("fake", "falsa", "is_valid", "spurious"):
        if val in agent_json.lower():
            raise ValueError(
                f"SECURITY BREACH in {condition_name}: Falsity indicator '{val}' found in agent map!"
            )

    print(f"[{condition_name}] Seguranca do mapa validada: ZERO vazamento do codigo secreto ou gabarito.")


def run_condition(
    condition_id: str,
    condition_name: str,
    document_map: Any,
    world_dir: Path,
    model_name: str,
    api_key: str,
    api_base: str | None,
    correct_code: str,
    solution_path: list[str],
    pricing: PricingConfig,
) -> dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"EXECUTANDO CONDIÇÃO {condition_id}: {condition_name.upper()}")
    print(f"{'='*75}")

    agent = BaselineAgent(
        world_dir=world_dir,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_steps=15,
        reasoning_effort="high",
        document_map=document_map,
        pricing_config=pricing,
    )

    t_start = time.perf_counter()
    agent_result = agent.run()
    wall_duration = time.perf_counter() - t_start
    agent_result["total_wall_clock_seconds"] = round(wall_duration, 4)

    traj = agent_result.get("trajectory", [])
    submitted = agent_result.get("submitted_code")
    is_correct = (submitted == correct_code) if submitted else False

    # Extract exploration path
    files_read = [
        step["arguments"].get("path")
        for step in traj
        if step["action"] == "read_file" and "path" in step["arguments"]
    ]
    map_calls = [step for step in traj if step["action"] == "get_map"]
    map_consulted_step = map_calls[0]["step"] if map_calls else None

    # Redundant / repeated files
    read_counts: dict[str, int] = {}
    for f in files_read:
        read_counts[f] = read_counts.get(f, 0) + 1
    repeated_reads = {f: c for f, c in read_counts.items() if c > 1}
    unnecessary_reads = [f for f in files_read if f not in solution_path]

    actions_sequence = [f"{step['step']}. {step['action']}({step['arguments']})" for step in traj]

    print(f"\nResultado da Condição {condition_id}:")
    print(f" - Status: {agent_result.get('status')}")
    print(f" - Código submetido: {submitted} ({'CORRETO' if is_correct else 'INCORRETO'})")
    print(f" - Mapa consultado: {len(map_calls) > 0} (passo {map_consulted_step})")
    print(f" - Total de ações: {len(traj)} | Requisições LLM: {agent_result.get('api_calls_count')}")
    print(f" - Arquivos lidos ({len(files_read)}): {files_read}")
    print(f" - Arquivos desnecessários: {unnecessary_reads}")
    print(f" - Tokens totais: {agent_result.get('total_tokens', {}).get('total_tokens')}")
    print(f" - Raciocínio (thinking): {agent_result.get('total_tokens', {}).get('reasoning_tokens')}")
    print(f" - Custo estimado USD: ${agent_result.get('estimated_cost_usd')}")
    print(f" - Duração real: {agent_result.get('total_wall_clock_seconds')} s")

    return {
        "condition_id": condition_id,
        "condition_name": condition_name,
        "evaluation": {
            "is_correct": is_correct,
            "submitted_code": submitted,
            "expected_code": correct_code,
        },
        "map_metrics": {
            "map_available": document_map is not None,
            "map_consulted": len(map_calls) > 0,
            "map_consulted_step": map_consulted_step,
            "map_calls_count": len(map_calls),
        },
        "path_traversal": {
            "files_read_sequence": files_read,
            "total_reads": len(files_read),
            "unnecessary_reads": unnecessary_reads,
            "repeated_reads": repeated_reads,
            "followed_solution_chain": files_read == solution_path,
        },
        "resource_consumption": {
            "api_calls": agent_result.get("api_calls_count"),
            "total_actions": len(traj),
            "input_tokens": agent_result.get("input_tokens"),
            "output_tokens": agent_result.get("output_tokens"),
            "reasoning_tokens": agent_result.get("reasoning_tokens"),
            "tool_result_tokens": agent_result.get("tool_result_tokens"),
            "total_tokens": agent_result.get("total_tokens", {}).get("total_tokens"),
            "estimated_cost_usd": agent_result.get("estimated_cost_usd"),
            "wall_clock_seconds": agent_result.get("total_wall_clock_seconds"),
        },
        "raw_agent_result": agent_result,
        "actions_sequence": actions_sequence,
    }


def main() -> None:
    load_dotenv()

    world_dir = Path("experiments/world_pt_001")
    answer_key_path = Path("experiments/answer_key.json")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if not answer_key_path.exists():
        raise FileNotFoundError(f"Arquivo {answer_key_path} não encontrado.")

    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    correct_code = answer_key["secret_code"]
    solution_path = answer_key["solution_path"]

    model_name = os.getenv("TREASURE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
    api_key = os.getenv("TREASURE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    api_base = os.getenv("TREASURE_API_BASE")

    if not api_key:
        print("Erro: Chave de API não configurada em .env ou variáveis de ambiente.")
        sys.exit(1)

    pricing = PricingConfig(
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
        reasoning_included_in_output=False,
    )

    print("=" * 75)
    print("TREASUREGRAPH - EXPERIMENTO 05: TOPOLOGICAL MAPS & AGENT DECISIONS")
    print("=" * 75)
    print(f"Modelo: {model_name} | Raciocínio: HIGH")
    print(f"Ambiente: {world_dir} | Código esperado: {correct_code}")

    # 1. Gerar os mapas para as Condições B e C
    print("\n--- GERAÇÃO E VALIDAÇÃO PRÉVIA DOS MAPAS ---")
    map_complete = generate_document_map(answer_key, mode=MapMode.COMPLETE, seed=42)
    map_noisy = generate_document_map(answer_key, mode=MapMode.NOISY, seed=42, num_fake_edges=2)

    # 2. VALIDAÇÃO PRÉ-EXECUÇÃO: Validar que os mapas não revelam o código secreto nem o gabarito
    validate_map_security(map_complete, correct_code, "Condição B (Completo)")
    validate_map_security(map_noisy, correct_code, "Condição C (Ruidoso)")

    # 3. Executar Condição A: Sem Mapa
    res_a = run_condition(
        condition_id="A",
        condition_name="Sem Mapa (Baseline)",
        document_map=None,
        world_dir=world_dir,
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        correct_code=correct_code,
        solution_path=solution_path,
        pricing=pricing,
    )

    # Intervalo de cortesia para taxa de API
    time.sleep(2)

    # 4. Executar Condição B: Com Mapa Completo
    res_b = run_condition(
        condition_id="B",
        condition_name="Com Mapa Completo",
        document_map=map_complete,
        world_dir=world_dir,
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        correct_code=correct_code,
        solution_path=solution_path,
        pricing=pricing,
    )

    time.sleep(2)

    # 5. Executar Condição C: Com Mapa Contendo Conexões Falsas
    res_c = run_condition(
        condition_id="C",
        condition_name="Com Mapa Ruidoso (Conexões Falsas)",
        document_map=map_noisy,
        world_dir=world_dir,
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        correct_code=correct_code,
        solution_path=solution_path,
        pricing=pricing,
    )

    # Salvar as trajetórias individuais
    file_a = results_dir / "experiment_05_condition_a.json"
    file_b = results_dir / "experiment_05_condition_b.json"
    file_c = results_dir / "experiment_05_condition_c.json"

    file_a.write_text(json.dumps(res_a, indent=2, ensure_ascii=False), encoding="utf-8")
    file_b.write_text(json.dumps(res_b, indent=2, ensure_ascii=False), encoding="utf-8")
    file_c.write_text(json.dumps(res_c, indent=2, ensure_ascii=False), encoding="utf-8")

    # Relatório Comparativo Consolidado
    report = {
        "experiment": "TreasureGraph - Experimento 05: Efeito de Mapas Topológicos na Tomada de Decisão",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name,
        "reasoning_effort": "high",
        "environment": str(world_dir),
        "secret_code": correct_code,
        "comparative_summary": {
            "A_sem_mapa": {
                "sucesso": res_a["evaluation"]["is_correct"],
                "total_acoes": res_a["resource_consumption"]["total_actions"],
                "chamadas_api": res_a["resource_consumption"]["api_calls"],
                "arquivos_lidos": res_a["path_traversal"]["total_reads"],
                "leituras_desnecessarias": len(res_a["path_traversal"]["unnecessary_reads"]),
                "mapa_consultado": res_a["map_metrics"]["map_consulted"],
                "passo_consulta_mapa": res_a["map_metrics"]["map_consulted_step"],
                "tokens_totais": res_a["resource_consumption"]["total_tokens"],
                "tokens_raciocinio": res_a["resource_consumption"]["reasoning_tokens"],
                "custo_usd": res_a["resource_consumption"]["estimated_cost_usd"],
                "latencia_s": res_a["resource_consumption"]["wall_clock_seconds"],
                "caminho_percorrido": res_a["path_traversal"]["files_read_sequence"],
            },
            "B_mapa_completo": {
                "sucesso": res_b["evaluation"]["is_correct"],
                "total_acoes": res_b["resource_consumption"]["total_actions"],
                "chamadas_api": res_b["resource_consumption"]["api_calls"],
                "arquivos_lidos": res_b["path_traversal"]["total_reads"],
                "leituras_desnecessarias": len(res_b["path_traversal"]["unnecessary_reads"]),
                "mapa_consultado": res_b["map_metrics"]["map_consulted"],
                "passo_consulta_mapa": res_b["map_metrics"]["map_consulted_step"],
                "tokens_totais": res_b["resource_consumption"]["total_tokens"],
                "tokens_raciocinio": res_b["resource_consumption"]["reasoning_tokens"],
                "custo_usd": res_b["resource_consumption"]["estimated_cost_usd"],
                "latencia_s": res_b["resource_consumption"]["wall_clock_seconds"],
                "caminho_percorrido": res_b["path_traversal"]["files_read_sequence"],
            },
            "C_mapa_ruidoso": {
                "sucesso": res_c["evaluation"]["is_correct"],
                "total_acoes": res_c["resource_consumption"]["total_actions"],
                "chamadas_api": res_c["resource_consumption"]["api_calls"],
                "arquivos_lidos": res_c["path_traversal"]["total_reads"],
                "leituras_desnecessarias": len(res_c["path_traversal"]["unnecessary_reads"]),
                "mapa_consultado": res_c["map_metrics"]["map_consulted"],
                "passo_consulta_mapa": res_c["map_metrics"]["map_consulted_step"],
                "tokens_totais": res_c["resource_consumption"]["total_tokens"],
                "tokens_raciocinio": res_c["resource_consumption"]["reasoning_tokens"],
                "custo_usd": res_c["resource_consumption"]["estimated_cost_usd"],
                "latencia_s": res_c["resource_consumption"]["wall_clock_seconds"],
                "caminho_percorrido": res_c["path_traversal"]["files_read_sequence"],
            },
        },
        "conditions_detail": {
            "A": res_a,
            "B": res_b,
            "C": res_c,
        },
    }

    report_file = results_dir / "experiment_05_report.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 75)
    print("EXPERIMENTO 05 CONCLUÍDO COM SUCESSO!")
    print(f"Resultados salvos em:")
    print(f" - {file_a}")
    print(f" - {file_b}")
    print(f" - {file_c}")
    print(f" - {report_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
