"""Runner script for Experiment 06: Scalability of Semantic Document Maps.

Evaluates how semantic document maps scale across problem sizes (5, 10, 20, 50, 100 files)
under three semantic description conditions:
1. sem_descricao: map with topological connections only (no node descriptions)
2. descricao_curta: map with short neutral topic summaries
3. descricao_detalhada: map with detailed neutral contextual summaries

Experimental Settings:
- Sizes: 5, 10, 20, 50, 100 files
- Seed: 42 in all environments
- Language: pt-BR
- Model: gemini-3.5-flash-lite
- Thinking: HIGH
- Action limits:
    - 5 files: 15 actions
    - 10 files: 20 actions
    - 20 files: 25 actions
    - 50 files: 35 actions
    - 100 files: 45 actions
- Safety budget ceiling: US$ 2.00
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from treasuregraph.agent import BaselineAgent, load_dotenv
from treasuregraph.generator import generate_world
from treasuregraph.instrumentation import PricingConfig
from treasuregraph.map_generator import MapMode, generate_document_map

SIZES = [5, 10, 20, 50, 100]
CONDITIONS = [
    ("sem_descricao", "none", "Sem Descrição"),
    ("descricao_curta", "short", "Descrição Curta"),
    ("descricao_detalhada", "detailed", "Descrição Detalhada"),
]

MAX_STEPS_BY_SIZE = {
    5: 15,
    10: 20,
    20: 25,
    50: 35,
    100: 45,
}

SPENDING_CAP_USD = 2.00


def validate_map_security(agent_map_dict: dict[str, Any], secret_code: str, label: str) -> None:
    """Verifies that an agent map reveals zero secret code, decoy flags, or ground truth labels."""
    agent_json = json.dumps(agent_map_dict)
    if secret_code in agent_json:
        raise ValueError(f"SECURITY BREACH in {label}: Secret code '{secret_code}' found in agent map!")
    for prefix in ("ISCA-", "DECOY-"):
        if prefix in agent_json:
            raise ValueError(f"SECURITY BREACH in {label}: Decoy code prefix '{prefix}' found in agent map!")
    forbidden_terms = ["solution_path", "is_terminal", "has_treasure", "is_true_clue", "role", "secret_code"]
    for term in forbidden_terms:
        if f'"{term}"' in agent_json:
            raise ValueError(f"SECURITY BREACH in {label}: Ground truth metadata key '{term}' found in agent map!")
    for val in ("fake", "falsa", "is_valid", "spurious"):
        if val in agent_json.lower():
            raise ValueError(f"SECURITY BREACH in {label}: Falsity indicator '{val}' found in agent map!")


def estimate_total_budget(pricing: PricingConfig) -> float:
    """Estimates the upper-bound cost for 15 runs."""
    # Conservative estimate: avg 10k prompt tokens, 500 completion tokens per run across 15 runs
    total_est = 0.0
    for s in SIZES:
        for _ in CONDITIONS:
            p_est = 2000 + s * 100 * 3
            c_est = 300
            cost = pricing.calculate_cost(p_est, c_est, reasoning_tokens=250, reasoning_included_in_output=False) or 0.001
            total_est += cost
    return round(total_est, 4)


def generate_plots(results: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Generates comparative scaling curves and saves PNG figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files: list[str] = []

    cond_labels = {
        "sem_descricao": ("Sem Descrição", "#1f77b4", "o-"),
        "descricao_curta": ("Descrição Curta", "#2ca02c", "s--"),
        "descricao_detalhada": ("Descrição Detalhada", "#d62728", "^-."),
    }

    metrics_to_plot = [
        ("total_actions", "Ações Totais", "Ações × Quantidade de Arquivos", "experiment_06_actions_vs_files.png"),
        ("total_tokens", "Tokens Totais", "Tokens × Quantidade de Arquivos", "experiment_06_tokens_vs_files.png"),
        ("estimated_cost_usd", "Custo Estimado (USD)", "Custo × Quantidade de Arquivos", "experiment_06_cost_vs_files.png"),
        ("latency_seconds", "Latência Total (s)", "Latência × Quantidade de Arquivos", "experiment_06_latency_vs_files.png"),
    ]

    # Individual plots
    for field_name, ylabel, title, filename in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        for cond_id, (label, color, marker) in cond_labels.items():
            subset = [r for r in results if r["condition"] == cond_id]
            subset.sort(key=lambda r: r["file_count"])
            xs = [r["file_count"] for r in subset]
            ys = [r[field_name] for r in subset]
            plt.plot(xs, ys, marker, color=color, label=label, linewidth=2, markersize=7)

        plt.title(f"TreasureGraph Exp 06: {title}", fontsize=13, fontweight="bold")
        plt.xlabel("Quantidade de Arquivos no Ambiente (N)", fontsize=11)
        plt.ylabel(ylabel, fontsize=11)
        plt.xticks(SIZES)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        out_path = output_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close()
        plot_files.append(str(out_path))

    # Consolidated 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, (field_name, ylabel, title, _) in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        for cond_id, (label, color, marker) in cond_labels.items():
            subset = [r for r in results if r["condition"] == cond_id]
            subset.sort(key=lambda r: r["file_count"])
            xs = [r["file_count"] for r in subset]
            ys = [r[field_name] for r in subset]
            ax.plot(xs, ys, marker, color=color, label=label, linewidth=2, markersize=6)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Quantidade de Arquivos (N)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(SIZES)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)

    plt.suptitle("TreasureGraph - Experimento 06: Escalabilidade de Mapas Semânticos", fontsize=15, fontweight="bold")
    plt.tight_layout()
    grid_path = output_dir / "experiment_06_summary_grid.png"
    plt.savefig(grid_path, dpi=200)
    plt.close()
    plot_files.append(str(grid_path))

    return plot_files


def main() -> None:
    load_dotenv()

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    worlds_base = Path("experiments/scalability_worlds")
    worlds_base.mkdir(parents=True, exist_ok=True)

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

    est_budget = estimate_total_budget(pricing)

    print("=" * 80)
    print("TREASUREGRAPH - EXPERIMENTO 06: ESCALABILIDADE DE MAPAS SEMÂNTICOS")
    print("=" * 80)
    print(f"Modelo: {model_name} | Raciocínio: HIGH | Seed: 42 | Idioma: pt-BR")
    print(f"Tamanhos: {SIZES} | Total de execuções planejadas: {len(SIZES) * len(CONDITIONS)}")
    print(f"Estimativa de custo total: ~${est_budget:.4f} USD (Teto de segurança: ${SPENDING_CAP_USD:.2f} USD)")

    if est_budget > SPENDING_CAP_USD:
        print(f"Abortando: estimativa inicial (${est_budget}) excede o teto de ${SPENDING_CAP_USD}.")
        sys.exit(1)

    # 1. Gerar os 5 ambientes e validar os mapas previamente
    print("\n[1/3] Geração dos ambientes e validação prévia de segurança...")
    env_keys: dict[int, dict[str, Any]] = {}
    env_dirs: dict[int, Path] = {}

    for size in SIZES:
        w_dir = worlds_base / f"world_pt_size_{size:03d}"
        ak_path = worlds_base / f"answer_key_size_{size:03d}.json"
        key = generate_world(
            seed=42,
            output_dir=w_dir,
            answer_key_path=ak_path,
            language="pt-BR",
            total_files=size,
        )
        env_keys[size] = key
        env_dirs[size] = w_dir

        correct_code = key["secret_code"]
        # Validar as 3 condições de mapa para este tamanho
        for cond_id, desc_mode, _ in CONDITIONS:
            g = generate_document_map(key, mode=MapMode.COMPLETE, description_mode=desc_mode)
            validate_map_security(g.to_agent_dict(), correct_code, f"Size {size} - {cond_id}")

    print("[OK] Todos os 5 ambientes e 15 mapas validados com ZERO vazamento de código ou gabarito.")

    # 2. Executar as 15 condições registrando consumo e verificando teto de custo
    print("\n[2/3] Executando os 15 testes experimentais...")
    all_results: list[dict[str, Any]] = []
    cumulative_cost_usd = 0.0
    run_idx = 0

    for size in SIZES:
        w_dir = env_dirs[size]
        key = env_keys[size]
        correct_code = key["secret_code"]
        solution_path = key["solution_path"]
        max_steps = MAX_STEPS_BY_SIZE[size]

        for cond_id, desc_mode, cond_label in CONDITIONS:
            run_idx += 1
            print(f"\n--- [Run {run_idx}/15] Tamanho: {size} arquivos | Condição: {cond_label} (Limite: {max_steps} passos) ---")

            if cumulative_cost_usd >= SPENDING_CAP_USD:
                print(f"ALERTA: Teto de gastos (${SPENDING_CAP_USD:.2f}) atingido (${cumulative_cost_usd:.4f}). Interrompendo.")
                break

            # Gerar o mapa da condição
            doc_map = generate_document_map(key, mode=MapMode.COMPLETE, description_mode=desc_mode)
            agent_map_dict = doc_map.to_agent_dict()
            map_bytes = len(json.dumps(agent_map_dict, ensure_ascii=False).encode("utf-8"))
            map_tokens_est = max(1, -(-map_bytes // 4))

            agent = BaselineAgent(
                world_dir=w_dir,
                model=model_name,
                api_key=api_key,
                api_base=api_base,
                max_steps=max_steps,
                reasoning_effort="high",
                document_map=doc_map,
                pricing_config=pricing,
            )

            t_start = time.perf_counter()
            agent_result = agent.run()
            wall_dur = round(time.perf_counter() - t_start, 4)

            traj = agent_result.get("trajectory", [])
            submitted = agent_result.get("submitted_code")
            is_correct = (submitted == correct_code) if submitted else False
            status = agent_result.get("status")
            interrupted = status == "max_steps_reached"

            files_read = [
                step["arguments"].get("path")
                for step in traj
                if step["action"] == "read_file" and "path" in step["arguments"]
            ]
            map_calls = [step for step in traj if step["action"] == "get_map"]
            irrelevant_reads = [f for f in files_read if f not in solution_path]

            cost_this_run = agent_result.get("estimated_cost_usd") or 0.0
            cumulative_cost_usd += cost_this_run

            record = {
                "run_id": run_idx,
                "file_count": size,
                "condition": cond_id,
                "condition_label": cond_label,
                "success": is_correct,
                "status": status,
                "interrupted_by_limit": interrupted,
                "submitted_code": submitted,
                "expected_code": correct_code,
                "total_actions": len(traj),
                "api_calls": agent_result.get("api_calls_count", 0),
                "files_read_count": len(files_read),
                "irrelevant_reads_count": len(irrelevant_reads),
                "parallel_reads_count": agent_result.get("parallel_reads_count", 0),
                "sequential_reads_count": agent_result.get("sequential_reads_count", 0),
                "input_tokens": agent_result.get("input_tokens", 0),
                "output_tokens": agent_result.get("output_tokens", 0),
                "reasoning_tokens": agent_result.get("reasoning_tokens", 0) or 0,
                "total_tokens": agent_result.get("total_tokens", {}).get("total_tokens", 0),
                "map_size_bytes": map_bytes,
                "map_size_tokens_est": map_tokens_est,
                "map_consulted": len(map_calls) > 0,
                "map_consulted_step": map_calls[0]["step"] if map_calls else None,
                "estimated_cost_usd": round(cost_this_run, 6),
                "cumulative_cost_usd": round(cumulative_cost_usd, 6),
                "latency_seconds": wall_dur,
                "files_read_sequence": files_read,
                "actions_sequence": [f"{step['step']}. {step['action']}({step['arguments']})" for step in traj],
            }
            all_results.append(record)

            print(f" [OK] Status: {status} | Sucesso: {is_correct} | Ações: {len(traj)} | APIs: {record['api_calls']}")
            print(f"   Arquivos lidos: {len(files_read)} (Irrelevantes: {len(irrelevant_reads)}) | Paralelo: {record['parallel_reads_count']} | Seq: {record['sequential_reads_count']}")
            print(f"   Tokens: {record['total_tokens']} (Thinking: {record['reasoning_tokens']}) | Custo: ${cost_this_run:.6f} | Tempo: {wall_dur}s")

            # Intervalo de cortesia
            time.sleep(2)

    # 3. Salvar dados brutos em JSON e CSV
    print("\n[3/3] Salvando dados brutos e gerando gráficos...")
    json_path = results_dir / "experiment_06_data.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = results_dir / "experiment_06_data.csv"
    if all_results:
        fieldnames = [
            "run_id", "file_count", "condition", "condition_label", "success", "status",
            "interrupted_by_limit", "total_actions", "api_calls", "files_read_count",
            "irrelevant_reads_count", "parallel_reads_count", "sequential_reads_count",
            "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
            "map_size_bytes", "map_size_tokens_est", "map_consulted", "map_consulted_step",
            "estimated_cost_usd", "cumulative_cost_usd", "latency_seconds"
        ]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

    # Gerar gráficos
    plot_files = generate_plots(all_results, results_dir)

    # Relatório comparativo
    report = {
        "experiment": "TreasureGraph - Experimento 06: Escalabilidade de Mapas Semânticos",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name,
        "reasoning_effort": "high",
        "spending_cap_usd": SPENDING_CAP_USD,
        "total_cost_usd": round(cumulative_cost_usd, 6),
        "total_runs_executed": len(all_results),
        "configurations": {
            "sizes": SIZES,
            "conditions": [c[0] for c in CONDITIONS],
            "max_steps_by_size": MAX_STEPS_BY_SIZE,
        },
        "plot_files": plot_files,
        "runs": all_results,
    }
    report_path = results_dir / "experiment_06_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("EXPERIMENTO 06 FINALIZADO!")
    print(f"Custo total incorrido: ${cumulative_cost_usd:.5f} USD (Teto de ${SPENDING_CAP_USD:.2f} respeitado).")
    print(f"Dados brutos JSON: {json_path}")
    print(f"Dados brutos CSV:  {csv_path}")
    print(f"Relatório final:   {report_path}")
    print("Gráficos gerados:")
    for pf in plot_files:
        print(f" - {pf}")
    print("=" * 80)


if __name__ == "__main__":
    main()
