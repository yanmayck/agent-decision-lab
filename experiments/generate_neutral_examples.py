"""Generates benchmark examples with 10 and 100 files using the neutral graph generator.

Outputs full graph statistics and verifies topological ambiguity.
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treasuregraph.generator import generate_world
from treasuregraph.map_generator import (
    compute_graph_statistics,
    generate_document_map,
    solve_topology_only,
)


def run_examples():
    output_base = Path("experiments/examples")
    output_base.mkdir(parents=True, exist_ok=True)

    results = {}

    for size in [10, 100]:
        w_dir = output_base / f"world_neutral_{size:03d}"
        key = generate_world(
            seed=42,
            output_dir=w_dir,
            language="pt-BR",
            total_files=size,
            neutral_ids=True,
        )

        doc_map = generate_document_map(key)
        stats = compute_graph_statistics(
            doc_map,
            solution_path=key["solution_path"],
            start_node_id=key["start_file"],
        )

        results[size] = {
            "key": key,
            "stats": stats,
            "world_dir": str(w_dir),
        }

        print(f"\n=======================================================")
        print(f"=== EXEMPLO: GRAFO NEUTRO COM {size} ARQUIVOS (SEED 42) ===")
        print(f"=======================================================")
        print(f"Diretório do Ambiente: {w_dir}")
        print(f"Código Secreto: {key['secret_code']}")
        print(f"Ponto de Partida Inicial: {key['start_file']}")
        print(f"Cadeia Gabarito (Genuína): {' -> '.join(key['solution_path'])}")
        print(f"Total de Nós (V): {stats['node_count']}")
        print(f"Total de Arestas (E): {stats['edge_count']} (Genuínas: 2, Distratoras: {stats['edge_count'] - 2})")
        print(f"Componentes Conectados (Fracos): {stats['connected_components_count']}")
        print(f"Nós Isolados (grau 0): {stats['isolated_nodes_count']}")
        print(f"Becos Sem Saída (Dead Ends - out=0, in>0): {stats['dead_ends_count']}")
        print(f"Nós com Ramificação (Branching - out>1): {stats['branching_nodes_count']}")
        print(f"Ciclos Detectados: {stats['cycles_count']}")
        print(f"Caminhos Candidatos de Comprimento 3 a partir de {key['start_file']}: {stats['candidate_paths_from_start']}")
        for idx, p in enumerate(stats['candidate_paths_start_list'], 1):
            is_true = (p == key['solution_path'])
            tag = "[GABARITO]" if is_true else "[DISTRATOR]"
            print(f"  Caminho {idx} {tag}: {' -> '.join(p)}")
        print(f"Total de Caminhos de Comprimento 3 em Todo o Grafo: {stats['total_candidate_paths']}")
        print(f"Ambiguidade Topológica Garantida: {stats['is_topologically_ambiguous']}")

    summary_file = output_base / "neutral_graphs_summary.json"
    summary_file.write_text(
        json.dumps(
            {
                str(k): {
                    "secret_code": v["key"]["secret_code"],
                    "solution_path": v["key"]["solution_path"],
                    "start_file": v["key"]["start_file"],
                    "stats": {
                        sk: sv
                        for sk, sv in v["stats"].items()
                        if sk != "candidate_paths_start_list"
                    },
                    "candidate_paths_start": v["stats"]["candidate_paths_start_list"],
                }
                for k, v in results.items()
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] Resumo dos grafos salvo em: {summary_file}")


if __name__ == "__main__":
    run_examples()
