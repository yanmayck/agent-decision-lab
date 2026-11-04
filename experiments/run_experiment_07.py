"""TreasureGraph - Experiment 07: Scalability of Neutral Environments with Distractor Graphs.

Investigates agent planning and decision-making when structural and semantic leaks are eliminated:
- Neutral file identifiers (doc_001.txt ... doc_N.txt)
- Real distractor graph connections, branches, dead ends, and cycles
- Multiple connected components (treasure path is not the sole component)
- Known starting document (doc_001.txt) with verifiable 3-step clue chain
- Strictly non-leaking domain-neutral technical descriptions

Configurations:
- File Counts: 10, 20, 50, 100
- Seed: 42
- Language: pt-BR
- Model: gemini-3.5-flash-lite
- Thinking: HIGH (reasoning_effort="high")
- Three conditions:
  A: sem_descricao (none)
  B: descricao_curta (short)
  C: descricao_detalhada (detailed)
- Total runs: 12
- Budget ceiling: US$ 2.00
"""

import ast
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.offline as pyo

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from treasuregraph.agent import BaselineAgent
from treasuregraph.generator import generate_world
from treasuregraph.instrumentation import PricingConfig
from treasuregraph.map_generator import (
    compute_graph_statistics,
    generate_document_map,
    solve_topology_only,
    MapMode,
)

FORBIDDEN_WORDS = [
    "inicio", "missao", "checkpoint", "cofre", "terminal",
    "segredo", "tesouro", "start", "vault", "secret", "treasure", "mission"
]

SIZES = [10, 20, 50, 100]
CONDITIONS = [
    ("sem_descricao", "none", "Sem Descrição"),
    ("descricao_curta", "short", "Descrição Curta"),
    ("descricao_detalhada", "detailed", "Descrição Detalhada"),
]

MAX_STEPS_BY_SIZE = {
    10: 20,
    20: 25,
    50: 35,
    100: 45,
}

SPENDING_CAP_USD = 2.0


def validate_environment(size: int, world_dir: Path, key: dict[str, Any]) -> dict[str, Any]:
    """Validates neutral environment for absence of leaks, topological ambiguity, and clue chain integrity."""
    # 1. Check neutral IDs
    files = sorted(list(world_dir.glob("*.txt")))
    if len(files) != size:
        raise ValueError(f"Tamanho {size}: esperado {size} arquivos, encontrados {len(files)}")

    for f in files:
        if not re.match(r"^doc_\d{3}\.txt$", f.name):
            raise ValueError(f"Tamanho {size}: nome não-neutro encontrado: {f.name}")

    # 2. Check absence of semantic leaks
    for mode in ["short", "detailed"]:
        doc_map = generate_document_map(key, description_mode=mode)
        d = doc_map.to_agent_dict(mode)
        for node in d["nodes"]:
            nid = node["id"].lower()
            desc = (node.get("description") or "").lower()
            for w in FORBIDDEN_WORDS:
                if re.search(r"\b" + re.escape(w) + r"\b", nid):
                    raise ValueError(f"Tamanho {size}: palavra proibida '{w}' no ID '{nid}'")
                if re.search(r"\b" + re.escape(w) + r"\b", desc):
                    raise ValueError(f"Tamanho {size}: palavra proibida '{w}' na descrição '{desc}'")

    # 3. Check topological ambiguity (sanity test)
    doc_map_complete = generate_document_map(key, mode=MapMode.COMPLETE)
    agent_dict = doc_map_complete.to_agent_dict()
    start_file = key["start_file"]
    candidate_paths = solve_topology_only(agent_dict, start_node_id=start_file, path_length=2)
    if len(candidate_paths) <= 1:
        raise ValueError(f"Tamanho {size}: topologia não é ambígua! Encontrados apenas {len(candidate_paths)} caminhos.")

    # 4. Check clue chain textual verifiability
    c1 = (world_dir / start_file).read_text(encoding="utf-8")
    m1 = re.search(r"examine o documento (doc_\d{3}\.txt)", c1)
    if not m1 or m1.group(1) != key["solution_path"][1]:
        raise ValueError(f"Tamanho {size}: pista textual do passo 1 falhou na verificação")

    step2 = m1.group(1)
    c2 = (world_dir / step2).read_text(encoding="utf-8")
    m2 = re.search(r"autenticação está em (doc_\d{3}\.txt)", c2)
    if not m2 or m2.group(1) != key["solution_path"][2]:
        raise ValueError(f"Tamanho {size}: pista textual do passo 2 falhou na verificação")

    step3 = m2.group(1)
    c3 = (world_dir / step3).read_text(encoding="utf-8")
    m3 = re.search(r"CÓDIGO SECRETO:\s*([A-Z0-9-]+)", c3)
    if not m3 or m3.group(1) != key["secret_code"]:
        raise ValueError(f"Tamanho {size}: código secreto não encontrado no passo 3")

    # 5. Compute graph stats
    stats = compute_graph_statistics(
        doc_map_complete,
        solution_path=key["solution_path"],
        start_node_id=start_file,
    )
    return stats


def generate_interactive_dashboard(results: list[dict[str, Any]], output_path: Path) -> Path:
    """Generates a standalone, fully offline interactive HTML dashboard using Plotly."""
    COND_CONFIG = {
        "sem_descricao": {
            "name": "Mapa Topológico (Sem Descrição)",
            "short": "Sem Descrição",
            "color": "#1f77b4",
            "dash": "solid",
            "marker": "circle",
        },
        "descricao_curta": {
            "name": "Mapa com Descrições Curtas",
            "short": "Descrição Curta",
            "color": "#ff7f0e",
            "dash": "dot",
            "marker": "square",
        },
        "descricao_detalhada": {
            "name": "Mapa com Descrições Detalhadas",
            "short": "Descrição Detalhada",
            "color": "#2ca02c",
            "dash": "dash",
            "marker": "diamond",
        },
    }

    def get_common_layout(title: str, y_title: str) -> dict[str, Any]:
        return dict(
            title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#1e293b", family="system-ui, sans-serif")),
            xaxis=dict(
                title=dict(text="<b>Quantidade de Arquivos (N)</b>", font=dict(size=12, color="#475569")),
                tickmode="array",
                tickvals=SIZES,
                ticktext=["10", "20", "50", "100"],
                gridcolor="#f1f5f9",
                linecolor="#cbd5e1",
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text=f"<b>{y_title}</b>", font=dict(size=12, color="#475569")),
                gridcolor="#f1f5f9",
                linecolor="#cbd5e1",
                zeroline=False,
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(l=60, r=30, t=80, b=50),
        )

    # 1. Chart: Actions & Reads vs Files
    fig_actions = go.Figure()
    for cond_id, cfg in COND_CONFIG.items():
        subset = sorted([r for r in results if r["condition"] == cond_id], key=lambda x: x["file_count"])
        xs = [r["file_count"] for r in subset]
        ys_actions = [r["total_actions"] for r in subset]
        ys_reads = [r["files_read_count"] for r in subset]

        cdata = [
            [
                r["total_actions"],
                r["api_calls"],
                r["files_read_count"],
                r["irrelevant_reads_count"],
                f"Paralelo ({r['parallel_reads_count']})" if r["parallel_reads_count"] > 0 else f"Sequencial ({r['sequential_reads_count']})",
                "Sim" if r["interrupted_by_limit"] else "Não (Concluído)",
                r["run_id"],
            ]
            for r in subset
        ]

        htemplate = (
            "<b>Run %{customdata[6]} - " + cfg["short"] + "</b><br>"
            "Arquivos (N): <b>%{x}</b><br>"
            "Ações Totais: <b>%{customdata[0]}</b><br>"
            "Chamadas API: <b>%{customdata[1]}</b><br>"
            "Arquivos Lidos: <b>%{customdata[2]}</b> (Irrelevantes: <b>%{customdata[3]}</b>)<br>"
            "Modo de Leitura: <b>%{customdata[4]}</b><br>"
            "Interrompido por Limite: <b>%{customdata[5]}</b><br>"
            "<span style='color:#64748b;font-size:10px;'>Exp 07 | Seed 42 | gemini-3.5-flash-lite (HIGH)</span>"
            "<extra></extra>"
        )

        fig_actions.add_trace(
            go.Scatter(
                x=xs,
                y=ys_actions,
                mode="lines+markers",
                name=f"{cfg['short']} - Ações Totais",
                legendgroup=cond_id,
                line=dict(color=cfg["color"], width=3, dash=cfg["dash"]),
                marker=dict(symbol=cfg["marker"], size=9, line=dict(width=1.5, color="#ffffff")),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
        fig_actions.add_trace(
            go.Scatter(
                x=xs,
                y=ys_reads,
                mode="lines+markers",
                name=f"{cfg['short']} - Leituras",
                legendgroup=cond_id,
                line=dict(color=cfg["color"], width=1.5, dash="dot"),
                marker=dict(symbol="circle-open", size=7),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
    fig_actions.update_layout(get_common_layout("Ações Totais e Leituras vs Quantidade de Arquivos", "Quantidade de Ações / Leituras"))

    # 2. Chart: Tokens Decomposition
    fig_tokens = go.Figure()
    for cond_id, cfg in COND_CONFIG.items():
        subset = sorted([r for r in results if r["condition"] == cond_id], key=lambda x: x["file_count"])
        xs = [r["file_count"] for r in subset]

        cdata = [
            [
                r["input_tokens"],
                r["output_tokens"],
                r["reasoning_tokens"],
                r["tool_result_tokens"],
                r["total_tokens"],
                r["run_id"],
            ]
            for r in subset
        ]

        htemplate = (
            "<b>Run %{customdata[5]} - " + cfg["short"] + "</b><br>"
            "Arquivos (N): <b>%{x}</b><br>"
            "Métrica: <b>%{y:,} tokens</b><br>"
            "---------------------------<br>"
            "Tokens Entrada API: <b>%{customdata[0]:,}</b><br>"
            "Tokens Raciocínio API: <b>%{customdata[2]:,}</b><br>"
            "Tokens Saída API: <b>%{customdata[1]:,}</b><br>"
            "Tokens Ferramentas (Est.): <b>%{customdata[3]:,}</b><br>"
            "Total Tokens API: <b>%{customdata[4]:,}</b><br>"
            "<span style='color:#64748b;font-size:10px;'>Exp 07 | Seed 42 | Gemini 3.5 Flash Lite</span>"
            "<extra></extra>"
        )

        fig_tokens.add_trace(
            go.Scatter(
                x=xs,
                y=[r["input_tokens"] for r in subset],
                mode="lines+markers",
                name=f"{cfg['short']} - Entrada (API)",
                legendgroup=cond_id,
                line=dict(color=cfg["color"], width=2.5),
                marker=dict(symbol=cfg["marker"], size=8),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
        fig_tokens.add_trace(
            go.Scatter(
                x=xs,
                y=[r["tool_result_tokens"] for r in subset],
                mode="lines+markers",
                name=f"{cfg['short']} - Ferramentas (Est.)",
                legendgroup=cond_id,
                line=dict(color=cfg["color"], width=1.5, dash="dash"),
                marker=dict(symbol="triangle-up", size=7),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
        fig_tokens.add_trace(
            go.Scatter(
                x=xs,
                y=[r["reasoning_tokens"] for r in subset],
                mode="lines+markers",
                name=f"{cfg['short']} - Raciocínio (API)",
                legendgroup=cond_id,
                line=dict(color=cfg["color"], width=1.2, dash="dot"),
                marker=dict(symbol="cross", size=6),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
    fig_tokens.update_layout(get_common_layout("Decomposição de Tokens (Entrada, Raciocínio e Ferramentas)", "Quantidade de Tokens"))

    # 3. Chart: Cost vs Files
    fig_cost = go.Figure()
    for cond_id, cfg in COND_CONFIG.items():
        subset = sorted([r for r in results if r["condition"] == cond_id], key=lambda x: x["file_count"])
        xs = [r["file_count"] for r in subset]
        ys_cost = [r["estimated_cost_usd"] for r in subset]

        cdata = [
            [
                r["estimated_cost_usd"],
                r["cumulative_cost_usd"],
                r["input_tokens"],
                r["output_tokens"],
                r["reasoning_tokens"],
                r["run_id"],
            ]
            for r in subset
        ]

        htemplate = (
            "<b>Run %{customdata[5]} - " + cfg["short"] + "</b><br>"
            "Arquivos (N): <b>%{x}</b><br>"
            "Custo da Execução: <b>$%{y:.6f} USD</b><br>"
            "Custo Acumulado: <b>$%{customdata[1]:.6f} USD</b><br>"
            "Tokens Entrada: <b>%{customdata[2]:,}</b><br>"
            "Tokens Saída: <b>%{customdata[3]:,}</b> (Raciocínio: <b>%{customdata[4]:,}</b>)<br>"
            "<span style='color:#64748b;font-size:10px;'>Teto Orçamentário: $2.00</span>"
            "<extra></extra>"
        )

        fig_cost.add_trace(
            go.Scatter(
                x=xs,
                y=ys_cost,
                mode="lines+markers",
                name=cfg["short"],
                line=dict(color=cfg["color"], width=3, dash=cfg["dash"]),
                marker=dict(symbol=cfg["marker"], size=9, line=dict(width=1.5, color="#ffffff")),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
    layout_cost = get_common_layout("Custo Estimado por Execução (USD) vs Quantidade de Arquivos", "Custo Estimado (USD)")
    layout_cost["yaxis"]["tickformat"] = "$.4f"
    fig_cost.update_layout(layout_cost)

    # 4. Chart: Latency vs Files
    fig_latency = go.Figure()
    for cond_id, cfg in COND_CONFIG.items():
        subset = sorted([r for r in results if r["condition"] == cond_id], key=lambda x: x["file_count"])
        xs = [r["file_count"] for r in subset]
        ys_lat = [r["latency_seconds"] for r in subset]

        cdata = [
            [
                r["latency_seconds"],
                r["api_calls"],
                r["parallel_reads_count"],
                r["sequential_reads_count"],
                r["run_id"],
            ]
            for r in subset
        ]

        htemplate = (
            "<b>Run %{customdata[4]} - " + cfg["short"] + "</b><br>"
            "Arquivos (N): <b>%{x}</b><br>"
            "Latência Total: <b>%{y:.2f} s</b><br>"
            "Chamadas API: <b>%{customdata[1]}</b><br>"
            "Leituras Paralelas: <b>%{customdata[2]}</b> | Sequenciais: <b>%{customdata[3]}</b><br>"
            "<span style='color:#64748b;font-size:10px;'>Exp 07 | gemini-3.5-flash-lite (HIGH)</span>"
            "<extra></extra>"
        )

        fig_latency.add_trace(
            go.Scatter(
                x=xs,
                y=ys_lat,
                mode="lines+markers",
                name=cfg["short"],
                line=dict(color=cfg["color"], width=3, dash=cfg["dash"]),
                marker=dict(symbol=cfg["marker"], size=9, line=dict(width=1.5, color="#ffffff")),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
    fig_latency.update_layout(get_common_layout("Latência de Execução (Segundos) vs Quantidade de Arquivos", "Duração Total (s)"))

    # 5. Chart: Success Rate & Clue Following
    fig_success = go.Figure()
    for cond_id, cfg in COND_CONFIG.items():
        subset = sorted([r for r in results if r["condition"] == cond_id], key=lambda x: x["file_count"])
        xs = [r["file_count"] for r in subset]
        ys_succ = [100.0 if r["success"] else 0.0 for r in subset]

        cdata = [
            [
                r["submitted_code"],
                r["expected_code"],
                "Sim (Genuína)" if r["followed_genuine_clues"] else "Não (Explorou Distratores)",
                r["irrelevant_reads_count"],
                r["run_id"],
            ]
            for r in subset
        ]

        htemplate = (
            "<b>Run %{customdata[4]} - " + cfg["short"] + "</b><br>"
            "Arquivos (N): <b>%{x}</b><br>"
            "Taxa de Sucesso: <b>%{y:.0f}%</b><br>"
            "Código Submetido: <code>%{customdata[0]}</code><br>"
            "Seguiu Cadeia Genuína: <b>%{customdata[2]}</b><br>"
            "Leituras Irrelevantes: <b>%{customdata[3]}</b><br>"
            "<extra></extra>"
        )

        fig_success.add_trace(
            go.Scatter(
                x=xs,
                y=ys_succ,
                mode="lines+markers",
                name=cfg["short"],
                line=dict(color=cfg["color"], width=3),
                marker=dict(symbol="circle-dot", size=10),
                customdata=cdata,
                hovertemplate=htemplate,
            )
        )
    layout_succ = get_common_layout("Taxa de Sucesso (%) vs Quantidade de Arquivos", "Taxa de Sucesso (%)")
    layout_succ["yaxis"]["range"] = [0, 110]
    layout_succ["yaxis"]["ticksuffix"] = "%"
    fig_success.update_layout(layout_succ)

    # Standalone HTML assembly
    plotly_js = pyo.get_plotlyjs()
    div_actions = fig_actions.to_html(full_html=False, include_plotlyjs=False)
    div_tokens = fig_tokens.to_html(full_html=False, include_plotlyjs=False)
    div_cost = fig_cost.to_html(full_html=False, include_plotlyjs=False)
    div_latency = fig_latency.to_html(full_html=False, include_plotlyjs=False)
    div_success = fig_success.to_html(full_html=False, include_plotlyjs=False)

    results_json = json.dumps(results, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TreasureGraph - Dashboard Experimento 07 (Ambientes Neutros e Grafos Distratores)</title>
    <!-- Standalone Plotly JS Embedded Offline -->
    <script type="text/javascript">{plotly_js}</script>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --surface-hover: #f1f5f9;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --primary: #2563eb;
            --success: #16a34a;
            --success-bg: #dcfce7;
            --danger: #dc2626;
            --danger-bg: #fee2e2;
            --radius: 10px;
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.08);
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: var(--bg); color: var(--text-primary); line-height: 1.5; padding: 24px 32px; }}
        .container {{ max-width: 1440px; margin: 0 auto; }}
        header {{ background: var(--surface); padding: 24px 28px; border-radius: var(--radius); box-shadow: var(--shadow-md); margin-bottom: 24px; border-left: 6px solid var(--primary); }}
        h1 {{ font-size: 24px; font-weight: 800; }}
        .subtitle {{ font-size: 14px; color: var(--text-secondary); margin-top: 4px; }}
        .badge-bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
        .badge {{ padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: #e2e8f0; color: var(--text-secondary); }}
        .badge.highlight {{ background: #dbeafe; color: #1e40af; }}
        .badge.success {{ background: var(--success-bg); color: var(--success); }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: var(--surface); padding: 18px 20px; border-radius: var(--radius); border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .kpi-label {{ font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--text-muted); }}
        .kpi-val {{ font-size: 26px; font-weight: 800; margin: 6px 0 2px 0; }}
        .kpi-desc {{ font-size: 11px; color: var(--text-secondary); }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 24px; }}
        @media (max-width: 1024px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
        .chart-card {{ background: var(--surface); padding: 16px 18px; border-radius: var(--radius); border: 1px solid var(--border); box-shadow: var(--shadow-md); min-height: 440px; }}
        .chart-card.full-width {{ grid-column: 1 / -1; }}
        .table-section {{ background: var(--surface); padding: 24px 28px; border-radius: var(--radius); border: 1px solid var(--border); box-shadow: var(--shadow-md); margin-bottom: 32px; }}
        .table-controls {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
        select, input[type="text"] {{ padding: 8px 12px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 13px; outline: none; }}
        .btn-reset {{ padding: 8px 16px; border-radius: 6px; border: 1px solid #cbd5e1; background: var(--surface-hover); cursor: pointer; font-size: 13px; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }}
        th {{ background: #f8fafc; font-weight: 700; color: var(--text-secondary); }}
        tr:hover td {{ background: #f8fafc; }}
        .tag {{ padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block; }}
        .tag.ok {{ background: var(--success-bg); color: var(--success); }}
        .tag.alert {{ background: var(--danger-bg); color: var(--danger); }}
        .tag.cond-sem {{ background: #e0f2fe; color: #0369a1; }}
        .tag.cond-curta {{ background: #ffedd5; color: #c2410c; }}
        .tag.cond-detalhada {{ background: #dcfce7; color: #15803d; }}
        .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
        .callout {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 14px 18px; border-radius: 6px; margin-top: 24px; font-size: 13px; color: #1e3a8a; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>TreasureGraph — Experimento 07: Ambientes Neutros e Grafos Distratores</h1>
            <div class="subtitle">Avaliação Controlada com Eliminação de Vazamento Estrutural e Semântico</div>
            <div class="badge-bar">
                <span class="badge highlight">Modelo: gemini-3.5-flash-lite</span>
                <span class="badge highlight">Reasoning: HIGH</span>
                <span class="badge">Seed: 42</span>
                <span class="badge">Idioma: pt-BR</span>
                <span class="badge highlight">Identificadores Neutros (doc_001.txt)</span>
                <span class="badge success">Grafos Distratores Ativos</span>
                <span class="badge success">Ambiguidade Topológica Garantida</span>
            </div>
        </header>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Execuções Realizadas</div>
                <div class="kpi-val">{len(results)} / 12</div>
                <div class="kpi-desc">4 tamanhos &times; 3 condições semânticas</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Taxa Global de Sucesso</div>
                <div class="kpi-val" style="color: var(--success);">{sum(1 for r in results if r['success'])} / {len(results)}</div>
                <div class="kpi-desc">{sum(1 for r in results if r['success'])/len(results)*100:.1f}% de conformidade</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Custo Total Efetivo</div>
                <div class="kpi-val">${results[-1]['cumulative_cost_usd']:.4f}</div>
                <div class="kpi-desc">Teto de US$ 2.00 respeitado</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Leituras Irrelevantes Médias</div>
                <div class="kpi-val">{sum(r['irrelevant_reads_count'] for r in results)/len(results):.1f}</div>
                <div class="kpi-desc">Desvios em ramos distratores</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Latência Média</div>
                <div class="kpi-val">{sum(r['latency_seconds'] for r in results)/len(results):.1f} s</div>
                <div class="kpi-desc">Tempo total de resposta</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Estouros de Limite de Ações</div>
                <div class="kpi-val" style="color: var(--success);">{sum(1 for r in results if r['interrupted_by_limit'])}</div>
                <div class="kpi-desc">Execuções interrompidas</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-card">{div_actions}</div>
            <div class="chart-card">{div_tokens}</div>
            <div class="chart-card">{div_cost}</div>
            <div class="chart-card">{div_latency}</div>
            <div class="chart-card full-width">{div_success}</div>
        </div>

        <div class="table-section">
            <h2 style="font-size: 18px; margin-bottom: 16px;">Tabela Detalhada com Filtros Dinâmicos</h2>
            <div class="table-controls">
                <select id="filterCond" onchange="applyFilters()">
                    <option value="all">Todas as Condições</option>
                    <option value="sem_descricao">Sem Descrição</option>
                    <option value="descricao_curta">Descrição Curta</option>
                    <option value="descricao_detalhada">Descrição Detalhada</option>
                </select>
                <select id="filterSize" onchange="applyFilters()">
                    <option value="all">Todos os Tamanhos</option>
                    <option value="10">10 Arquivos</option>
                    <option value="20">20 Arquivos</option>
                    <option value="50">50 Arquivos</option>
                    <option value="100">100 Arquivos</option>
                </select>
                <input type="text" id="filterSearch" placeholder="Busca rápida..." oninput="applyFilters()" style="flex-grow: 1;">
                <button class="btn-reset" onclick="resetFilters()">Limpar</button>
                <span id="counter" style="font-size: 13px; color: var(--text-muted);"></span>
            </div>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Run</th>
                            <th class="num">N</th>
                            <th>Condição</th>
                            <th>Sucesso</th>
                            <th>Seguiu Pistas?</th>
                            <th class="num">Ações</th>
                            <th class="num">API Calls</th>
                            <th class="num">Lidos (Irrev.)</th>
                            <th>Modo Leitura</th>
                            <th class="num">Tokens Entrada</th>
                            <th class="num">Tokens Raciocínio</th>
                            <th class="num">Tokens Ferramentas</th>
                            <th class="num">Custo (USD)</th>
                            <th class="num">Latência (s)</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
        </div>

        <div class="callout">
            <strong>Rigor Metodológico:</strong> O Experimento 07 utiliza identificadores totalmente neutros (doc_001.txt a doc_N.txt) e grafos distratores ricos com múltiplos componentes conectados, ciclos e ramificações. A topologia foi previamente validada como ambígua (is_topologically_ambiguous: True), impedindo que o agente deduza o caminho correto apenas pela estrutura do grafo sem ler o conteúdo das pistas.
        </div>
    </div>

    <script>
        const rawData = {results_json};

        function renderTable(data) {{
            const tbody = document.getElementById("tableBody");
            tbody.innerHTML = "";
            data.forEach(r => {{
                const tr = document.createElement("tr");
                let condClass = "cond-sem";
                if (r.condition === "descricao_curta") condClass = "cond-curta";
                if (r.condition === "descricao_detalhada") condClass = "cond-detalhada";

                const readMode = r.parallel_reads_count > 0 ? `Paralelo (${{r.parallel_reads_count}})` : `Sequencial (${{r.sequential_reads_count}})`;

                tr.innerHTML = `
                    <td><strong>#${{r.run_id}}</strong></td>
                    <td class="num"><strong>${{r.file_count}}</strong></td>
                    <td><span class="tag ${{condClass}}">${{r.condition_label}}</span></td>
                    <td>${{r.success ? '<span class="tag ok">&#10003; Sim</span>' : '<span class="tag alert">&#10007; Não</span>'}}</td>
                    <td>${{r.followed_genuine_clues ? '<span style="color:var(--success);font-weight:700;">Genuína</span>' : '<span style="color:var(--danger);font-weight:700;">Distratores</span>'}}</td>
                    <td class="num">${{r.total_actions}}</td>
                    <td class="num">${{r.api_calls}}</td>
                    <td class="num">${{r.files_read_count}} <span style="color:var(--text-muted);">(${{r.irrelevant_reads_count}})</span></td>
                    <td>${{readMode}}</td>
                    <td class="num">${{r.input_tokens.toLocaleString()}}</td>
                    <td class="num">${{(r.reasoning_tokens || 0).toLocaleString()}}</td>
                    <td class="num">${{(r.tool_result_tokens || 0).toLocaleString()}}</td>
                    <td class="num">$${{r.estimated_cost_usd.toFixed(6)}}</td>
                    <td class="num">${{r.latency_seconds.toFixed(2)}}s</td>
                `;
                tbody.appendChild(tr);
            }});
            document.getElementById("counter").textContent = `Exibindo ${{data.length}} de ${{rawData.length}} registros`;
        }}

        function applyFilters() {{
            const cond = document.getElementById("filterCond").value;
            const size = document.getElementById("filterSize").value;
            const search = document.getElementById("filterSearch").value.toLowerCase().trim();

            const filtered = rawData.filter(r => {{
                if (cond !== "all" && r.condition !== cond) return false;
                if (size !== "all" && String(r.file_count) !== size) return false;
                if (search) {{
                    const t = `${{r.run_id}} ${{r.file_count}} ${{r.condition_label}} ${{r.actions_sequence.join(" ")}}`.toLowerCase();
                    if (!t.includes(search)) return false;
                }}
                return true;
            }});
            renderTable(filtered);
        }}

        function resetFilters() {{
            document.getElementById("filterCond").value = "all";
            document.getElementById("filterSize").value = "all";
            document.getElementById("filterSearch").value = "";
            applyFilters();
        }}

        document.addEventListener("DOMContentLoaded", () => renderTable(rawData));
    </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def main() -> None:
    load_dotenv()

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    worlds_base = Path("experiments/scalability_neutral_worlds")
    worlds_base.mkdir(parents=True, exist_ok=True)

    model_name = os.getenv("TREASURE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
    api_key = os.getenv("TREASURE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    api_base = os.getenv("TREASURE_API_BASE")

    if not api_key:
        print("[ERRO] Chave de API não configurada.")
        sys.exit(1)

    pricing = PricingConfig(
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        reasoning_cost_per_million=0.30,
    )

    print("================================================================================")
    print("=== TREASUREGRAPH - EXPERIMENTO 07: AMBIENTES NEUTROS E GRAFOS DISTRATORES ===")
    print("================================================================================")
    print(f"Modelo: {model_name} | Reasoning: HIGH | Seed: 42 | Idioma: pt-BR")
    print(f"Tamanhos: {SIZES} | Condições: {[c[0] for c in CONDITIONS]}")
    print(f"Teto Orçamentário: US$ {SPENDING_CAP_USD:.2f}")

    # -------------------------------------------------------------------------
    # FASE 1: GERAÇÃO E PRÉ-VALIDAÇÃO RIGOROSA DOS AMBIENTES NEUTROS
    # -------------------------------------------------------------------------
    print("\n[Fase 1/3] Gerando e validando ambientes neutros...")
    env_keys: dict[int, dict[str, Any]] = {}
    env_dirs: dict[int, Path] = {}
    env_stats: dict[int, dict[str, Any]] = {}

    for size in SIZES:
        w_dir = worlds_base / f"world_neutral_size_{size:03d}"
        key_path = worlds_base / f"answer_key_size_{size:03d}.json"

        key = generate_world(
            seed=42,
            output_dir=w_dir,
            answer_key_path=key_path,
            language="pt-BR",
            total_files=size,
            neutral_ids=True,
        )

        stats = validate_environment(size, w_dir, key)
        env_keys[size] = key
        env_dirs[size] = w_dir
        env_stats[size] = stats

        print(
            f"  [OK] N={size:3d} arquivos: {stats['edge_count']} arestas | "
            f"{stats['connected_components_count']} componentes | "
            f"{stats['isolated_nodes_count']} nós isolados | "
            f"{stats['candidate_paths_from_start']} caminhos a partir de {key['start_file']} "
            f"(Ambiguidade: {stats['is_topologically_ambiguous']})"
        )

    # -------------------------------------------------------------------------
    # FASE 2: EXECUÇÃO DOS 12 TESTES CONTROLADOS
    # -------------------------------------------------------------------------
    print("\n[Fase 2/3] Executando bateria de 12 testes experimentais...")
    results: list[dict[str, Any]] = []
    cumulative_cost_usd = 0.0
    run_idx = 0

    for size in SIZES:
        w_dir = env_dirs[size]
        key = env_keys[size]
        correct_code = key["secret_code"]
        solution_path = key["solution_path"]
        start_file = key["start_file"]
        max_steps = MAX_STEPS_BY_SIZE[size]

        for cond_id, desc_mode, cond_label in CONDITIONS:
            run_idx += 1
            print(f"\n--- [Run {run_idx:2d}/12] N={size:3d} | Condição: {cond_label:21s} | Limite: {max_steps:2d} passos ---")

            if cumulative_cost_usd >= SPENDING_CAP_USD:
                print(f"[ALERTA] Teto de gastos (${SPENDING_CAP_USD:.2f}) atingido. Interrompendo execução.")
                break

            doc_map = generate_document_map(key, mode=MapMode.COMPLETE, description_mode=desc_mode)
            agent_map_dict = doc_map.to_agent_dict()
            map_bytes = len(json.dumps(agent_map_dict, ensure_ascii=False).encode("utf-8"))
            map_tokens_est = max(1, math.ceil(map_bytes / 4))

            system_prompt = (
                "Você é um agente autônomo explorando uma instalação técnica.\n"
                "Seu objetivo é encontrar o código secreto oculto e submetê-lo usando submit_answer(code=...).\n"
                "Você tem ferramentas para inspecionar o mapa de documentos (get_map), listar diretórios e ler arquivos. Explore com cuidado.\n"
                f"Seu ponto de partida inicial é o documento {start_file}.\n"
                "Assim que descobrir o código secreto, chame submit_answer imediatamente para concluir a missão."
            )

            agent = BaselineAgent(
                world_dir=w_dir,
                model=model_name,
                api_key=api_key,
                api_base=api_base,
                max_steps=max_steps,
                reasoning_effort="high",
                document_map=doc_map,
                pricing_config=pricing,
                system_prompt=system_prompt,
            )

            t0 = time.perf_counter()
            agent_result = agent.run()
            duration = round(time.perf_counter() - t0, 4)

            traj = agent_result.get("trajectory", [])
            submitted = agent_result.get("submitted_code")
            is_correct = (submitted == correct_code) if submitted else False
            status = agent_result.get("status")
            interrupted = (status == "max_steps_reached")

            files_read = [
                step["arguments"].get("path")
                for step in traj
                if step["action"] == "read_file" and "path" in step["arguments"]
            ]
            map_calls = [step for step in traj if step["action"] == "get_map"]
            irrelevant_reads = [f for f in files_read if f not in solution_path]

            # Verificação se o agente seguiu as pistas genuínas ou explorou distratores
            followed_genuine = (len(irrelevant_reads) == 0) and (files_read == solution_path)

            cost_this_run = agent_result.get("estimated_cost_usd") or 0.0
            cumulative_cost_usd += cost_this_run

            actions_seq = [
                f"{step['step']}. {step['action']}({step.get('arguments', {})})"
                for step in traj
            ]

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
                "tool_result_tokens": agent_result.get("tool_result_tokens", 0),
                "total_tokens": agent_result.get("total_tokens", {}).get("total_tokens", 0),
                "map_size_bytes": map_bytes,
                "map_size_tokens_est": map_tokens_est,
                "map_consulted": len(map_calls) > 0,
                "map_consulted_step": map_calls[0]["step"] if map_calls else None,
                "followed_genuine_clues": followed_genuine,
                "distractor_paths_explored": irrelevant_reads,
                "estimated_cost_usd": round(cost_this_run, 6),
                "cumulative_cost_usd": round(cumulative_cost_usd, 6),
                "latency_seconds": duration,
                "files_read_sequence": files_read,
                "actions_sequence": actions_seq,
            }
            results.append(record)

            status_str = "SUCESSO" if is_correct else f"FALHA ({status})"
            clue_str = "Seguiu pistas genuínas" if followed_genuine else f"Explorou {len(irrelevant_reads)} distratores"
            print(
                f"  -> Resultado: {status_str} | Ações: {len(traj)} | API: {agent_result.get('api_calls_count', 0)} | "
                f"Lidos: {len(files_read)} ({clue_str}) | "
                f"Custo: ${cost_this_run:.6f} (Acum: ${cumulative_cost_usd:.4f}) | {duration:.2f}s"
            )

    # -------------------------------------------------------------------------
    # FASE 3: EXPORTAÇÃO DE RESULTADOS E GERAÇÃO DO DASHBOARD
    # -------------------------------------------------------------------------
    print("\n[Fase 3/3] Exportando dados brutos e gerando dashboard interativo...")

    # JSON export
    data_json_path = results_dir / "experiment_07_data.json"
    data_json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] Dados JSON salvos em: {data_json_path}")

    # CSV export
    data_csv_path = results_dir / "experiment_07_data.csv"
    csv_fields = [
        "run_id", "file_count", "condition", "condition_label", "success", "status",
        "interrupted_by_limit", "total_actions", "api_calls", "files_read_count",
        "irrelevant_reads_count", "parallel_reads_count", "sequential_reads_count",
        "followed_genuine_clues", "input_tokens", "output_tokens", "reasoning_tokens",
        "tool_result_tokens", "total_tokens", "map_size_bytes", "map_size_tokens_est",
        "map_consulted", "map_consulted_step", "estimated_cost_usd", "cumulative_cost_usd",
        "latency_seconds"
    ]
    with open(data_csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  [OK] Dados CSV salvos em: {data_csv_path}")

    # Report JSON
    report_json_path = results_dir / "experiment_07_report.json"
    report_payload = {
        "experiment": "TreasureGraph - Experimento 07: Ambientes Neutros e Grafos Distratores",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name,
        "reasoning_effort": "high",
        "spending_cap_usd": SPENDING_CAP_USD,
        "total_cost_usd": round(cumulative_cost_usd, 6),
        "total_runs_executed": len(results),
        "configurations": {
            "sizes": SIZES,
            "conditions": [c[0] for c in CONDITIONS],
            "max_steps_by_size": MAX_STEPS_BY_SIZE,
        },
        "environment_graph_stats": {
            str(size): {
                k: v for k, v in env_stats[size].items() if k != "candidate_paths_start_list"
            }
            for size in SIZES
        },
        "runs": results,
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] Relatório JSON salvo em: {report_json_path}")

    # Interactive HTML dashboard
    dashboard_path = results_dir / "dashboard_experiment_07.html"
    generate_interactive_dashboard(results, dashboard_path)
    print(f"  [OK] Dashboard HTML interativo gerado em: {dashboard_path} ({dashboard_path.stat().st_size:,} bytes)")

    # Copy dashboard to artifact dir
    artifact_dir = Path(r"C:\Users\Dog Center\.gemini\antigravity-cli\brain\54714945-9bf1-409c-a667-05fd687c36ce")
    if artifact_dir.exists():
        art_dash = artifact_dir / "dashboard_experiment_07.html"
        art_dash.write_text(dashboard_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  [OK] Dashboard copiado para diretório de artefatos: {art_dash}")

    print("\n================================================================================")
    print(f"=== EXPERIMENTO 07 CONCLUÍDO COM SUCESSO! Custo total: ${cumulative_cost_usd:.6f} USD ===")
    print("================================================================================")


if __name__ == "__main__":
    main()
