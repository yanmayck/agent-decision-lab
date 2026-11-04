"""TreasureGraph - Experiment 08: Decision Theory and Expected Cost Optimization.

Investigates whether LLM agents select actions that minimize expected cost under uncertainty
using classical Decision Theory formulations from `treasuregraph/decision_theory.py`.

Requirements:
- Evaluates 20 deterministic scenarios with varying probabilities and action costs.
- Presents two options to the LLM:
  Option A: Open candidate location first; consult perfect report only upon failure.
  Option B: Consult perfect report first; then open confirmed location.
- Independent mathematical evaluator computes optimal strategy and regret:
  regret = expected_cost(model_choice) - optimal_expected_cost
- Model: Gemini (gemini-3.5-flash-lite) with reasoning HIGH (reasoning_effort="high").
- Preserves separation between scenarios and model responses.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from dotenv import load_dotenv

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from treasuregraph.decision_theory import (
    expected_cost_open_first,
    expected_cost_report_first,
)
from treasuregraph.instrumentation import (
    PricingConfig,
    parse_api_usage,
)

DEFAULT_SCENARIOS_PATH = REPO_ROOT / "experiments" / "scenarios_experiment_08.json"
DEFAULT_RESULTS_PATH = REPO_ROOT / "experiments" / "results" / "experiment_08_results.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "experiments" / "results" / "experiment_08_report.json"

GEMINI_OPENAI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai"

SYSTEM_PROMPT = (
    "Você é um especialista em Teoria da Decisão, Pesquisa Operacional e Otimização sob Incerteza. "
    "Seu objetivo estrito é escolher a alternativa que MINIMIZA o custo financeiro esperado total (Expected Cost)."
)


def build_scenario_prompt(scenario: dict[str, Any]) -> str:
    """Builds the neutral, non-leaking decision prompt for a given scenario.

    Explicitly specifies the probability and action costs, describes Options A and B,
    and instructs the model to choose solely between A and B to minimize expected cost.
    Does NOT reveal expected costs, thresholds, or optimal choices.
    """
    p = float(scenario["probability"])
    c_open = float(scenario["open_cost"])
    c_report = float(scenario["report_cost"])
    name = scenario.get("name", f"Cenário {scenario.get('scenario_id', '')}")
    description = scenario.get("description", "")

    p_pct = round(p * 100, 2)
    p_fail_pct = round((1.0 - p) * 100, 2)

    prompt = f"""### Problema de Decisão: {name}
{description}

Você precisa obter acesso garantido ao conteúdo correto (alvo final), que está em um dos locais possíveis.
Você possui uma avaliação inicial sobre o local mais provável, e tem à sua disposição um serviço de relatório perfeito que identifica com 100% de precisão o local exato onde o alvo está.

**Parâmetros Financeiros e Probabilidades:**
- Probabilidade de o local mais provável conter o alvo: **{p_pct}%** (p = {p:.2f}).
- Consequentemente, probabilidade de o local mais provável estar vazio: **{p_fail_pct}%** (1 - p = {1.0 - p:.2f}).
- Custo unitário para abrir qualquer cofre/local: **C_open = {c_open:.2f}** unidades.
- Custo para consultar o relatório perfeito: **C_report = {c_report:.2f}** unidades.

**Opções Estratégicas Disponíveis:**

- **Opção A (Abrir Primeiro):**
  Você abre imediatamente o local mais provável.
  * Se o alvo estiver presente (probabilidade p), a operação é concluída com sucesso. Custo incorrido: C_open.
  * Se o local estiver vazio (probabilidade 1 - p), você é obrigado a consultar o relatório perfeito e depois abrir o local correto indicado. Custo incorrido neste caso: C_open (primeira tentativa) + C_report (relatório) + C_open (abertura final) = 2 * C_open + C_report.

- **Opção B (Consultar Relatório Primeiro):**
  Você contrata primeiro o relatório perfeito e, em seguida, abre diretamente o local indicado com certeza absoluta.
  * O relatório é consultado com certeza e o cofre correto é aberto. Custo total incorrido: C_report + C_open.

**Sua Tarefa:**
Determine qual das duas estratégias (**A** ou **B**) apresenta o **MENOR CUSTO ESPERADO TOTAL** para estes parâmetros.

Apresente uma justificativa matemática suscinta do seu cálculo de valor esperado e encerre OBRIGATORIAMENTE sua resposta na última linha com o seguinte formato:
DECISÃO: [A ou B]
"""
    return prompt.strip()


def parse_model_choice(response_text: str | None) -> str | None:
    """Extracts the model's decision ('A' or 'B') from response text.

    Uses robust regex heuristics to handle varied formatting while preventing
    false positives with Portuguese definite articles ('a', 'o').
    Returns 'A', 'B', or None if ambiguous/unparseable.
    """
    if not response_text or not isinstance(response_text, str):
        return None

    cleaned = response_text.strip()

    # Pattern 1: Explicit decision label (highest precedence)
    # e.g.: "DECISÃO: A", "DECISAO: B", "DECISION: **A**", "ESCOLHA: B", "OPÇÃO: B"
    m_tag = re.search(
        r"(?:DECISÃO|DECISAO|DECISION|ESCOLHA|RESPOSTA|CONCLUSÃO|CONCLUSAO|OPÇÃO|OPCAO|OPTION)\s*:\s*[*_`\"']*([AB])\b",
        cleaned,
        re.IGNORECASE,
    )
    if m_tag:
        return m_tag.group(1).upper()

    # Pattern 2: Inspect lines in reverse order (bottom-up) for terminal decision statements
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        # Look for explicit option mention at the end of the line, e.g. "Opção B", "**B**", "DECISÃO: B"
        m_opt = re.search(
            r"(?:opção|opcao|option|estratégia|estrategia|alternativa)?\s*[*_`\"']*([AB])[*_`\"']*\.?$",
            line,
            re.IGNORECASE,
        )
        if m_opt:
            # Verify it's preceded by decision intent or stand-alone
            prefix = line[:m_opt.start()].strip()
            if not prefix or re.search(r"(?:recomendo|escolho|adoto|optamos|opto|decisão|decisao|portanto|concluo|logo|resposta)\b", prefix, re.IGNORECASE):
                return m_opt.group(1).upper()

    # Pattern 3: Explicit phrases like "recomendo a Opção A" or "estratégia ótima é a B"
    all_markers = re.findall(
        r"(?:recomendo|escolho|optamos pela|melhor opção é a|melhor estrategia é a|estratégia ótima é a|estrategia otima é a)\s+[*_`\"']*(?:opção\s+|opcao\s+)?([AB])\b",
        cleaned,
        re.IGNORECASE,
    )
    if all_markers:
        return all_markers[-1].upper()

    return None


def evaluate_decision(
    scenario: dict[str, Any],
    model_choice: str | None,
) -> dict[str, Any]:
    """Independent mathematical evaluation using decision_theory.py.

    Calculates:
    - expected_cost_open_first
    - expected_cost_report_first
    - optimal_choice
    - optimal_expected_cost
    - choice_expected_cost
    - regret = choice_expected_cost - optimal_expected_cost
    - is_optimal (boolean)
    """
    p = float(scenario["probability"])
    c_open = float(scenario["open_cost"])
    c_report = float(scenario["report_cost"])

    ec_a = expected_cost_open_first(p, c_open, c_report)
    ec_b = expected_cost_report_first(c_open, c_report)

    # Indifference threshold: p* = c_open / (c_open + c_report)
    threshold_p = round(c_open / (c_open + c_report), 6)

    # Allow tiny float tolerance (1e-9)
    if abs(ec_a - ec_b) < 1e-9:
        optimal_choice = "INDIFF"
        optimal_cost = ec_a
    elif ec_a < ec_b:
        optimal_choice = "A"
        optimal_cost = ec_a
    else:
        optimal_choice = "B"
        optimal_cost = ec_b

    if model_choice == "A":
        choice_cost = ec_a
        regret = round(choice_cost - optimal_cost, 6)
        is_optimal = (optimal_choice in ("A", "INDIFF"))
    elif model_choice == "B":
        choice_cost = ec_b
        regret = round(choice_cost - optimal_cost, 6)
        is_optimal = (optimal_choice in ("B", "INDIFF"))
    else:
        choice_cost = None
        regret = None
        is_optimal = False

    return {
        "expected_cost_a": ec_a,
        "expected_cost_b": ec_b,
        "threshold_probability": threshold_p,
        "optimal_choice": optimal_choice,
        "optimal_expected_cost": optimal_cost,
        "model_choice": model_choice,
        "choice_expected_cost": choice_cost,
        "regret": regret,
        "is_optimal": is_optimal,
    }


def call_llm(
    prompt: str,
    model: str,
    api_key: str,
    api_base: str,
    reasoning_effort: str = "high",
    pricing_config: PricingConfig | None = None,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Sends prompt to OpenAI-compatible LLM endpoint and records usage."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "api-key": api_key,
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    t0 = time.perf_counter()
    resp_data = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if e.code in (503, 429) and attempt < max_retries - 1:
                sleep_sec = 3 * (attempt + 1)
                time.sleep(sleep_sec)
                continue
            raise RuntimeError(f"LLM API request failed ({e.code}): {err_body}")
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt < max_retries - 1:
                sleep_sec = 3 * (attempt + 1)
                time.sleep(sleep_sec)
                continue
            raise RuntimeError(f"LLM API connection error: {e}")

    latency_sec = round(time.perf_counter() - t0, 4)

    if not resp_data:
        raise RuntimeError("Empty response received from LLM API.")

    choice = resp_data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")

    # Usage parsing
    usage_raw = resp_data.get("usage", {})
    usage_parsed = parse_api_usage(usage_raw)

    cost_usd = None
    if pricing_config is not None:
        cost_usd = pricing_config.calculate_cost(
            input_tokens=usage_parsed["prompt_tokens"],
            output_tokens=usage_parsed["completion_tokens"],
            reasoning_tokens=usage_parsed["reasoning_tokens"],
            reasoning_included_in_output=usage_parsed["reasoning_included_in_output"],
        )

    return {
        "content": content,
        "usage": usage_parsed,
        "estimated_cost_usd": cost_usd,
        "latency_seconds": latency_sec,
    }


def run_experiment_08(
    scenarios_path: Path = DEFAULT_SCENARIOS_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    model_override: str | None = None,
    api_key_override: str | None = None,
    api_base_override: str | None = None,
    reasoning_effort: str = "high",
    llm_callable: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Executes Experiment 08 across all 20 scenarios."""
    load_dotenv()

    scenarios_file = Path(scenarios_path)
    if not scenarios_file.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenarios_file}")

    scenarios: list[dict[str, Any]] = json.loads(scenarios_file.read_text(encoding="utf-8"))
    if len(scenarios) != 20:
        raise ValueError(f"Expected exactly 20 scenarios, found {len(scenarios)}")

    pricing = PricingConfig(
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        reasoning_cost_per_million=0.30,
    )

    model_name = (
        model_override
        or os.getenv("TREASURE_MODEL")
        or os.getenv("GEMINI_MODEL")
        or "gemini-3.5-flash-lite"
    )

    api_key = (
        api_key_override
        or os.getenv("TREASURE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )

    if api_base_override:
        api_base = api_base_override
    elif os.getenv("TREASURE_API_BASE"):
        api_base = os.environ["TREASURE_API_BASE"]
    elif (
        model_name.lower().startswith("gemini")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    ):
        api_base = GEMINI_OPENAI_ENDPOINT
    else:
        api_base = "https://api.openai.com/v1"

    if llm_callable is None and not api_key:
        raise ValueError("API key not configured and no llm_callable provided.")

    print("================================================================================")
    print("=== AGENT DECISION LAB - EXPERIMENTO 08: TEORIA DA DECISÃO & EXPECTED COST ===")
    print("================================================================================")
    print(f"Modelo: {model_name} | Reasoning: {reasoning_effort.upper()} | Cenários: {len(scenarios)}")
    print(f"Endpoint: {api_base}")
    print("--------------------------------------------------------------------------------")

    results_dir = results_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    total_cost_usd = 0.0
    total_latency_sec = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0

    for idx, sc in enumerate(scenarios, 1):
        sid = sc["scenario_id"]
        p = sc["probability"]
        c_o = sc["open_cost"]
        c_r = sc["report_cost"]
        name = sc.get("name", f"Cenário {sid}")

        prompt = build_scenario_prompt(sc)

        print(f"\n[Cenário {sid:02d}/20] p={p:.2f} | C_open={c_o:4.1f} | C_report={c_r:4.1f} | {name}")

        if llm_callable is not None:
            llm_result = llm_callable(prompt)
        else:
            llm_result = call_llm(
                prompt=prompt,
                model=model_name,
                api_key=api_key,
                api_base=api_base,
                reasoning_effort=reasoning_effort,
                pricing_config=pricing,
            )

        content = llm_result.get("content", "")
        usage = llm_result.get("usage", {})
        cost = llm_result.get("estimated_cost_usd")
        latency = llm_result.get("latency_seconds", 0.0)

        if cost is not None:
            total_cost_usd += cost
        total_latency_sec += latency
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        if usage.get("reasoning_tokens"):
            total_reasoning_tokens += usage["reasoning_tokens"]

        # Parse choice and evaluate independently
        model_choice = parse_model_choice(content)
        eval_metrics = evaluate_decision(sc, model_choice)

        opt = eval_metrics["optimal_choice"]
        is_opt = eval_metrics["is_optimal"]
        regret = eval_metrics["regret"]
        ec_a = eval_metrics["expected_cost_a"]
        ec_b = eval_metrics["expected_cost_b"]

        opt_symbol = "[OK] ÓTIMO" if is_opt else "[SUB] SUB-ÓTIMO"
        print(f"  -> Escolha do Modelo: '{model_choice}' | Ótimo Teórico: '{opt}' {opt_symbol}")
        print(f"  -> EC(A)={ec_a:.2f} | EC(B)={ec_b:.2f} | Regret: {regret if regret is not None else 'N/A'}")
        print(f"  -> Latência: {latency:.2f}s | Tokens: in={usage.get('prompt_tokens', 0)}, out={usage.get('completion_tokens', 0)}, rzn={usage.get('reasoning_tokens')}")

        record = {
            "scenario_id": sid,
            "name": name,
            "description": sc.get("description", ""),
            "probability": p,
            "open_cost": c_o,
            "report_cost": c_r,
            "threshold_probability": eval_metrics["threshold_probability"],
            "expected_cost_a": ec_a,
            "expected_cost_b": ec_b,
            "optimal_choice": opt,
            "optimal_expected_cost": eval_metrics["optimal_expected_cost"],
            "model_choice": model_choice,
            "choice_expected_cost": eval_metrics["choice_expected_cost"],
            "regret": regret,
            "is_optimal": is_opt,
            "model_response": content,
            "latency_seconds": latency,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "estimated_cost_usd": cost,
        }
        records.append(record)

    # Compute consolidated summary statistics
    optimal_count = sum(1 for r in records if r["is_optimal"])
    suboptimal_count = len(records) - optimal_count
    optimal_rate_pct = round((optimal_count / len(records)) * 100, 2)

    valid_regrets = [r["regret"] for r in records if r["regret"] is not None]
    mean_regret = round(sum(valid_regrets) / len(valid_regrets), 4) if valid_regrets else 0.0
    max_regret = round(max(valid_regrets), 4) if valid_regrets else 0.0

    # Categorize by optimal strategy
    records_opt_a = [r for r in records if r["optimal_choice"] == "A"]
    records_opt_b = [r for r in records if r["optimal_choice"] == "B"]

    acc_when_a_opt = (
        round((sum(1 for r in records_opt_a if r["is_optimal"]) / len(records_opt_a)) * 100, 2)
        if records_opt_a else 0.0
    )
    acc_when_b_opt = (
        round((sum(1 for r in records_opt_b if r["is_optimal"]) / len(records_opt_b)) * 100, 2)
        if records_opt_b else 0.0
    )

    summary = {
        "experiment": "Experiment 08 - Decision Theory & Expected Cost Optimization",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "reasoning_effort": reasoning_effort,
        "total_scenarios": len(records),
        "optimal_choices_count": optimal_count,
        "suboptimal_choices_count": suboptimal_count,
        "optimal_rate_pct": optimal_rate_pct,
        "mean_regret": mean_regret,
        "max_regret": max_regret,
        "accuracy_when_opt_a_pct": acc_when_a_opt,
        "accuracy_when_opt_b_pct": acc_when_b_opt,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_estimated_cost_usd": round(total_cost_usd, 6),
        "total_latency_seconds": round(total_latency_sec, 2),
    }

    # Save outputs
    results_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    report_payload = {
        "summary": summary,
        "scenarios": records,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================================================================================")
    print("=== RELATÓRIO CONSOLIDADO: EXPERIMENTO 08 ===")
    print("================================================================================")
    print(f"Taxa de Escolhas Ótimas: {optimal_rate_pct}% ({optimal_count}/{len(records)})")
    print(f"Regret Médio: {mean_regret:.4f} | Regret Máximo: {max_regret:.4f}")
    print(f"Acurácia quando Ótimo era A: {acc_when_a_opt}% ({sum(1 for r in records_opt_a if r['is_optimal'])}/{len(records_opt_a)})")
    print(f"Acurácia quando Ótimo era B: {acc_when_b_opt}% ({sum(1 for r in records_opt_b if r['is_optimal'])}/{len(records_opt_b)})")
    print(f"Custo Total Estimado: US$ {total_cost_usd:.4f} | Latência Total: {total_latency_sec:.2f}s")
    print(f"Resultados detalhados salvos em: {results_path}")
    print(f"Relatório analítico salvo em:    {report_path}")

    return report_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 08: Decision Theory Optimization")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--reasoning", type=str, default="high", help="Reasoning effort level")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH, help="Path to scenarios JSON")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Path to output results JSON")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Path to output report JSON")
    args = parser.parse_args()

    run_experiment_08(
        scenarios_path=args.scenarios,
        results_path=args.results,
        report_path=args.report,
        model_override=args.model,
        reasoning_effort=args.reasoning,
    )
