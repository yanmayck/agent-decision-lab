"""Runner script for Experiment 01 (Baseline Agent on world_pt_001)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from treasuregraph.agent import BaselineAgent


def main():
    world_dir = Path("experiments/world_pt_001")
    answer_key_path = Path("experiments/answer_key.json")
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_result_file = results_dir / "baseline_seed42.json"

    # Pre-flight environment check
    model_name = os.getenv("TREASURE_MODEL") or os.getenv("OPENAI_MODEL")
    api_key = (
        os.getenv("TREASURE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    api_base = os.getenv("TREASURE_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    if not model_name or not api_key:
        print("ERROR: API configuration missing.")
        if not model_name:
            print(" - TREASURE_MODEL is not set.")
        if not api_key:
            print(" - TREASURE_API_KEY (or OPENAI_API_KEY / GEMINI_API_KEY) is not set.")
        sys.exit(1)

    print(f"Starting Baseline Agent Experiment 01...")
    print(f"Environment: {world_dir}")
    print(f"Model: {model_name}")
    print(f"Action limit: 15")

    # Run agent - completely sandboxed in world_dir
    agent = BaselineAgent(
        world_dir=world_dir,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        max_steps=15,
    )

    agent_result = agent.run()

    # Save complete trajectory
    output_result_file.write_text(
        json.dumps(agent_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Trajectory saved to {output_result_file}")

    # EXTERNAL EVALUATION (only after agent finished)
    if not answer_key_path.exists():
        print("Warning: answer_key.json not found for external evaluation.")
        return

    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    correct_code = answer_key.get("secret_code")
    solution_path = answer_key.get("solution_path", [])
    submitted_code = agent_result.get("submitted_code")

    is_correct = submitted_code == correct_code if submitted_code else False

    # Action analysis
    trajectory = agent_result.get("trajectory", [])
    actions_sequence = [f"{step['step']}. {step['action']}({step['arguments']})" for step in trajectory]
    files_consulted = [
        step["arguments"].get("path")
        for step in trajectory
        if step["action"] == "read_file" and "path" in step["arguments"]
    ]
    directories_listed = [
        step["arguments"].get("path", ".")
        for step in trajectory
        if step["action"] == "list_directory"
    ]

    # Repeated / redundant actions
    read_counts = {}
    for f in files_consulted:
        read_counts[f] = read_counts.get(f, 0) + 1
    repeated_reads = {f: c for f, c in read_counts.items() if c > 1}

    list_counts = {}
    for d in directories_listed:
        list_counts[d] = list_counts.get(d, 0) + 1
    repeated_lists = {d: c for d, c in list_counts.items() if c > 1}

    unnecessary_files = [f for f in files_consulted if f not in solution_path]

    errors = [step for step in trajectory if step["status"] == "error"]

    report = {
        "evaluation": {
            "is_correct": is_correct,
            "submitted_code": submitted_code,
            "expected_code": correct_code,
        },
        "actions_summary": {
            "total_actions": len(trajectory),
            "sequence": actions_sequence,
        },
        "files_consulted": files_consulted,
        "efficiency_analysis": {
            "repeated_reads": repeated_reads,
            "repeated_lists": repeated_lists,
            "unnecessary_files_read": unnecessary_files,
        },
        "execution_metrics": {
            "status": agent_result.get("status"),
            "total_duration_seconds": agent_result.get("total_duration_seconds"),
            "total_tokens": agent_result.get("total_tokens"),
            "errors_count": len(errors),
            "errors": errors,
        },
    }

    # Save evaluation report
    eval_file = results_dir / "baseline_seed42_report.json"
    eval_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n--- EXTERNAL EVALUATION REPORT ---")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
