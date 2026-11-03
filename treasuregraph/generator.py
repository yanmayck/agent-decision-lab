"""TreasureGraph environment generator.

Generates reproducible environments with interconnected clues leading to a
secret treasure code, designed for testing agent planning and decision making.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

FOLDER_POOLS = [
    ("archive", "the repository of records, memos, and historical documentation"),
    ("laboratory", "the testing division with experimental logs and technical data"),
    ("vault", "the high-security sector housing terminal controls and restricted vaults"),
    ("telemetry", "the communications sector monitoring signals and sensors"),
    ("operations", "the tactical management office handling facility logistics"),
]

FILE_NAMES = {
    "step1": ["briefing.txt", "mission_start.txt", "dispatch.txt"],
    "step2": ["research_notes.txt", "field_log.txt", "signal_analysis.txt"],
    "step3": ["vault_terminal.txt", "final_chamber.txt", "safe_deposit.txt"],
    "decoy": ["security_advisory.txt", "intercepted_memo.txt", "alert_bulletin.txt"],
    "noise": ["maintenance_log.txt", "inventory_manifest.txt", "system_diagnostics.txt"],
}

CODE_WORDS = [
    "PHOENIX", "ORION", "NEBULA", "AURORA", "VORTEX",
    "CHRONOS", "VALKYRIE", "ECLIPSE", "HORIZON", "ZENITH"
]


def generate_world(
    seed: int,
    output_dir: str | Path,
    answer_key_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generates a reproducible world with 5 .txt files across 3 folders.

    Requirements met:
    - 5 .txt files distributed across 3 folders.
    - Chain of 3 clues leading to a secret code.
    - Mix of direct and semantic clues.
    - 2 irrelevant files (1 logically solvable decoy, 1 noise).
    - Fully reproducible using random.Random(seed).
    - Solution path and secret code saved to answer_key.json OUTSIDE output_dir.
    - No external dependencies or AI APIs.

    Args:
        seed: Integer seed for reproducible random generation.
        output_dir: Directory where the agent world will be generated.
        answer_key_path: Optional path for answer_key.json. If None, saved in parent of output_dir.

    Returns:
        dict containing the generated answer key metadata.
    """
    rng = random.Random(seed)
    base_dir = Path(output_dir).resolve()

    # Determine answer_key.json destination (must be outside output_dir)
    if answer_key_path is not None:
        key_file = Path(answer_key_path).resolve()
    else:
        key_file = base_dir.parent / "answer_key.json"

    if key_file == base_dir or base_dir in key_file.parents:
        raise ValueError(
            f"answer_key_path ({key_file}) must be outside the agent accessible directory ({base_dir})."
        )

    # Clean / prepare base directory
    base_dir.mkdir(parents=True, exist_ok=True)
    key_file.parent.mkdir(parents=True, exist_ok=True)

    # Select 3 distinct folders
    selected_folders = rng.sample(FOLDER_POOLS, 3)
    folder_names = [f[0] for f in selected_folders]
    folder_descs = {f[0]: f[1] for f in selected_folders}

    folder_paths = {}
    for name in folder_names:
        f_path = base_dir / name
        f_path.mkdir(parents=True, exist_ok=True)
        folder_paths[name] = f_path

    # Assign filenames from pools
    name_step1 = rng.choice(FILE_NAMES["step1"])
    name_step2 = rng.choice(FILE_NAMES["step2"])
    name_step3 = rng.choice(FILE_NAMES["step3"])
    name_decoy = rng.choice(FILE_NAMES["decoy"])
    name_noise = rng.choice(FILE_NAMES["noise"])

    # Distribution across 3 folders:
    # Folder 0 gets Step 1 and Decoy (2 files)
    # Folder 1 gets Step 2 and Noise (2 files)
    # Folder 2 gets Step 3 (1 file)
    # Total = 5 files distributed across 3 folders (2 + 2 + 1)
    f_step1 = folder_names[0]
    f_step2 = folder_names[1]
    f_step3 = folder_names[2]

    # Distribute decoy and noise into folders ensuring 2 folders have 2 files, 1 has 1 file
    f_decoy = folder_names[0]
    f_noise = folder_names[1]

    path_step1 = f"{f_step1}/{name_step1}"
    path_step2 = f"{f_step2}/{name_step2}"
    path_step3 = f"{f_step3}/{name_step3}"
    path_decoy = f"{f_decoy}/{name_decoy}"
    path_noise = f"{f_noise}/{name_noise}"

    # Secret code generation
    secret_code = f"TREASURE-{rng.randint(1000, 9999)}-{rng.choice(CODE_WORDS)}"

    # Determine clue styles (one direct, one semantic to mix styles)
    step1_is_semantic = rng.choice([True, False])

    # Build Step 1 Clue Content
    if step1_is_semantic:
        step1_clue_type = "semantic"
        step1_hint = (
            f"Next clue (Semantic): Proceed to {folder_descs[f_step2]} (folder: '{f_step2}') "
            f"and examine the document titled '{name_step2}'."
        )
    else:
        step1_clue_type = "direct"
        step1_hint = f"Next clue (Direct): Open file '{path_step2}'."

    step1_content = (
        f"=== AGENT BRIEFING: MISSION INITIATION ===\n"
        f"Welcome, operative. A classified treasure has been secured in this sector.\n"
        f"To recover the secret code, you must strictly follow the authenticated chain.\n\n"
        f"{step1_hint}\n"
    )

    # Build Step 2 Clue Content
    if step1_is_semantic:
        step2_clue_type = "direct"
        step2_hint = (
            f"Next clue (Direct): Move directly to '{path_step3}' where the final vault terminal resides."
        )
    else:
        step2_clue_type = "semantic"
        step2_hint = (
            f"Next clue (Semantic): Access {folder_descs[f_step3]} (folder: '{f_step3}') "
            f"and inspect the terminal file '{name_step3}'."
        )

    step2_content = (
        f"=== CHECKPOINT ANALYSIS: STAGE 2 ===\n"
        f"Authentication confirmed from Step 1.\n"
        f"You are approaching the inner perimeter.\n\n"
        f"{step2_hint}\n"
    )

    # Build Step 3 (Treasure) Content
    step3_content = (
        f"=== SECURE VAULT TERMINAL: FINAL STAGE ===\n"
        f"Access granted. Chain verification complete.\n\n"
        f"SECRET CODE: {secret_code}\n\n"
        f"Record this secret code to conclude the mission.\n"
    )

    # Build Decoy (Misleading clue, but logically solvable)
    fake_code = f"DECOY-{rng.randint(1000, 9999)}-INVALID"
    decoy_content = (
        f"=== INTERCEPTED ADVISORY (CONFIDENTIAL) ===\n"
        f"Notice: Agent report claims the secret code is: {fake_code}\n\n"
        f"[LOGICAL VERIFICATION ADVISORY]\n"
        f"- Status: REVOKED / UNVERIFIED.\n"
        f"- Reason: This file does not belong to the authenticated briefing sequence.\n"
        f"- Rule 4.1: Genuine treasure codes follow the authenticated 3-step chain starting at '{path_step1}'.\n"
        f"- Any code from unverified advisories is a simulated decoy trap and must be discarded.\n"
    )

    # Build Noise (Irrelevant content)
    noise_content = (
        f"=== ROUTINE FACILITY MAINTENANCE LOG ===\n"
        f"Inspector ID: TECH-{rng.randint(10, 99)}\n"
        f"Status: Normal Operations\n"
        f"- Atmospheric pressure: 101.3 kPa (Nominal)\n"
        f"- Air recycling unit: Filters replaced in Section {rng.randint(1, 5)}\n"
        f"- Coffee dispenser: Cleaned and restocked\n"
        f"No operational or mission-related data in this log.\n"
    )

    # Write files to disk
    files_to_write = [
        (base_dir / path_step1, step1_content),
        (base_dir / path_step2, step2_content),
        (base_dir / path_step3, step3_content),
        (base_dir / path_decoy, decoy_content),
        (base_dir / path_noise, noise_content),
    ]

    for file_path, content in files_to_write:
        file_path.write_text(content, encoding="utf-8")

    # Build answer key
    answer_key = {
        "seed": seed,
        "secret_code": secret_code,
        "solution_path": [path_step1, path_step2, path_step3],
        "start_file": path_step1,
        "folders": folder_names,
        "clue_chain": [
            {
                "step": 1,
                "file": path_step1,
                "type": step1_clue_type,
                "leads_to": path_step2,
            },
            {
                "step": 2,
                "file": path_step2,
                "type": step2_clue_type,
                "leads_to": path_step3,
            },
            {
                "step": 3,
                "file": path_step3,
                "type": "terminal",
                "secret_code": secret_code,
            },
        ],
        "distractors": [
            {
                "file": path_decoy,
                "type": "decoy",
                "flaw": "Marked REVOKED/UNVERIFIED; breaks Rule 4.1 of the authenticated chain.",
                "fake_code": fake_code,
            },
            {
                "file": path_noise,
                "type": "noise",
                "description": "Routine facility maintenance log without actionable clues.",
            },
        ],
    }

    # Save answer_key.json
    key_file.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")

    return answer_key
