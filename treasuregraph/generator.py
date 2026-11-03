"""TreasureGraph environment generator.

Generates reproducible environments with interconnected clues leading to a
secret treasure code, designed for testing agent planning and decision making.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

# Semantic metadata for sectors (folders).
# The descriptor is used in semantic clues and NEVER reveals the folder name.
SECTOR_METADATA = {
    "archive": {
        "descriptor": "the historical records repository and document preservation facility",
        "semantic_label": "Historical Documentation & Records",
    },
    "laboratory": {
        "descriptor": "the scientific testing division and experimental workshop",
        "semantic_label": "Scientific Testing & Experimentation",
    },
    "vault": {
        "descriptor": "the maximum-security containment sector and lockdown zone",
        "semantic_label": "Maximum-Security Containment",
    },
    "telemetry": {
        "descriptor": "the signal tracking station and communications outpost",
        "semantic_label": "Signal Tracking & Communications",
    },
    "operations": {
        "descriptor": "the tactical management office and logistics coordination center",
        "semantic_label": "Tactical Logistics & Management",
    },
}

STEP1_TEMPLATES = [
    ("briefing.txt", "mission initiation parameters and procedural objectives"),
    ("mission_start.txt", "dispatch briefing and primary procedural directives"),
    ("entry_dossier.txt", "entry clearance and starting mission profile"),
]

STEP2_TEMPLATES = [
    (
        "checkpoint_record.txt",
        "mid-stage authentication data and path coordinates",
        "Checkpoint Verification & Navigation",
    ),
    (
        "status_report.txt",
        "intermediate phase assessment and corridor clearance",
        "Intermediate Phase Clearance",
    ),
    (
        "field_dossier.txt",
        "intermediate survey observations and sensor readings",
        "Intermediate Route Survey",
    ),
]

STEP3_TEMPLATES = [
    (
        "terminal_access.txt",
        "restricted mainframe console interface and emergency override",
        "Restricted Mainframe Console",
    ),
    (
        "central_core.txt",
        "primary data storage bank and final master credentials",
        "Primary Data Storage Bank",
    ),
    (
        "master_ledger.txt",
        "sealed authorization node and final authentication payload",
        "Sealed Authorization Node",
    ),
]

DECOY_TEMPLATES = [
    ("security_advisory.txt", "Unverified Security Advisory"),
    ("intercepted_memo.txt", "Intercepted Signal Memo"),
    ("clearance_bulletin.txt", "Revoked Clearance Bulletin"),
]

NOISE_TEMPLATES = [
    (
        "maintenance_log.txt",
        "routine environmental maintenance and HVAC servicing",
        "Facility Environmental Maintenance",
    ),
    (
        "inventory_manifest.txt",
        "cafeteria provisions and janitorial supply count",
        "Provisions & Sanitation Inventory",
    ),
    (
        "system_diagnostics.txt",
        "ambient sensor pressure and power grid diagnostics",
        "Ambient Sensor & Power Diagnostics",
    ),
]

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
    1. Semantic clues DO NOT reveal exact file or folder names.
    2. Agents relate semantic descriptions to the purpose/content of files.
    3. Direct clues and irrelevant files (decoy + noise) are preserved.
    4. Each environment has an unequivocal, unambiguous solution path.
    5. Fully reproducible using random.Random(seed).
    6. Solution path and secret code saved to answer_key.json OUTSIDE output_dir.
    7. No external dependencies or AI APIs.

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

    # Clean existing generated files in subdirectories if regenerating in the same folder
    for existing_file in base_dir.glob("*/*.txt"):
        existing_file.unlink(missing_ok=True)

    # Select 3 distinct folders from pool
    available_folder_keys = list(SECTOR_METADATA.keys())
    folder_names = rng.sample(available_folder_keys, 3)

    for name in folder_names:
        (base_dir / name).mkdir(parents=True, exist_ok=True)

    f_step1 = folder_names[0]
    f_step2 = folder_names[1]
    f_step3 = folder_names[2]

    # Distribute 5 files across 3 folders: [2, 2, 1]
    # Folder 0: Step 1 + Decoy (2)
    # Folder 1: Step 2 + Noise (2)
    # Folder 2: Step 3 (1)
    f_decoy = folder_names[0]
    f_noise = folder_names[1]

    # Select file templates
    step1_name, step1_desc = rng.choice(STEP1_TEMPLATES)
    step2_name, step2_desc, step2_label = rng.choice(STEP2_TEMPLATES)
    step3_name, step3_desc, step3_label = rng.choice(STEP3_TEMPLATES)
    decoy_name, decoy_label = rng.choice(DECOY_TEMPLATES)
    noise_name, noise_desc, noise_label = rng.choice(NOISE_TEMPLATES)

    path_step1 = f"{f_step1}/{step1_name}"
    path_step2 = f"{f_step2}/{step2_name}"
    path_step3 = f"{f_step3}/{step3_name}"
    path_decoy = f"{f_decoy}/{decoy_name}"
    path_noise = f"{f_noise}/{noise_name}"

    # Secret code generation
    secret_code = f"TREASURE-{rng.randint(1000, 9999)}-{rng.choice(CODE_WORDS)}"

    # Decide clue styles: one semantic, one direct to mix styles
    step1_is_semantic = rng.choice([True, False])

    # Build Step 1 Clue:
    # If semantic: describes the sector purpose and document purpose WITHOUT exact names.
    if step1_is_semantic:
        step1_clue_type = "semantic"
        step1_target_sector = SECTOR_METADATA[f_step2]["descriptor"]
        step1_target_purpose = step2_desc
        step1_hint = (
            f"Next clue (Semantic): Proceed to {step1_target_sector} "
            f"and examine the document detailing {step1_target_purpose}."
        )
    else:
        step1_clue_type = "direct"
        step1_target_sector = None
        step1_target_purpose = None
        step1_hint = f"Next clue (Direct): Open file '{path_step2}'."

    step1_content = (
        f"=== AGENT BRIEFING: MISSION INITIATION ===\n"
        f"Sector: {SECTOR_METADATA[f_step1]['semantic_label']}\n"
        f"Subject: {step1_desc}\n\n"
        f"Welcome, operative. A classified treasure has been secured in this facility.\n"
        f"To recover the secret code, you must strictly follow the authenticated chain.\n\n"
        f"{step1_hint}\n"
    )

    # Build Step 2 Clue:
    # If Step 1 was semantic, Step 2 is direct; if Step 1 was direct, Step 2 is semantic.
    if step1_is_semantic:
        step2_clue_type = "direct"
        step2_target_sector = None
        step2_target_purpose = None
        step2_hint = (
            f"Next clue (Direct): Move directly to '{path_step3}' where the final vault terminal resides."
        )
    else:
        step2_clue_type = "semantic"
        step2_target_sector = SECTOR_METADATA[f_step3]["descriptor"]
        step2_target_purpose = step3_desc
        step2_hint = (
            f"Next clue (Semantic): Access {step2_target_sector} "
            f"and inspect the console record storing {step2_target_purpose}."
        )

    step2_content = (
        f"=== CHECKPOINT ANALYSIS: STAGE 2 ===\n"
        f"Sector: {SECTOR_METADATA[f_step2]['semantic_label']}\n"
        f"Subject: {step2_label} - {step2_desc}\n\n"
        f"Authentication confirmed from Step 1.\n"
        f"You are approaching the inner perimeter.\n\n"
        f"{step2_hint}\n"
    )

    # Build Step 3 (Treasure) Content
    step3_content = (
        f"=== SECURE VAULT TERMINAL: FINAL STAGE ===\n"
        f"Sector: {SECTOR_METADATA[f_step3]['semantic_label']}\n"
        f"Subject: {step3_label} - {step3_desc}\n\n"
        f"Access granted. Chain verification complete.\n\n"
        f"SECRET CODE: {secret_code}\n\n"
        f"Record this secret code to conclude the mission.\n"
    )

    # Build Decoy (Misleading clue, but logically solvable)
    fake_code = f"DECOY-{rng.randint(1000, 9999)}-INVALID"
    decoy_content = (
        f"=== INTERCEPTED ADVISORY (CONFIDENTIAL) ===\n"
        f"Sector: Unverified Transmission Channel\n"
        f"Subject: {decoy_label}\n\n"
        f"Notice: Agent report claims the secret code is: {fake_code}\n\n"
        f"[LOGICAL VERIFICATION ADVISORY]\n"
        f"- Status: REVOKED / UNVERIFIED.\n"
        f"- Reason: This advisory does not belong to the authenticated briefing sequence.\n"
        f"- Rule 4.1: Genuine treasure codes follow the authenticated 3-step chain starting at '{path_step1}'.\n"
        f"- Any code from unverified advisories is a simulated decoy trap and must be discarded.\n"
    )

    # Build Noise (Irrelevant content)
    noise_content = (
        f"=== ROUTINE FACILITY MAINTENANCE LOG ===\n"
        f"Sector: Facility Infrastructure Support\n"
        f"Subject: {noise_label} - {noise_desc}\n"
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
                "semantic_sector": step1_target_sector,
                "semantic_purpose": step1_target_purpose,
            },
            {
                "step": 2,
                "file": path_step2,
                "type": step2_clue_type,
                "leads_to": path_step3,
                "semantic_sector": step2_target_sector,
                "semantic_purpose": step2_target_purpose,
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
                "description": noise_desc,
            },
        ],
    }

    # Save answer_key.json
    key_file.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")

    return answer_key
