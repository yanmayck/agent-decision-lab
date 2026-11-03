"""TreasureGraph environment generator.

Generates reproducible environments with interconnected clues leading to a
secret treasure code, designed for testing agent planning and decision making.
Supports bilingual generation in English ('en') and Brazilian Portuguese ('pt-BR').
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

# Sector definitions (folders) with semantic descriptors that never leak folder names.
SECTORS = {
    "archive": {
        "en": {
            "folder": "archive",
            "descriptor": "the historical records repository and document preservation facility",
            "label": "Historical Documentation & Records",
        },
        "pt-BR": {
            "folder": "arquivo",
            "descriptor": "o repositório de registros históricos e instalação de preservação documental",
            "label": "Documentação Histórica e Registros",
        },
    },
    "laboratory": {
        "en": {
            "folder": "laboratory",
            "descriptor": "the scientific testing division and experimental workshop",
            "label": "Scientific Testing & Experimentation",
        },
        "pt-BR": {
            "folder": "laboratorio",
            "descriptor": "a divisão de testes científicos e oficina experimental",
            "label": "Testes Científicos e Experimentação",
        },
    },
    "vault": {
        "en": {
            "folder": "vault",
            "descriptor": "the maximum-security containment sector and lockdown zone",
            "label": "Maximum-Security Containment",
        },
        "pt-BR": {
            "folder": "cofre",
            "descriptor": "o setor de contenção de segurança máxima e zona de isolamento",
            "label": "Contenção de Segurança Máxima",
        },
    },
    "telemetry": {
        "en": {
            "folder": "telemetry",
            "descriptor": "the signal tracking station and communications outpost",
            "label": "Signal Tracking & Communications",
        },
        "pt-BR": {
            "folder": "telemetria",
            "descriptor": "a estação de rastreamento de sinais e posto de comunicações",
            "label": "Rastreamento de Sinais e Comunicações",
        },
    },
    "operations": {
        "en": {
            "folder": "operations",
            "descriptor": "the tactical management office and logistics coordination center",
            "label": "Tactical Logistics & Management",
        },
        "pt-BR": {
            "folder": "operacoes",
            "descriptor": "o escritório de gestão tática e centro de coordenação logística",
            "label": "Gestão Tática e Logística",
        },
    },
}

STEP1_TEMPLATES = [
    {
        "en": {
            "filename": "briefing.txt",
            "desc": "mission initiation parameters and procedural objectives",
        },
        "pt-BR": {
            "filename": "briefing.txt",
            "desc": "parâmetros de iniciação da missão e objetivos procedimentais",
        },
    },
    {
        "en": {
            "filename": "mission_start.txt",
            "desc": "dispatch briefing and primary procedural directives",
        },
        "pt-BR": {
            "filename": "inicio_missao.txt",
            "desc": "briefing de despacho e diretrizes procedimentais primárias",
        },
    },
    {
        "en": {
            "filename": "entry_dossier.txt",
            "desc": "entry clearance and starting mission profile",
        },
        "pt-BR": {
            "filename": "dossie_entrada.txt",
            "desc": "autorização de entrada e perfil inicial da missão",
        },
    },
]

STEP2_TEMPLATES = [
    {
        "en": {
            "filename": "checkpoint_record.txt",
            "desc": "mid-stage authentication data and path coordinates",
            "label": "Checkpoint Verification & Navigation",
        },
        "pt-BR": {
            "filename": "registro_checkpoint.txt",
            "desc": "dados de autenticação intermediária e coordenadas de navegação",
            "label": "Verificação de Checkpoint e Navegação",
        },
    },
    {
        "en": {
            "filename": "status_report.txt",
            "desc": "intermediate phase assessment and corridor clearance",
            "label": "Intermediate Phase Clearance",
        },
        "pt-BR": {
            "filename": "relatorio_status.txt",
            "desc": "avaliação de fase intermediária e liberação de corredor",
            "label": "Liberação de Fase Intermediária",
        },
    },
    {
        "en": {
            "filename": "field_dossier.txt",
            "desc": "intermediate survey observations and sensor readings",
            "label": "Intermediate Route Survey",
        },
        "pt-BR": {
            "filename": "dossie_campo.txt",
            "desc": "observações de levantamento intermediário e leituras de sensores",
            "label": "Levantamento de Rota Intermediária",
        },
    },
]

STEP3_TEMPLATES = [
    {
        "en": {
            "filename": "terminal_access.txt",
            "desc": "restricted mainframe console interface and emergency override",
            "label": "Restricted Mainframe Console",
        },
        "pt-BR": {
            "filename": "acesso_terminal.txt",
            "desc": "interface de console do mainframe restrito e sobreposição de emergência",
            "label": "Console de Mainframe Restrito",
        },
    },
    {
        "en": {
            "filename": "central_core.txt",
            "desc": "primary data storage bank and final master credentials",
            "label": "Primary Data Storage Bank",
        },
        "pt-BR": {
            "filename": "nucleo_central.txt",
            "desc": "banco de armazenamento de dados primário e credenciais mestras finais",
            "label": "Banco de Armazenamento Central",
        },
    },
    {
        "en": {
            "filename": "master_ledger.txt",
            "desc": "sealed authorization node and final authentication payload",
            "label": "Sealed Authorization Node",
        },
        "pt-BR": {
            "filename": "livro_mestre.txt",
            "desc": "nó de autorização lacrado e carga útil de autenticação final",
            "label": "Nó de Autorização Lacrado",
        },
    },
]

DECOY_TEMPLATES = [
    {
        "en": ("security_advisory.txt", "Unverified Security Advisory"),
        "pt-BR": ("aviso_seguranca.txt", "Aviso de Segurança Não Verificado"),
    },
    {
        "en": ("intercepted_memo.txt", "Intercepted Signal Memo"),
        "pt-BR": ("memorando_interceptado.txt", "Memorando de Sinal Interceptado"),
    },
    {
        "en": ("clearance_bulletin.txt", "Revoked Clearance Bulletin"),
        "pt-BR": ("boletim_liberacao.txt", "Boletim de Liberação Revogado"),
    },
]

NOISE_TEMPLATES = [
    {
        "en": (
            "maintenance_log.txt",
            "routine environmental maintenance and HVAC servicing",
            "Facility Environmental Maintenance",
        ),
        "pt-BR": (
            "registro_manutencao.txt",
            "manutenção ambiental de rotina e revisão de climatização",
            "Manutenção Ambiental da Instalação",
        ),
    },
    {
        "en": (
            "inventory_manifest.txt",
            "cafeteria provisions and janitorial supply count",
            "Provisions & Sanitation Inventory",
        ),
        "pt-BR": (
            "manifesto_inventario.txt",
            "provisões do refeitório e contagem de materiais de limpeza",
            "Inventário de Suprimentos e Limpeza",
        ),
    },
    {
        "en": (
            "system_diagnostics.txt",
            "ambient sensor pressure and power grid diagnostics",
            "Ambient Sensor & Power Diagnostics",
        ),
        "pt-BR": (
            "diagnostico_sistema.txt",
            "pressão de sensores ambientais e diagnósticos da rede elétrica",
            "Diagnóstico de Sensores Ambientais e Energia",
        ),
    },
]

CODE_WORDS = [
    "PHOENIX", "ORION", "NEBULA", "AURORA", "VORTEX",
    "CHRONOS", "VALKYRIE", "ECLIPSE", "HORIZON", "ZENITH"
]


def normalize_language(language: str) -> str:
    """Validates and normalizes the requested language code."""
    cleaned = language.strip().lower()
    if cleaned in ("pt-br", "pt_br", "pt", "portuguese", "português"):
        return "pt-BR"
    elif cleaned in ("en", "en-us", "en_us", "english"):
        return "en"
    else:
        raise ValueError(
            f"Unsupported language '{language}'. Supported values are 'en' and 'pt-BR'."
        )


def generate_world(
    seed: int,
    output_dir: str | Path,
    answer_key_path: str | Path | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Generates a reproducible world with 5 .txt files across 3 folders.

    Requirements met:
    1. Supports 'en' and 'pt-BR' languages for all generated files and metadata.
    2. Semantic clues DO NOT reveal exact file or folder names.
    3. Agents relate semantic descriptions to the purpose/content of files.
    4. Direct clues and irrelevant files (decoy + noise) are preserved.
    5. Each environment has an unequivocal, unambiguous solution path.
    6. Fully reproducible using random.Random(seed).
    7. Solution path and secret code saved to answer_key.json OUTSIDE output_dir.
    8. Backwards compatible, defaulting to 'en'.

    Args:
        seed: Integer seed for reproducible random generation.
        output_dir: Directory where the agent world will be generated.
        answer_key_path: Optional path for answer_key.json. If None, saved in parent of output_dir.
        language: Language code ('en' or 'pt-BR'). Defaults to 'en'.

    Returns:
        dict containing the generated answer key metadata.
    """
    lang = normalize_language(language)
    is_pt = lang == "pt-BR"

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

    # Select 3 distinct sectors from canonical pool
    sector_keys = list(SECTORS.keys())
    selected_sector_keys = rng.sample(sector_keys, 3)

    # Sector physical directory names in chosen language
    sector_folders = [SECTORS[k][lang]["folder"] for k in selected_sector_keys]
    for folder_name in sector_folders:
        (base_dir / folder_name).mkdir(parents=True, exist_ok=True)

    k_step1, k_step2, k_step3 = selected_sector_keys
    f_step1 = SECTORS[k_step1][lang]["folder"]
    f_step2 = SECTORS[k_step2][lang]["folder"]
    f_step3 = SECTORS[k_step3][lang]["folder"]

    # Distribute 5 files across 3 folders: [2, 2, 1]
    # Folder 0: Step 1 + Decoy (2)
    # Folder 1: Step 2 + Noise (2)
    # Folder 2: Step 3 (1)
    f_decoy = f_step1
    f_noise = f_step2

    # Template selections (seeded via choice)
    step1_idx = rng.randrange(len(STEP1_TEMPLATES))
    step2_idx = rng.randrange(len(STEP2_TEMPLATES))
    step3_idx = rng.randrange(len(STEP3_TEMPLATES))
    decoy_idx = rng.randrange(len(DECOY_TEMPLATES))
    noise_idx = rng.randrange(len(NOISE_TEMPLATES))

    t_step1 = STEP1_TEMPLATES[step1_idx][lang]
    t_step2 = STEP2_TEMPLATES[step2_idx][lang]
    t_step3 = STEP3_TEMPLATES[step3_idx][lang]
    t_decoy_name, t_decoy_label = DECOY_TEMPLATES[decoy_idx][lang]
    t_noise_name, t_noise_desc, t_noise_label = NOISE_TEMPLATES[noise_idx][lang]

    path_step1 = f"{f_step1}/{t_step1['filename']}"
    path_step2 = f"{f_step2}/{t_step2['filename']}"
    path_step3 = f"{f_step3}/{t_step3['filename']}"
    path_decoy = f"{f_decoy}/{t_decoy_name}"
    path_noise = f"{f_noise}/{t_noise_name}"

    # Secret code generation
    code_prefix = "TESOURO" if is_pt else "TREASURE"
    secret_code = f"{code_prefix}-{rng.randint(1000, 9999)}-{rng.choice(CODE_WORDS)}"

    # Determine clue styles (one semantic, one direct)
    step1_is_semantic = rng.choice([True, False])

    # Build Step 1 Clue
    if step1_is_semantic:
        step1_clue_type = "semantic"
        step1_target_sector = SECTORS[k_step2][lang]["descriptor"]
        step1_target_purpose = t_step2["desc"]
        if is_pt:
            step1_hint = (
                f"Próxima pista (Semântica): Prossiga até {step1_target_sector} "
                f"e examine o documento detalhando {step1_target_purpose}."
            )
        else:
            step1_hint = (
                f"Next clue (Semantic): Proceed to {step1_target_sector} "
                f"and examine the document detailing {step1_target_purpose}."
            )
    else:
        step1_clue_type = "direct"
        step1_target_sector = None
        step1_target_purpose = None
        if is_pt:
            step1_hint = f"Próxima pista (Direta): Abra o arquivo '{path_step2}'."
        else:
            step1_hint = f"Next clue (Direct): Open file '{path_step2}'."

    if is_pt:
        step1_content = (
            f"=== BRIEFING DO AGENTE: INICIAÇÃO DA MISSÃO ===\n"
            f"Setor: {SECTORS[k_step1][lang]['label']}\n"
            f"Assunto: {t_step1['desc']}\n\n"
            f"Bem-vindo, operativo. Um tesouro confidencial foi protegido nesta instalação.\n"
            f"Para recuperar o código secreto, você deve seguir rigorosamente a cadeia autenticada.\n\n"
            f"{step1_hint}\n"
        )
    else:
        step1_content = (
            f"=== AGENT BRIEFING: MISSION INITIATION ===\n"
            f"Sector: {SECTORS[k_step1][lang]['label']}\n"
            f"Subject: {t_step1['desc']}\n\n"
            f"Welcome, operative. A classified treasure has been secured in this facility.\n"
            f"To recover the secret code, you must strictly follow the authenticated chain.\n\n"
            f"{step1_hint}\n"
        )

    # Build Step 2 Clue
    if step1_is_semantic:
        step2_clue_type = "direct"
        step2_target_sector = None
        step2_target_purpose = None
        if is_pt:
            step2_hint = (
                f"Próxima pista (Direta): Mova-se diretamente para '{path_step3}' onde reside o terminal final."
            )
        else:
            step2_hint = (
                f"Next clue (Direct): Move directly to '{path_step3}' where the final vault terminal resides."
            )
    else:
        step2_clue_type = "semantic"
        step2_target_sector = SECTORS[k_step3][lang]["descriptor"]
        step2_target_purpose = t_step3["desc"]
        if is_pt:
            step2_hint = (
                f"Próxima pista (Semântica): Acesse {step2_target_sector} "
                f"e inspecione o registro de console armazenando {step2_target_purpose}."
            )
        else:
            step2_hint = (
                f"Next clue (Semantic): Access {step2_target_sector} "
                f"and inspect the console record storing {step2_target_purpose}."
            )

    if is_pt:
        step2_content = (
            f"=== ANÁLISE DE CHECKPOINT: ETAPA 2 ===\n"
            f"Setor: {SECTORS[k_step2][lang]['label']}\n"
            f"Assunto: {t_step2['label']} - {t_step2['desc']}\n\n"
            f"Autenticação confirmada da Etapa 1.\n"
            f"Você está se aproximando do perímetro interno.\n\n"
            f"{step2_hint}\n"
        )
    else:
        step2_content = (
            f"=== CHECKPOINT ANALYSIS: STAGE 2 ===\n"
            f"Sector: {SECTORS[k_step2][lang]['label']}\n"
            f"Subject: {t_step2['label']} - {t_step2['desc']}\n\n"
            f"Authentication confirmed from Step 1.\n"
            f"You are approaching the inner perimeter.\n\n"
            f"{step2_hint}\n"
        )

    # Build Step 3 (Treasure) Content
    if is_pt:
        step3_content = (
            f"=== TERMINAL DO COFRE SEGURO: ETAPA FINAL ===\n"
            f"Setor: {SECTORS[k_step3][lang]['label']}\n"
            f"Assunto: {t_step3['label']} - {t_step3['desc']}\n\n"
            f"Acesso concedido. Verificação da cadeia concluída.\n\n"
            f"CÓDIGO SECRETO: {secret_code}\n\n"
            f"Registre este código secreto para concluir a missão.\n"
        )
    else:
        step3_content = (
            f"=== SECURE VAULT TERMINAL: FINAL STAGE ===\n"
            f"Sector: {SECTORS[k_step3][lang]['label']}\n"
            f"Subject: {t_step3['label']} - {t_step3['desc']}\n\n"
            f"Access granted. Chain verification complete.\n\n"
            f"SECRET CODE: {secret_code}\n\n"
            f"Record this secret code to conclude the mission.\n"
        )

    # Build Decoy
    decoy_prefix = "ISCA" if is_pt else "DECOY"
    fake_code = f"{decoy_prefix}-{rng.randint(1000, 9999)}-INVALID"
    if is_pt:
        decoy_content = (
            f"=== AVISO INTERCEPTADO (CONFIDENCIAL) ===\n"
            f"Setor: Canal de Transmissão Não Verificado\n"
            f"Assunto: {t_decoy_label}\n\n"
            f"Aviso: Relatório de agente alega que o código secreto é: {fake_code}\n\n"
            f"[AVISO DE VERIFICAÇÃO LÓGICA]\n"
            f"- Status: REVOGADO / NÃO VERIFICADO.\n"
            f"- Motivo: Este aviso não pertence à sequência de briefing autenticada.\n"
            f"- Regra 4.1: Códigos de tesouro genuínos seguem a cadeia autenticada de 3 etapas com início em '{path_step1}'.\n"
            f"- Qualquer código proveniente de avisos não verificados é uma armadilha simulada e deve ser descartado.\n"
        )
    else:
        decoy_content = (
            f"=== INTERCEPTED ADVISORY (CONFIDENTIAL) ===\n"
            f"Sector: Unverified Transmission Channel\n"
            f"Subject: {t_decoy_label}\n\n"
            f"Notice: Agent report claims the secret code is: {fake_code}\n\n"
            f"[LOGICAL VERIFICATION ADVISORY]\n"
            f"- Status: REVOKED / UNVERIFIED.\n"
            f"- Reason: This advisory does not belong to the authenticated briefing sequence.\n"
            f"- Rule 4.1: Genuine treasure codes follow the authenticated 3-step chain starting at '{path_step1}'.\n"
            f"- Any code from unverified advisories is a simulated decoy trap and must be discarded.\n"
        )

    # Build Noise
    tech_id = rng.randint(10, 99)
    section_num = rng.randint(1, 5)
    if is_pt:
        noise_content = (
            f"=== REGISTRO DE MANUTENÇÃO ROTINEIRA DA INSTALAÇÃO ===\n"
            f"Setor: Suporte de Infraestrutura da Instalação\n"
            f"Assunto: {t_noise_label} - {t_noise_desc}\n"
            f"Inspetor ID: TECH-{tech_id}\n"
            f"Status: Operações Normais\n"
            f"- Pressão atmosférica: 101.3 kPa (Nominal)\n"
            f"- Unidade de reciclagem de ar: Filtros substituídos na Seção {section_num}\n"
            f"- Dispensador de café: Limpo e reabastecido\n"
            f"Nenhum dado operacional ou relacionado à missão neste registro.\n"
        )
    else:
        noise_content = (
            f"=== ROUTINE FACILITY MAINTENANCE LOG ===\n"
            f"Sector: Facility Infrastructure Support\n"
            f"Subject: {t_noise_label} - {t_noise_desc}\n"
            f"Inspector ID: TECH-{tech_id}\n"
            f"Status: Normal Operations\n"
            f"- Atmospheric pressure: 101.3 kPa (Nominal)\n"
            f"- Air recycling unit: Filters replaced in Section {section_num}\n"
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
        "language": lang,
        "secret_code": secret_code,
        "solution_path": [path_step1, path_step2, path_step3],
        "start_file": path_step1,
        "folders": sector_folders,
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
                "description": t_noise_desc,
            },
        ],
    }

    key_file.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")

    return answer_key
