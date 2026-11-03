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

# Expanded neutral engineering topics for neutral world generation (no functional/mission clues)
NEUTRAL_TECHNICAL_TOPICS = [
    {
        "short_en": "Analog Instrumentation Calibration",
        "short_pt": "Calibração de Instrumentos Analógicos",
        "detailed_en": "Periodic technical verification and zero-offset calibration of pressure transducers",
        "detailed_pt": "Aferição técnica periódica e ajuste de zero de transdutores de pressão analógicos",
        "subject_en": "Pressure Transducer Calibration",
        "subject_pt": "Calibração de Transdutores de Pressão",
    },
    {
        "short_en": "HVAC Environmental Telemetry",
        "short_pt": "Telemetria de Climatização Ambiental",
        "detailed_en": "Chilled water loop flow rates and air handling unit motor telemetry",
        "detailed_pt": "Vazão de água gelada e telemetria de motores das unidades de tratamento de ar",
        "subject_en": "HVAC Air Flow Readings",
        "subject_pt": "Leituras de Fluxo de Ar e Climatização",
    },
    {
        "short_en": "Sanitary Waste Disposal Protocol",
        "short_pt": "Protocolo de Descarte Sanitário",
        "detailed_en": "Custody transfer documentation for recycled hydraulic fluid and cleanroom filters",
        "detailed_pt": "Documentação de transferência de custódia para óleo hidráulico e filtros de sala limpa",
        "subject_en": "Industrial Waste Disposal",
        "subject_pt": "Descarte de Resíduos Industriais",
    },
    {
        "short_en": "Technical Shift Duty Roster",
        "short_pt": "Escala Semanal de Turnos Técnicos",
        "detailed_en": "Duty supervisor assignments and technical rotation schedules for facility maintenance",
        "detailed_pt": "Escala de supervisores e turnos de rotação técnica de manutenção predial",
        "subject_en": "Facility Operations Duty Roster",
        "subject_pt": "Escala de Operações e Manutenção",
    },
    {
        "short_en": "Substation Power Distribution Log",
        "short_pt": "Diagnóstico de Distribuição Elétrica",
        "detailed_en": "Harmonic distortion readings, transformer temperature logs, and busbar telemetry",
        "detailed_pt": "Leituras de distorção harmônica, temperatura de transformadores e barramentos",
        "subject_en": "Substation Busbar Telemetry",
        "subject_pt": "Telemetria de Barramentos da Subestação",
    },
    {
        "short_en": "Structural Facility Integrity Check",
        "short_pt": "Inspeção Estrutural da Instalação",
        "detailed_en": "Seismic damping joint expansion measurements and expansion joint seal audit",
        "detailed_pt": "Medições de dilatação em juntas estruturais e vedação de juntas de expansão",
        "subject_en": "Structural Joint Inspection",
        "subject_pt": "Inspeção de Juntas Estruturais",
    },
    {
        "short_en": "Consumables Inventory Requisition",
        "short_pt": "Requisição de Suprimentos de Consumo",
        "detailed_en": "Standard replenishment order for lubricants, wipes, and PPE cartridges",
        "detailed_pt": "Pedido padrão de reposição de lubrificantes, panos industriais e filtros de EPI",
        "subject_en": "Consumable Stock Replenishment",
        "subject_pt": "Reposição de Estoque de Consumíveis",
    },
    {
        "short_en": "Hydraulic Auxiliary Pump Inspection",
        "short_pt": "Inspeção de Bombas Hidráulicas",
        "detailed_en": "Differential pressure across primary filters and oil reservoir temperature audit",
        "detailed_pt": "Pressão diferencial em filtros principais e auditoria térmica do reservatório de óleo",
        "subject_en": "Hydraulic Pump Circuit Audit",
        "subject_pt": "Auditoria de Circuitos Hidráulicos",
    },
    {
        "short_en": "Emergency Lighting Battery Test",
        "short_pt": "Teste de Baterias de Emergência",
        "detailed_en": "Discharge test cycle for backup LED arrays and decentralized battery banks",
        "detailed_pt": "Ciclo de teste de descarga de luminárias LED de emergência e bancos de baterias",
        "subject_en": "Emergency Power Battery Bank",
        "subject_pt": "Banco de Baterias de Emergência",
    },
    {
        "short_en": "Fire Suppression Nozzle Verification",
        "short_pt": "Verificação de Bicos de Incêndio",
        "detailed_en": "Inert gas cylinder hydrostatic pressure check and discharge valve seal verification",
        "detailed_pt": "Inspeção de pressão hidrostática de cilindros de gás inerte e lacres de válvulas",
        "subject_en": "Inert Gas Suppression Cylinders",
        "subject_pt": "Cilindros de Supressão por Gás Inerte",
    },
    {
        "short_en": "Refrigeration Chiller Performance Log",
        "short_pt": "Registro de Operação de Chillers",
        "detailed_en": "Compressor vibration analysis, evaporator approach temperature, and refrigerant levels",
        "detailed_pt": "Análise de vibração de compressores, temperatura de aproximação e nível de refrigerante",
        "subject_en": "Chiller Compressor Vibration",
        "subject_pt": "Vibração de Compressores de Chillers",
    },
    {
        "short_en": "Backup Diesel Generator Test",
        "short_pt": "Ensaio de Gerador Diesel Auxiliar",
        "detailed_en": "Cold crank sequencing, automatic transfer switch timing, and exhaust temperature check",
        "detailed_pt": "Sequência de partida a frio, tempo de comutação da chave de transferência e exaustão",
        "subject_en": "Generator ATS Switch Timing",
        "subject_pt": "Tempo de Comutação do Gerador",
    },
    {
        "short_en": "Serial Bus Telemetry Log",
        "short_pt": "Telemetria de Barramento Serial",
        "detailed_en": "Packet synchronization rates, frame error telemetry, and parity verification",
        "detailed_pt": "Taxa de sincronização de pacotes, telemetria de erros de quadro e verificação de paridade",
        "subject_en": "Serial Bus Packet Metrics",
        "subject_pt": "Métricas de Pacotes de Barramento",
    },
    {
        "short_en": "Network Redundancy Specification",
        "short_pt": "Especificação de Redundância de Rede",
        "detailed_en": "Topology routing table convergence timings and failover latency measurements",
        "detailed_pt": "Tempo de convergência de tabelas de roteamento e latência de comutação por falha",
        "subject_en": "Routing Table Convergence",
        "subject_pt": "Convergência de Roteamento de Rede",
    },
    {
        "short_en": "Processing Unit Efficiency Report",
        "short_pt": "Eficiência Térmica de Processamento",
        "detailed_en": "Power usage effectiveness calculations and heat sink thermal gradient metrics",
        "detailed_pt": "Cálculos de eficiência energética e gradiente térmico de dissipadores de calor",
        "subject_en": "Processor Thermal Dissipation",
        "subject_pt": "Dissipação Térmica de Processadores",
    },
    {
        "short_en": "Optical Fiber Interface Survey",
        "short_pt": "Inspeção de Interfaces Ópticas",
        "detailed_en": "Optical signal attenuation per kilometer and optical connector face cleanliness",
        "detailed_pt": "Atenuação de sinal óptico por quilômetro e conformidade de conectores de fibra",
        "subject_en": "Fiber Attenuation Profile",
        "subject_pt": "Perfil de Atenuação de Fibra Óptica",
    },
    {
        "short_en": "Atomic Clock Synchronization Log",
        "short_pt": "Sincronização de Relógio Atômico",
        "detailed_en": "Frequency stability measurements, phase offset tracking, and NTP drift log",
        "detailed_pt": "Medições de estabilidade de frequência, rastreamento de fase e desvio de relógio",
        "subject_en": "Time Base Frequency Drift",
        "subject_pt": "Desvio de Frequência de Base Temporal",
    },
    {
        "short_en": "Memory Bus Parity Analysis",
        "short_pt": "Análise de Paridade de Memória",
        "detailed_en": "Single-bit error correction counters and multi-channel memory bus integrity test",
        "detailed_pt": "Contadores de correção de erro de bit único e integridade de barramento de memória",
        "subject_en": "Memory Error Correction Rates",
        "subject_pt": "Taxa de Correção de Erros de Memória",
    },
    {
        "short_en": "Galvanic Isolation Safety Audit",
        "short_pt": "Isolamento Galvânico de Circuitos",
        "detailed_en": "Dielectric breakdown threshold test across high-voltage isolation barriers",
        "detailed_pt": "Ensaio de tensão disruptiva dielétrica em barreiras de isolamento galvânico",
        "subject_en": "Dielectric Breakdown Log",
        "subject_pt": "Registro de Rigidez Dielétrica",
    },
    {
        "short_en": "Electromagnetic Compatibility Survey",
        "short_pt": "Compatibilidade Eletromagnética",
        "detailed_en": "Broadband radiofrequency scanning across containment enclosures",
        "detailed_pt": "Varredura de radiofrequência em gabinetes de contenção de sinal",
        "subject_en": "RF Spectral Profile",
        "subject_pt": "Perfil Espectral de Radiofrequência",
    },
]

PROCEDURAL_NOISE_TOPICS = [
    {
        "prefix_en": "calibration_log",
        "prefix_pt": "registro_calibracao",
        "short_en": "Analog Instrumentation Calibration",
        "short_pt": "Calibração de Instrumentos Analógicos",
        "detailed_en": "Periodic technical verification and zero-offset calibration of pressure transducers",
        "detailed_pt": "Aferição técnica periódica e ajuste de zero de transdutores de pressão analógicos",
        "subject_en": "Pressure Transducer Calibration",
        "subject_pt": "Calibração de Transdutores de Pressão",
    },
    {
        "prefix_en": "hvac_telemetry",
        "prefix_pt": "telemetria_climatizacao",
        "short_en": "HVAC Environmental Telemetry",
        "short_pt": "Telemetria de Climatização Ambiental",
        "detailed_en": "Chilled water loop flow rates and air handling unit motor telemetry",
        "detailed_pt": "Vazão de água gelada e telemetria de motores das unidades de tratamento de ar",
        "subject_en": "HVAC Air Flow Readings",
        "subject_pt": "Leituras de Fluxo de Ar e Climatização",
    },
    {
        "prefix_en": "waste_manifest",
        "prefix_pt": "manifesto_descarte",
        "short_en": "Sanitary Waste Disposal Protocol",
        "short_pt": "Protocolo de Descarte Sanitário",
        "detailed_en": "Custody transfer documentation for recycled hydraulic fluid and cleanroom filters",
        "detailed_pt": "Documentação de transferência de custódia para óleo hidráulico e filtros de sala limpa",
        "subject_en": "Industrial Waste Disposal",
        "subject_pt": "Descarte de Resíduos Industriais",
    },
    {
        "prefix_en": "shift_roster",
        "prefix_pt": "escala_turnos",
        "short_en": "Technical Shift Duty Roster",
        "short_pt": "Escala Semanal de Turnos Técnicos",
        "detailed_en": "Duty supervisor assignments and technical rotation schedules for facility maintenance",
        "detailed_pt": "Escala de supervisores e turnos de rotação técnica de manutenção predial",
        "subject_en": "Facility Operations Duty Roster",
        "subject_pt": "Escala de Operações e Manutenção",
    },
    {
        "prefix_en": "power_diagnostics",
        "prefix_pt": "consumo_energia",
        "short_en": "Substation Power Distribution Log",
        "short_pt": "Diagnóstico de Distribuição Elétrica",
        "detailed_en": "Harmonic distortion readings, transformer temperature logs, and busbar telemetry",
        "detailed_pt": "Leituras de distorção harmônica, temperatura de transformadores e barramentos",
        "subject_en": "Substation Busbar Telemetry",
        "subject_pt": "Telemetria de Barramentos da Subestação",
    },
    {
        "prefix_en": "building_inspection",
        "prefix_pt": "inspecao_predial",
        "short_en": "Structural Facility Integrity Check",
        "short_pt": "Inspeção Estrutural da Instalação",
        "detailed_en": "Seismic damping joint expansion measurements and expansion joint seal audit",
        "detailed_pt": "Medições de dilatação em juntas estruturais e vedação de juntas de expansão",
        "subject_en": "Structural Joint Inspection",
        "subject_pt": "Inspeção de Juntas Estruturais",
    },
    {
        "prefix_en": "supply_requisition",
        "prefix_pt": "requisicao_suprimentos",
        "short_en": "Consumables Inventory Requisition",
        "short_pt": "Requisição de Suprimentos de Consumo",
        "detailed_en": "Standard replenishment order for lubricants, wipes, and PPE cartridges",
        "detailed_pt": "Pedido padrão de reposição de lubrificantes, panos industriais e filtros de EPI",
        "subject_en": "Consumable Stock Replenishment",
        "subject_pt": "Reposição de Estoque de Consumíveis",
    },
    {
        "prefix_en": "hydraulic_pressure",
        "prefix_pt": "pressao_hidraulica",
        "short_en": "Hydraulic Auxiliary Pump Inspection",
        "short_pt": "Inspeção de Bombas Hidráulicas",
        "detailed_en": "Differential pressure across primary filters and oil reservoir temperature audit",
        "detailed_pt": "Pressão diferencial em filtros principais e auditoria térmica do reservatório de óleo",
        "subject_en": "Hydraulic Pump Circuit Audit",
        "subject_pt": "Auditoria de Circuitos Hidráulicos",
    },
    {
        "prefix_en": "lighting_audit",
        "prefix_pt": "auditoria_iluminacao",
        "short_en": "Emergency Lighting Battery Test",
        "short_pt": "Teste de Baterias de Emergência",
        "detailed_en": "Discharge test cycle for backup LED arrays and decentralized battery banks",
        "detailed_pt": "Ciclo de teste de descarga de luminárias LED de emergência e bancos de baterias",
        "subject_en": "Emergency Power Battery Bank",
        "subject_pt": "Banco de Baterias de Emergência",
    },
    {
        "prefix_en": "fire_safety",
        "prefix_pt": "seguranca_incendio",
        "short_en": "Fire Suppression Nozzle Verification",
        "short_pt": "Verificação de Bicos de Incêndio",
        "detailed_en": "Inert gas cylinder hydrostatic pressure check and discharge valve seal verification",
        "detailed_pt": "Inspeção de pressão hidrostática de cilindros de gás inerte e lacres de válvulas",
        "subject_en": "Inert Gas Suppression Cylinders",
        "subject_pt": "Cilindros de Supressão por Gás Inerte",
    },
    {
        "prefix_en": "chiller_log",
        "prefix_pt": "registro_chiller",
        "short_en": "Refrigeration Chiller Performance Log",
        "short_pt": "Registro de Operação de Chillers",
        "detailed_en": "Compressor vibration analysis, evaporator approach temperature, and refrigerant levels",
        "detailed_pt": "Análise de vibração de compressores, temperatura de aproximação e nível de refrigerante",
        "subject_en": "Chiller Compressor Vibration",
        "subject_pt": "Vibração de Compressores de Chillers",
    },
    {
        "prefix_en": "telecom_status",
        "prefix_pt": "status_telecomunicacoes",
        "short_en": "Internal Optical Fiber Link Telemetry",
        "short_pt": "Telemetria de Enlace Óptico Interno",
        "detailed_en": "Optical time-domain reflectometer attenuation logs for inter-sector trunk fibers",
        "detailed_pt": "Atenuação de enlaces de fibra óptica de comunicação tronco entre setores",
        "subject_en": "Inter-Sector Trunk Fiber Audit",
        "subject_pt": "Auditoria de Fibras de Comunicação Tronco",
    },
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


def generate_neutral_world(
    seed: int,
    output_dir: str | Path,
    answer_key_path: str | Path | None = None,
    language: str = "en",
    total_files: int = 5,
) -> dict[str, Any]:
    """Generates a neutral benchmark world using doc_001.txt identifiers and rich distractor graph.

    Eliminates structural, lexical, and topological leaks:
    1. Neutral IDs: All files named doc_001.txt to doc_N.txt, avoiding functional folder/file names.
    2. Rich Distractor Graph: Generates real directed connections among irrelevant documents.
    3. Multiple Chains & Topologies: Distributes documents into branches, dead ends, and cycles.
    4. Multiple Components: Solution path is not the sole connected component.
    5. Invariant 3-step Solution: start_file (doc_001.txt) -> mid_file -> goal_file.
    6. Non-leaking Descriptions: Descriptions drawn from domain-neutral engineering topics.
    7. Bilingual & Deterministic: Reproducible via random.Random(seed) in 'en' or 'pt-BR'.
    8. Separate Answer Key: Answer key with full ground truth written outside world sandbox.
    """
    if total_files < 5:
        raise ValueError(f"total_files must be >= 5, got {total_files}")

    lang = normalize_language(language)
    is_pt = (lang == "pt-BR")
    rng = random.Random(seed)

    base_dir = Path(output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if answer_key_path:
        key_file = Path(answer_key_path).resolve()
    else:
        key_file = base_dir.parent / "answer_key.json"
    key_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Neutral Node Identifiers
    node_ids = [f"doc_{i:03d}.txt" for i in range(1, total_files + 1)]
    start_file = node_ids[0]

    # Sample mid and goal nodes
    available = [n for n in node_ids if n != start_file]
    rng.shuffle(available)
    step2_file = available.pop()
    step3_file = available.pop()
    solution_path = [start_file, step2_file, step3_file]
    true_edges = [(start_file, step2_file), (step2_file, step3_file)]

    # 2. Secret code
    code_prefix = "TESOURO" if is_pt else "TREASURE"
    secret_code = f"{code_prefix}-{rng.randint(1000, 9999)}-{rng.choice(CODE_WORDS)}"

    # 3. Distractor Graph Generation
    distractor_edges: list[tuple[str, str]] = []

    # Competing branch from start_node: creates alternative path(s) of length 2 from start_node
    branch_count = 1 if total_files <= 10 else 2
    for _ in range(branch_count):
        if available:
            b1 = available.pop()
            distractor_edges.append((start_file, b1))
            if available:
                b2 = available.pop()
                distractor_edges.append((b1, b2))

    # Determine number of components (at least 3 components for <= 20, 4 for <= 50, 5 for > 50)
    num_components = 3 if total_files <= 20 else (4 if total_files <= 50 else 5)
    clusters: list[list[str]] = [[] for _ in range(num_components - 1)]
    idx = 0
    while available:
        clusters[idx % len(clusters)].append(available.pop())
        idx += 1

    # Connect nodes within each distractor cluster
    for cluster in clusters:
        if not cluster:
            continue
        c_nodes = list(cluster)
        root = c_nodes[0]
        assigned = [root]
        for node in c_nodes[1:]:
            parent = rng.choice(assigned)
            distractor_edges.append((parent, node))
            assigned.append(node)

        # Add directed cycles among distractors with probability 0.8
        if len(c_nodes) >= 2 and rng.random() < 0.8:
            n1, n2 = rng.sample(c_nodes, 2)
            if (n1, n2) not in distractor_edges:
                distractor_edges.append((n1, n2))

        # Add candidate paths of length 2 within cluster with probability 0.7
        if len(c_nodes) >= 3 and rng.random() < 0.7:
            cand = rng.sample(c_nodes, 3)
            if (cand[0], cand[1]) not in distractor_edges:
                distractor_edges.append((cand[0], cand[1]))
            if (cand[1], cand[2]) not in distractor_edges:
                distractor_edges.append((cand[1], cand[2]))

    # Deduplicate edges preserving order
    all_dist_set = set()
    cleaned_dist_edges = []
    for u, v in distractor_edges:
        if (u, v) not in all_dist_set and (u, v) not in true_edges:
            all_dist_set.add((u, v))
            cleaned_dist_edges.append((u, v))
    distractor_edges = cleaned_dist_edges

    # Outgoing references map for generating document contents
    outgoing_map: dict[str, list[str]] = {n: [] for n in node_ids}
    for u, v in true_edges:
        outgoing_map[u].append(v)
    for u, v in distractor_edges:
        outgoing_map[u].append(v)

    # 4. Neutral Technical Topics Assignment
    topics_pool = list(NEUTRAL_TECHNICAL_TOPICS)
    rng.shuffle(topics_pool)
    topics_map = {}
    for i, nid in enumerate(node_ids):
        tmpl = topics_pool[i % len(topics_pool)]
        cycle = (i // len(topics_pool)) + 1
        s_suffix = f" {cycle}" if cycle > 1 else ""
        topics_map[nid] = {
            "short_en": f"{tmpl['short_en']}{s_suffix}",
            "short_pt": f"{tmpl['short_pt']}{s_suffix}",
            "detailed_en": f"{tmpl['detailed_en']}{s_suffix}",
            "detailed_pt": f"{tmpl['detailed_pt']}{s_suffix}",
            "subject_en": f"{tmpl['subject_en']}{s_suffix}",
            "subject_pt": f"{tmpl['subject_pt']}{s_suffix}",
        }

    # 5. File Contents Generation
    files_to_write = []
    for nid in node_ids:
        t_info = topics_map[nid]
        subject = t_info["subject_pt"] if is_pt else t_info["subject_en"]
        header = (
            f"=== REGISTRO TÉCNICO: {nid} ===\n"
            f"Assunto: {subject}\n"
            f"Status Operacional: Verificado\n\n"
        ) if is_pt else (
            f"=== TECHNICAL RECORD: {nid} ===\n"
            f"Subject: {subject}\n"
            f"Operational Status: Verified\n\n"
        )

        if nid == start_file:
            clue = (
                f"Parâmetros operacionais verificados e registrados em conformidade técnica.\n"
                f"Para prosseguir na cadeia de verificação, examine o documento {step2_file} com os registros de telemetria.\n"
            ) if is_pt else (
                f"Operational parameters verified and recorded in technical compliance.\n"
                f"To continue the verification chain, examine document {step2_file} with telemetry logs.\n"
            )
            content = header + clue
        elif nid == step2_file:
            clue = (
                f"Validação intermediária aprovada. O registro de encerramento com os parâmetros de autenticação está em {step3_file}.\n"
            ) if is_pt else (
                f"Intermediate validation approved. The concluding record with authentication parameters is in {step3_file}.\n"
            )
            content = header + clue
        elif nid == step3_file:
            goal_text = (
                f"Procedimento de verificação concluído com êxito.\n"
                f"Todos os subsistemas operam em conformidade técnica.\n\n"
                f"CÓDIGO SECRETO: {secret_code}\n\n"
                f"Registre este código para validação final.\n"
            ) if is_pt else (
                f"Verification procedure completed successfully.\n"
                f"All subsystems operating in technical compliance.\n\n"
                f"SECRET CODE: {secret_code}\n\n"
                f"Record this code for final validation.\n"
            )
            content = header + goal_text
        else:
            # Distractor node
            targets = outgoing_map.get(nid, [])
            if targets:
                # Mention the first outgoing target as a technical cross reference
                ref = targets[0]
                dist_text = (
                    f"Inspeção técnica de rotina concluída. Parâmetros dentro das faixas nominais.\n"
                    f"Referência técnica cruzada: consulte o documento {ref} para especificações complementares.\n"
                ) if is_pt else (
                    f"Routine technical inspection completed. Parameters within nominal range.\n"
                    f"Technical cross-reference: refer to document {ref} for complementary specifications.\n"
                )
            else:
                dist_text = (
                    f"Procedimento operacional normalizado. Leituras estáveis em toda a faixa de tolerância.\n"
                    f"Nenhum procedimento adicional requerido.\n"
                ) if is_pt else (
                    f"Operational procedure standardized. Stable readings across tolerance range.\n"
                    f"No further procedure required.\n"
                )
            content = header + dist_text

        files_to_write.append((base_dir / nid, content))

    for fpath, fcontent in files_to_write:
        fpath.write_text(fcontent, encoding="utf-8")

    # 6. Build Answer Key
    answer_key = {
        "seed": seed,
        "language": lang,
        "total_files": total_files,
        "neutral_naming": True,
        "secret_code": secret_code,
        "start_file": start_file,
        "solution_path": solution_path,
        "folders": [],
        "clue_chain": [
            {
                "step": 1,
                "file": start_file,
                "type": "direct",
                "leads_to": step2_file,
                "short_desc": topics_map[start_file]["short_pt" if is_pt else "short_en"],
                "detailed_desc": topics_map[start_file]["detailed_pt" if is_pt else "detailed_en"],
            },
            {
                "step": 2,
                "file": step2_file,
                "type": "direct",
                "leads_to": step3_file,
                "short_desc": topics_map[step2_file]["short_pt" if is_pt else "short_en"],
                "detailed_desc": topics_map[step2_file]["detailed_pt" if is_pt else "detailed_en"],
            },
            {
                "step": 3,
                "file": step3_file,
                "type": "terminal",
                "secret_code": secret_code,
                "short_desc": topics_map[step3_file]["short_pt" if is_pt else "short_en"],
                "detailed_desc": topics_map[step3_file]["detailed_pt" if is_pt else "detailed_en"],
            },
        ],
        "distractor_edges": [
            {"source": u, "target": v} for u, v in sorted(distractor_edges)
        ],
        "distractors": [
            {
                "file": n,
                "type": "distractor",
                "short_desc": topics_map[n]["short_pt" if is_pt else "short_en"],
                "detailed_desc": topics_map[n]["detailed_pt" if is_pt else "detailed_en"],
            }
            for n in node_ids
            if n not in solution_path
        ],
    }

    key_file.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")
    return answer_key


def generate_world(
    seed: int,
    output_dir: str | Path,
    answer_key_path: str | Path | None = None,
    language: str = "en",
    total_files: int = 5,
    neutral_ids: bool = False,
) -> dict[str, Any]:
    if neutral_ids:
        return generate_neutral_world(
            seed=seed,
            output_dir=output_dir,
            answer_key_path=answer_key_path,
            language=language,
            total_files=total_files,
        )
    """Generates a reproducible world with N .txt files across 3 folders.

    Requirements met:
    1. Supports 'en' and 'pt-BR' languages for all generated files and metadata.
    2. Semantic clues DO NOT reveal exact file or folder names.
    3. Agents relate semantic descriptions to the purpose/content of files.
    4. Direct clues and irrelevant files (decoy + noise) are preserved.
    5. Each environment has an unequivocal, unambiguous solution path.
    6. Fully reproducible using random.Random(seed).
    7. Solution path and secret code saved to answer_key.json OUTSIDE output_dir.
    8. Backwards compatible, defaulting to 'en' and total_files=5.
    9. Supports scalable environments (5, 10, 20, 50, 100+ files) while holding
       the 3-step solution chain and difficulty constant.

    Args:
        seed: Integer seed for reproducible random generation.
        output_dir: Directory where the agent world will be generated.
        answer_key_path: Optional path for answer_key.json. If None, saved in parent of output_dir.
        language: Language code ('en' or 'pt-BR'). Defaults to 'en'.
        total_files: Total number of files to generate (minimum 5). Defaults to 5.

    Returns:
        dict containing the generated answer key metadata.
    """
    if total_files < 5:
        raise ValueError(f"total_files must be at least 5 (got {total_files}).")

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

    # Write initial files to disk
    files_to_write = [
        (base_dir / path_step1, step1_content),
        (base_dir / path_step2, step2_content),
        (base_dir / path_step3, step3_content),
        (base_dir / path_decoy, decoy_content),
        (base_dir / path_noise, noise_content),
    ]

    # Neutral topic descriptions for semantic map integration
    step1_short = "Briefing de Despacho e Diretrizes" if is_pt else "Dispatch Briefing & Directives"
    step1_detailed = (
        "Briefing formal de iniciação da missão com parâmetros operacionais e diretrizes para identificação da cadeia"
        if is_pt
        else "Formal mission initiation briefing with operational parameters and chain identification directives"
    )

    step2_short = "Registro de Checkpoint e Navegação" if is_pt else "Checkpoint & Navigation Log"
    step2_detailed = (
        "Registro de verificação intermediária de perímetro com coordenadas para o terminal seguro"
        if is_pt
        else "Intermediate perimeter verification log with coordinates to the secure terminal"
    )

    step3_short = "Console de Contenção e Isolamento" if is_pt else "Containment & Isolation Console"
    step3_detailed = (
        "Terminal técnico do setor de segurança máxima para verificação e encerramento de procedimentos"
        if is_pt
        else "Technical terminal in maximum-security sector for verification and procedure conclusion"
    )

    decoy_short = "Boletim de Transmissões Interceptadas" if is_pt else "Intercepted Transmissions Bulletin"
    decoy_detailed = (
        "Aviso de segurança referente a transmissões externas captadas e notas sobre verificação de canais autenticados"
        if is_pt
        else "Security notice regarding intercepted external transmissions and unverified channel advisories"
    )

    noise_short = t_noise_label
    noise_detailed = t_noise_desc

    extra_distractors: list[dict[str, Any]] = []
    if total_files > 5:
        extra_count = total_files - 5
        for idx in range(1, extra_count + 1):
            tmpl = PROCEDURAL_NOISE_TOPICS[(idx - 1) % len(PROCEDURAL_NOISE_TOPICS)]
            target_folder = sector_folders[(idx - 1) % len(sector_folders)]
            prefix = tmpl["prefix_pt"] if is_pt else tmpl["prefix_en"]
            short_desc = tmpl["short_pt"] if is_pt else tmpl["short_en"]
            detailed_desc = tmpl["detailed_pt"] if is_pt else tmpl["detailed_en"]
            subject = tmpl["subject_pt"] if is_pt else tmpl["subject_en"]
            filename = f"{prefix}_{idx:03d}.txt"
            file_rel_path = f"{target_folder}/{filename}"

            tech_num = 100 + (idx * 7) % 899
            sec_num = 1 + (idx % 8)

            if is_pt:
                extra_content = (
                    f"=== REGISTRO TÉCNICO ROTINEIRO DA INSTALAÇÃO ===\n"
                    f"Setor: {target_folder}\n"
                    f"Assunto: {subject}\n"
                    f"Inspetor ID: TECH-{tech_num}\n"
                    f"Status Operacional: Nominal\n"
                    f"- Telemetria: Sistema operando dentro dos parâmetros de conformidade.\n"
                    f"- Procedimento: Inspeção preventiva periódica na Seção {sec_num}.\n"
                    f"- Registro arquivado sem pendências ou intervenções ativas.\n"
                    f"Nenhum dado operacional ou relacionado à missão neste registro.\n"
                )
            else:
                extra_content = (
                    f"=== ROUTINE TECHNICAL FACILITY LOG ===\n"
                    f"Sector: {target_folder}\n"
                    f"Subject: {subject}\n"
                    f"Inspector ID: TECH-{tech_num}\n"
                    f"Operational Status: Nominal\n"
                    f"- Telemetry: System operating within compliance standards.\n"
                    f"- Procedure: Periodic preventive check in Section {sec_num}.\n"
                    f"- Log filed with no active open work orders.\n"
                    f"No operational or mission-related data in this log.\n"
                )

            files_to_write.append((base_dir / file_rel_path, extra_content))
            extra_distractors.append(
                {
                    "file": file_rel_path,
                    "type": "noise",
                    "description": detailed_desc,
                    "short_desc": short_desc,
                    "detailed_desc": detailed_desc,
                }
            )

    for file_path, content in files_to_write:
        file_path.write_text(content, encoding="utf-8")

    # Build answer key
    answer_key = {
        "seed": seed,
        "language": lang,
        "total_files": len(files_to_write),
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
                "short_desc": step1_short,
                "detailed_desc": step1_detailed,
            },
            {
                "step": 2,
                "file": path_step2,
                "type": step2_clue_type,
                "leads_to": path_step3,
                "semantic_sector": step2_target_sector,
                "semantic_purpose": step2_target_purpose,
                "short_desc": step2_short,
                "detailed_desc": step2_detailed,
            },
            {
                "step": 3,
                "file": path_step3,
                "type": "terminal",
                "secret_code": secret_code,
                "short_desc": step3_short,
                "detailed_desc": step3_detailed,
            },
        ],
        "distractors": [
            {
                "file": path_decoy,
                "type": "decoy",
                "flaw": "Marked REVOKED/UNVERIFIED; breaks Rule 4.1 of the authenticated chain.",
                "fake_code": fake_code,
                "short_desc": decoy_short,
                "detailed_desc": decoy_detailed,
            },
            {
                "file": path_noise,
                "type": "noise",
                "description": t_noise_desc,
                "short_desc": noise_short,
                "detailed_desc": noise_detailed,
            },
            *extra_distractors,
        ],
    }

    key_file.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")

    return answer_key
