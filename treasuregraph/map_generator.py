"""TreasureGraph directed document graph and map generator.

Represents documents as nodes and clues as directed edges in a directed graph.
Supports generating complete, partial, and noisy (fake edges) topological maps.
Enforces strict separation between the sanitized agent-facing map and the evaluator ground truth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import random
from typing import Any


class MapMode(str, Enum):
    """Supported graph map generation modes."""
    COMPLETE = "complete"  # 100% of true edges, 0 fake edges
    PARTIAL = "partial"    # Subset of true edges, 0 fake edges
    NOISY = "noisy"        # True edges + spurious/fake connections


@dataclass(frozen=True)
class DocumentNode:
    """Represents a document file as a node in the graph."""

    id: str  # Canonical relative path, e.g. 'arquivo/inicio_missao.txt'
    folder: str
    filename: str
    is_terminal: bool = False  # Evaluator only: contains secret code
    has_treasure: bool = False  # Evaluator only
    role: str = "document"  # Evaluator only: 'start', 'checkpoint', 'terminal', 'decoy', 'noise'
    short_desc: str | None = None
    detailed_desc: str | None = None

    def to_agent_dict(self, description_mode: str = "none") -> dict[str, str]:
        """Sanitized representation for agent map: no secret, no role, uniform schema."""
        d: dict[str, str] = {
            "id": self.id,
            "folder": self.folder,
            "filename": self.filename,
        }
        if description_mode in ("short", "curta", "descricao_curta"):
            d["description"] = self.short_desc or ""
        elif description_mode in ("detailed", "detalhada", "descricao_detalhada"):
            d["description"] = self.detailed_desc or ""
        return d

    def to_evaluator_dict(self) -> dict[str, Any]:
        """Full representation for external evaluator."""
        return {
            "id": self.id,
            "folder": self.folder,
            "filename": self.filename,
            "is_terminal": self.is_terminal,
            "has_treasure": self.has_treasure,
            "role": self.role,
            "short_desc": self.short_desc,
            "detailed_desc": self.detailed_desc,
        }


@dataclass(frozen=True)
class DirectedEdge:
    """Represents a directed clue or connection from source document to target document."""

    source: str
    target: str
    is_true_clue: bool = True  # Evaluator only
    clue_type: str = "direct"  # Evaluator only: 'direct', 'semantic', 'fake'
    step: int | None = None  # Evaluator only: step in solution chain
    metadata: str | None = None  # Evaluator only

    def to_agent_dict(self) -> dict[str, str]:
        """Sanitized representation for agent map: identical schema for true and fake edges."""
        return {
            "source": self.source,
            "target": self.target,
        }

    def to_evaluator_dict(self) -> dict[str, Any]:
        """Full representation for external evaluator."""
        return {
            "source": self.source,
            "target": self.target,
            "is_true_clue": self.is_true_clue,
            "clue_type": self.clue_type,
            "step": self.step,
            "metadata": self.metadata,
        }


class DocumentGraph:
    """Directed graph representing documents and directed clue connections."""

    def __init__(
        self,
        nodes: list[DocumentNode],
        true_edges: list[DirectedEdge],
        fake_edges: list[DirectedEdge] | None = None,
        active_true_edges: list[DirectedEdge] | None = None,
        seed: int = 42,
        mode: MapMode | str = MapMode.COMPLETE,
        start_node_id: str | None = None,
        terminal_node_id: str | None = None,
        secret_code: str | None = None,
        solution_path: list[str] | None = None,
        description_mode: str = "none",
    ) -> None:
        self.seed = seed
        self.mode = MapMode(mode) if isinstance(mode, str) else mode
        self.description_mode = description_mode
        self.nodes = list(nodes)
        self.true_edges = list(true_edges)
        self.fake_edges = list(fake_edges or [])
        self.active_true_edges = list(
            active_true_edges if active_true_edges is not None else true_edges
        )

        self.start_node_id = start_node_id
        self.terminal_node_id = terminal_node_id
        self.secret_code = secret_code
        self.solution_path = list(solution_path or [])

        # Internal node lookup
        self._node_map: dict[str, DocumentNode] = {n.id: n for n in self.nodes}

    def get_node(self, node_id: str) -> DocumentNode | None:
        """Returns node by relative path identifier."""
        return self._node_map.get(node_id)

    def active_edges(self) -> list[DirectedEdge]:
        """Returns all edges currently included in this map, sorted deterministically."""
        combined = list(self.active_true_edges) + list(self.fake_edges)
        # Deterministic sorting eliminates positional clues regarding creation order
        return sorted(combined, key=lambda e: (e.source, e.target))

    def get_successors(self, node_id: str, edges_pool: list[DirectedEdge] | None = None) -> list[str]:
        """Returns sorted list of target node IDs connected from node_id."""
        edges = self.active_edges() if edges_pool is None else edges_pool
        return sorted(e.target for e in edges if e.source == node_id)

    def get_predecessors(self, node_id: str, edges_pool: list[DirectedEdge] | None = None) -> list[str]:
        """Returns sorted list of source node IDs pointing to node_id."""
        edges = self.active_edges() if edges_pool is None else edges_pool
        return sorted(e.source for e in edges if e.target == node_id)

    def in_degree(self, node_id: str, edges_pool: list[DirectedEdge] | None = None) -> int:
        """Returns number of incoming directed edges to node_id."""
        return len(self.get_predecessors(node_id, edges_pool))

    def out_degree(self, node_id: str, edges_pool: list[DirectedEdge] | None = None) -> int:
        """Returns number of outgoing directed edges from node_id."""
        return len(self.get_successors(node_id, edges_pool))

    def is_reachable(self, source: str, target: str, edges_pool: list[DirectedEdge] | None = None) -> bool:
        """Returns True if there is a directed path from source to target using BFS."""
        if source == target:
            return True
        edges = self.active_edges() if edges_pool is None else edges_pool
        visited: set[str] = {source}
        queue: deque[str] = deque([source])

        while queue:
            curr = queue.popleft()
            for nxt in self.get_successors(curr, edges):
                if nxt == target:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def find_all_paths(
        self,
        source: str,
        target: str,
        edges_pool: list[DirectedEdge] | None = None,
        max_depth: int = 10,
    ) -> list[list[str]]:
        """Finds all simple directed paths from source to target without cycles."""
        edges = self.active_edges() if edges_pool is None else edges_pool
        results: list[list[str]] = []

        def _dfs(current: str, path: list[str]) -> None:
            if current == target:
                results.append(list(path))
                return
            if len(path) > max_depth:
                return
            for nxt in self.get_successors(current, edges):
                if nxt not in path:  # Avoid cycles
                    path.append(nxt)
                    _dfs(nxt, path)
                    path.pop()

        if source in self._node_map and target in self._node_map:
            _dfs(source, [source])
        return sorted(results, key=lambda p: (len(p), p))

    def is_solution_reachable(self) -> bool:
        """Checks if the genuine terminal node is reachable from the start node."""
        if not self.start_node_id or not self.terminal_node_id:
            return False
        return self.is_reachable(self.start_node_id, self.terminal_node_id)

    # -------------------------------------------------------------------------
    # Serialization Views
    # -------------------------------------------------------------------------

    def to_agent_dict(self, description_mode: str | None = None) -> dict[str, Any]:
        """Generates the sanitized agent view.

        Guarantees:
        - Nodes are sorted alphabetically by id (no sequence or positional leaks).
        - Edges have only 'source' and 'target' (identical schema for true and fake).
        - Absolutely no secret code, solution path, or ground-truth roles exposed.
        """
        sorted_nodes = sorted(self.nodes, key=lambda n: n.id)
        sorted_edges = self.active_edges()
        active_desc_mode = description_mode if description_mode is not None else self.description_mode

        return {
            "format": "treasuregraph_digraph_v1",
            "node_count": len(sorted_nodes),
            "edge_count": len(sorted_edges),
            "nodes": [n.to_agent_dict(active_desc_mode) for n in sorted_nodes],
            "edges": [e.to_agent_dict() for e in sorted_edges],
        }

    def to_evaluator_dict(self) -> dict[str, Any]:
        """Generates the comprehensive evaluator ground-truth view."""
        all_act = self.active_edges()
        return {
            "format": "treasuregraph_digraph_evaluator_v1",
            "seed": self.seed,
            "mode": self.mode.value,
            "start_node_id": self.start_node_id,
            "terminal_node_id": self.terminal_node_id,
            "secret_code": self.secret_code,
            "solution_path": self.solution_path,
            "node_count": len(self.nodes),
            "edge_count": len(all_act),
            "nodes": [n.to_evaluator_dict() for n in sorted(self.nodes, key=lambda n: n.id)],
            "true_edges": [e.to_evaluator_dict() for e in self.true_edges],
            "active_true_edges": [e.to_evaluator_dict() for e in self.active_true_edges],
            "fake_edges": [e.to_evaluator_dict() for e in self.fake_edges],
            "all_active_edges": [e.to_evaluator_dict() for e in all_act],
            "reachability": {
                "is_solution_reachable": self.is_solution_reachable(),
                "paths_from_start_to_terminal": self.find_all_paths(
                    self.start_node_id or "", self.terminal_node_id or ""
                ),
            },
        }

    def to_mermaid(self, evaluator_view: bool = False) -> str:
        """Generates a Mermaid diagram representing the directed document graph."""
        lines = ["flowchart TD"]

        # Clean node ID helper for Mermaid
        def _clean_id(raw_id: str) -> str:
            return raw_id.replace("/", "_").replace(".", "_").replace("-", "_")

        # Define nodes
        for node in sorted(self.nodes, key=lambda n: n.id):
            cid = _clean_id(node.id)
            label = f'"{node.id}"'
            if evaluator_view:
                if node.is_terminal:
                    lines.append(f'    {cid}["{node.id} <br/> [TESOURO]"]')
                elif node.id == self.start_node_id:
                    lines.append(f'    {cid}["{node.id} <br/> [INICIO]"]')
                elif node.role == "decoy":
                    lines.append(f'    {cid}["{node.id} <br/> [ISCA]"]')
                elif node.role == "noise":
                    lines.append(f'    {cid}["{node.id} <br/> [RUIDO]"]')
                else:
                    lines.append(f'    {cid}["{node.id}"]')
            else:
                lines.append(f'    {cid}["{node.id}"]')

        # Define edges
        for edge in self.active_edges():
            src_id = _clean_id(edge.source)
            tgt_id = _clean_id(edge.target)
            if evaluator_view:
                if edge.is_true_clue:
                    lines.append(f"    {src_id} ==>|pista genuina| {tgt_id}")
                else:
                    lines.append(f"    {src_id} -.->|conexao falsa| {tgt_id}")
            else:
                lines.append(f"    {src_id} --> {tgt_id}")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Safe Persistence
    # -------------------------------------------------------------------------

    def save_agent_map(self, output_path: str | Path) -> Path:
        """Saves the sanitized agent map to JSON."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_agent_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    def save_ground_truth(
        self,
        output_path: str | Path,
        world_dir: str | Path | None = None,
    ) -> Path:
        """Saves the ground-truth map strictly outside the agent sandbox.

        Raises:
            PermissionError: If output_path is located inside world_dir.
        """
        target = Path(output_path).resolve()
        if world_dir is not None:
            canonical_world = Path(world_dir).resolve()
            if target == canonical_world or canonical_world in target.parents:
                raise PermissionError(
                    f"Security Error: Ground truth map ({target}) cannot be written "
                    f"inside the agent sandbox directory ({canonical_world})."
                )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_evaluator_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


def generate_document_map(
    source: dict[str, Any] | str | Path,
    mode: MapMode | str = MapMode.COMPLETE,
    seed: int = 42,
    num_fake_edges: int = 0,
    edge_drop_rate: float = 0.0,
    preserved_path: bool = True,
    description_mode: str = "none",
) -> DocumentGraph:
    """Generates a directed DocumentGraph from an answer_key dict or world directory.

    Args:
        source: answer_key dictionary, path to answer_key.json, or world directory.
        mode: MapMode.COMPLETE, MapMode.PARTIAL, or MapMode.NOISY.
        seed: Random seed for deterministic generation of fake or dropped edges.
        num_fake_edges: Number of spurious edges to add when mode is NOISY (defaults to 2).
        edge_drop_rate: Probability of dropping true edges in PARTIAL mode (0.0 to 1.0).
        preserved_path: If True in PARTIAL mode, ensures the solution path remains connected
            unless edge_drop_rate is explicitly 1.0.
        description_mode: Level of topic detail in agent map ('none', 'short', 'detailed').

    Returns:
        DocumentGraph instance with both sanitized agent and evaluator views.
    """
    map_mode = MapMode(mode) if isinstance(mode, str) else mode
    rng = random.Random(seed)

    # 1. Load answer_key
    if isinstance(source, dict):
        answer_key = source
    else:
        path = Path(source).resolve()
        if path.is_file():
            answer_key = json.loads(path.read_text(encoding="utf-8"))
        elif path.is_dir():
            # Check parent for answer_key.json
            candidate = path.parent / "answer_key.json"
            if candidate.is_file():
                answer_key = json.loads(candidate.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(f"answer_key.json not found in parent of '{path}'.")
        else:
            raise FileNotFoundError(f"Source path '{path}' does not exist.")

    solution_path: list[str] = answer_key.get("solution_path", [])
    start_file: str = answer_key.get("start_file", solution_path[0] if solution_path else "")
    secret_code: str = answer_key.get("secret_code", "")
    clue_chain: list[dict[str, Any]] = answer_key.get("clue_chain", [])
    distractors: list[dict[str, Any]] = answer_key.get("distractors", [])

    # 2. Build Document Nodes
    nodes: list[DocumentNode] = []
    seen_ids: set[str] = set()

    clue_meta = {item.get("file"): item for item in clue_chain if item.get("file")}

    for idx, path_str in enumerate(solution_path):
        p = Path(path_str)
        is_terminal = (idx == len(solution_path) - 1)
        role = "start" if idx == 0 else ("terminal" if is_terminal else "checkpoint")
        c_item = clue_meta.get(path_str, {})
        s_desc = c_item.get("short_desc")
        d_desc = c_item.get("detailed_desc")
        if not s_desc:
            if idx == 0:
                s_desc = "Briefing de Despacho e Diretrizes"
                d_desc = "Briefing formal de iniciação da missão com parâmetros operacionais e diretrizes para identificação da cadeia"
            elif is_terminal:
                s_desc = "Console de Contenção e Isolamento"
                d_desc = "Terminal técnico do setor de segurança máxima para verificação e encerramento de procedimentos"
            else:
                s_desc = "Registro de Checkpoint e Navegação"
                d_desc = "Registro de verificação intermediária de perímetro com coordenadas para o terminal seguro"

        node = DocumentNode(
            id=path_str,
            folder=p.parts[0] if len(p.parts) > 1 else "",
            filename=p.name,
            is_terminal=is_terminal,
            has_treasure=is_terminal,
            role=role,
            short_desc=s_desc,
            detailed_desc=d_desc,
        )
        nodes.append(node)
        seen_ids.add(path_str)

    for dist in distractors:
        d_path = dist.get("file", "")
        if d_path and d_path not in seen_ids:
            p = Path(d_path)
            s_desc = dist.get("short_desc")
            d_desc = dist.get("detailed_desc")
            d_type = dist.get("type", "distractor")
            if not s_desc:
                if d_type == "decoy":
                    s_desc = "Boletim de Transmissões Interceptadas"
                    d_desc = "Aviso de segurança referente a transmissões externas captadas e notas sobre verificação de canais autenticados"
                else:
                    s_desc = "Diagnóstico de Sensores Ambientais e Energia"
                    d_desc = dist.get("description", "pressão de sensores ambientais e diagnósticos da rede elétrica")

            node = DocumentNode(
                id=d_path,
                folder=p.parts[0] if len(p.parts) > 1 else "",
                filename=p.name,
                is_terminal=False,
                has_treasure=False,
                role=d_type,
                short_desc=s_desc,
                detailed_desc=d_desc,
            )
            nodes.append(node)
            seen_ids.add(d_path)

    # 3. Build True Directed Edges from Clue Chain
    true_edges: list[DirectedEdge] = []
    for item in clue_chain:
        src = item.get("file")
        tgt = item.get("leads_to")
        if src and tgt:
            true_edges.append(
                DirectedEdge(
                    source=src,
                    target=tgt,
                    is_true_clue=True,
                    clue_type=item.get("type", "direct"),
                    step=item.get("step"),
                    metadata=item.get("semantic_purpose"),
                )
            )

    # 4. Handle Map Mode (Partial vs Complete)
    active_true_edges: list[DirectedEdge] = list(true_edges)
    if map_mode == MapMode.PARTIAL:
        if edge_drop_rate > 0.0:
            # Deterministically drop edges according to rate
            retained: list[DirectedEdge] = []
            for edge in true_edges:
                if rng.random() >= edge_drop_rate:
                    retained.append(edge)
            # If all were dropped but not 1.0 rate, retain at least one
            if not retained and true_edges and edge_drop_rate < 1.0:
                retained.append(rng.choice(true_edges))
            active_true_edges = retained
        else:
            # Default partial behavior: omit the last clue step to create an incomplete blueprint
            if len(true_edges) > 1:
                active_true_edges = true_edges[:-1]

    # 5. Handle Fake Edges (Noisy Mode)
    fake_edges: list[DirectedEdge] = []
    if map_mode == MapMode.NOISY or num_fake_edges > 0:
        target_fake_count = num_fake_edges if num_fake_edges > 0 else 2
        all_node_ids = sorted(seen_ids)
        true_pairs = {(e.source, e.target) for e in true_edges}

        # Generate all plausible candidate non-true directed pairs
        candidates: list[tuple[str, str]] = []
        for u in all_node_ids:
            for v in all_node_ids:
                if u != v and (u, v) not in true_pairs:
                    candidates.append((u, v))

        # Deterministic sampling
        candidates.sort()
        rng.shuffle(candidates)
        chosen_pairs = candidates[: min(target_fake_count, len(candidates))]

        for u, v in chosen_pairs:
            fake_edges.append(
                DirectedEdge(
                    source=u,
                    target=v,
                    is_true_clue=False,
                    clue_type="fake",
                    step=None,
                    metadata="Spurious distractor connection",
                )
            )

    terminal_id = solution_path[-1] if solution_path else None

    return DocumentGraph(
        nodes=nodes,
        true_edges=true_edges,
        fake_edges=fake_edges,
        active_true_edges=active_true_edges,
        seed=seed,
        mode=map_mode,
        start_node_id=start_file,
        terminal_node_id=terminal_id,
        secret_code=secret_code,
        solution_path=solution_path,
        description_mode=description_mode,
    )
