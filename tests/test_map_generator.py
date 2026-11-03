"""Unit tests for TreasureGraph directed document graph and map generator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from treasuregraph.generator import generate_world
from treasuregraph.map_generator import (
    DirectedEdge,
    DocumentGraph,
    DocumentNode,
    MapMode,
    generate_document_map,
)


class TestMapGenerator(unittest.TestCase):
    """Validates digraph document modeling, connectivity, reachability, and zero-leakage security."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.world_dir = self.base_path / "test_world"
        self.answer_key_path = self.base_path / "answer_key.json"

        # Generate standard reproducible environment (pt-BR, seed 42)
        self.answer_key = generate_world(
            output_dir=self.world_dir,
            seed=42,
            language="pt-BR",
            answer_key_path=self.answer_key_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Requirement 1: Each file is a node
    # -------------------------------------------------------------------------

    def test_each_file_becomes_a_node(self) -> None:
        """Verifies that all 5 documents in the environment correspond to graph nodes."""
        graph = generate_document_map(self.answer_key, mode=MapMode.COMPLETE)

        expected_files = {
            "arquivo/inicio_missao.txt",
            "operacoes/registro_checkpoint.txt",
            "cofre/acesso_terminal.txt",
            "arquivo/aviso_seguranca.txt",
            "operacoes/diagnostico_sistema.txt",
        }
        node_ids = {n.id for n in graph.nodes}
        self.assertEqual(node_ids, expected_files)
        self.assertEqual(len(graph.nodes), 5)

        # Verify folder and filename decomposition
        node_inicio = graph.get_node("arquivo/inicio_missao.txt")
        self.assertIsNotNone(node_inicio)
        self.assertEqual(node_inicio.folder, "arquivo")
        self.assertEqual(node_inicio.filename, "inicio_missao.txt")

    # -------------------------------------------------------------------------
    # Requirement 2: Clues create directed connections between nodes
    # -------------------------------------------------------------------------

    def test_clues_create_directed_edges(self) -> None:
        """Verifies that true clues establish directed edges: step 1 -> step 2 -> step 3."""
        graph = generate_document_map(self.answer_key, mode=MapMode.COMPLETE)

        # In complete mode, exactly 2 true directed edges exist
        self.assertEqual(len(graph.true_edges), 2)
        edge1 = graph.true_edges[0]
        edge2 = graph.true_edges[1]

        self.assertEqual(edge1.source, "arquivo/inicio_missao.txt")
        self.assertEqual(edge1.target, "operacoes/registro_checkpoint.txt")
        self.assertTrue(edge1.is_true_clue)

        self.assertEqual(edge2.source, "operacoes/registro_checkpoint.txt")
        self.assertEqual(edge2.target, "cofre/acesso_terminal.txt")
        self.assertTrue(edge2.is_true_clue)

        # In-degree and out-degree
        self.assertEqual(graph.out_degree("arquivo/inicio_missao.txt"), 1)
        self.assertEqual(graph.in_degree("arquivo/inicio_missao.txt"), 0)

        self.assertEqual(graph.in_degree("operacoes/registro_checkpoint.txt"), 1)
        self.assertEqual(graph.out_degree("operacoes/registro_checkpoint.txt"), 1)

        self.assertEqual(graph.in_degree("cofre/acesso_terminal.txt"), 1)
        self.assertEqual(graph.out_degree("cofre/acesso_terminal.txt"), 0)

    # -------------------------------------------------------------------------
    # Requirement 3: Support Complete, Partial, and Noisy (Fake Edges) Maps
    # -------------------------------------------------------------------------

    def test_complete_map_generation(self) -> None:
        """Verifies complete map contains all genuine edges and zero fake connections."""
        graph = generate_document_map(self.answer_key, mode=MapMode.COMPLETE)

        self.assertEqual(graph.mode, MapMode.COMPLETE)
        self.assertEqual(len(graph.active_edges()), 2)
        self.assertEqual(len(graph.fake_edges), 0)
        self.assertTrue(graph.is_solution_reachable())

    def test_partial_map_generation(self) -> None:
        """Verifies partial map contains an incomplete subset of the clue graph."""
        graph = generate_document_map(self.answer_key, mode=MapMode.PARTIAL)

        self.assertEqual(graph.mode, MapMode.PARTIAL)
        # Default partial drops the terminal edge: only step 1 -> step 2 is mapped
        self.assertEqual(len(graph.active_edges()), 1)
        self.assertEqual(graph.active_edges()[0].source, "arquivo/inicio_missao.txt")
        self.assertEqual(graph.active_edges()[0].target, "operacoes/registro_checkpoint.txt")
        # Since terminal edge is unmapped, terminal is not reachable purely via graph edges
        self.assertFalse(graph.is_solution_reachable())

    def test_noisy_map_generation_with_fake_edges(self) -> None:
        """Verifies noisy map includes genuine edges plus spurious decoy connections."""
        graph = generate_document_map(
            self.answer_key,
            mode=MapMode.NOISY,
            seed=42,
            num_fake_edges=3,
        )

        self.assertEqual(graph.mode, MapMode.NOISY)
        self.assertEqual(len(graph.fake_edges), 3)
        self.assertEqual(len(graph.active_edges()), 5)  # 2 true + 3 fake

        # Verify all fake edges point strictly between existing valid nodes
        all_node_ids = {n.id for n in graph.nodes}
        for fake in graph.fake_edges:
            self.assertIn(fake.source, all_node_ids)
            self.assertIn(fake.target, all_node_ids)
            self.assertNotEqual(fake.source, fake.target)
            self.assertFalse(fake.is_true_clue)

    # -------------------------------------------------------------------------
    # Requirement 4: Agent Map Security & Zero Leakage
    # -------------------------------------------------------------------------

    def test_agent_map_has_zero_leakage(self) -> None:
        """Verifies that the agent-facing map reveals NO treasure code, terminal status, or fake tags."""
        graph = generate_document_map(
            self.answer_key,
            mode=MapMode.NOISY,
            seed=42,
            num_fake_edges=3,
        )

        agent_dict = graph.to_agent_dict()
        agent_json = json.dumps(agent_dict)

        # 1. No secret code or decoy codes
        self.assertNotIn(self.answer_key["secret_code"], agent_json)
        self.assertNotIn("ISCA-", agent_json)
        self.assertNotIn("DECOY-", agent_json)

        # 2. No ground-truth / evaluator labels (must not expose metadata fields)
        forbidden_metadata_keys = [
            "solution_path",
            "is_terminal",
            "has_treasure",
            "is_true_clue",
            "clue_type",
            "secret_code",
            "role",
            "metadata",
        ]
        for key in forbidden_metadata_keys:
            self.assertNotIn(f'"{key}"', agent_json)

        # 3. No evaluator semantic indicators in any value
        forbidden_semantic_values = [
            "fake",
            "falsa",
            "is_valid",
            "spurious",
            "distractor_connection",
        ]
        for val in forbidden_semantic_values:
            self.assertNotIn(val, agent_json.lower())

        # 3. Node schema is uniform: exactly {'id', 'folder', 'filename'}
        for node in agent_dict["nodes"]:
            self.assertEqual(set(node.keys()), {"id", "folder", "filename"})

        # 4. Edge schema is uniform: exactly {'source', 'target'} for both true and fake edges
        for edge in agent_dict["edges"]:
            self.assertEqual(set(edge.keys()), {"source", "target"})

        # 5. Nodes are sorted alphabetically by ID (no sequence or positional bias)
        node_ids = [n["id"] for n in agent_dict["nodes"]]
        self.assertEqual(node_ids, sorted(node_ids))

    # -------------------------------------------------------------------------
    # Requirement 5: Store true graph separately (Evaluator only)
    # -------------------------------------------------------------------------

    def test_save_ground_truth_blocks_sandbox_leak(self) -> None:
        """Verifies save_ground_truth enforces strict separation and prevents writing inside world_dir."""
        graph = generate_document_map(self.answer_key, mode=MapMode.NOISY)

        # Saving inside world_dir must raise PermissionError
        illegal_dest = self.world_dir / "evaluator_graph.json"
        with self.assertRaises(PermissionError):
            graph.save_ground_truth(illegal_dest, world_dir=self.world_dir)

        # Saving outside world_dir succeeds
        legal_dest = self.base_path / "evaluator_graph.json"
        saved_path = graph.save_ground_truth(legal_dest, world_dir=self.world_dir)
        self.assertTrue(saved_path.is_file())

        # Ground truth file contains complete evaluation metadata
        eval_data = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(eval_data["secret_code"], self.answer_key["secret_code"])
        self.assertIn("true_edges", eval_data)
        self.assertIn("fake_edges", eval_data)

    def test_save_agent_map_inside_sandbox(self) -> None:
        """Verifies save_agent_map can be safely saved inside world_dir for agent access."""
        graph = generate_document_map(self.answer_key, mode=MapMode.COMPLETE)
        agent_map_path = self.world_dir / "mapa_documentos.json"

        saved_path = graph.save_agent_map(agent_map_path)
        self.assertTrue(saved_path.is_file())

        content = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(content["node_count"], 5)
        self.assertNotIn("secret_code", content)

    # -------------------------------------------------------------------------
    # Requirement 6: Deterministic generation by seed and backwards compatibility
    # -------------------------------------------------------------------------

    def test_deterministic_generation_by_seed(self) -> None:
        """Verifies identical seed produces identical graphs and fake edges byte-for-byte."""
        graph1 = generate_document_map(self.answer_key, mode=MapMode.NOISY, seed=123, num_fake_edges=2)
        graph2 = generate_document_map(self.answer_key, mode=MapMode.NOISY, seed=123, num_fake_edges=2)

        json1 = json.dumps(graph1.to_evaluator_dict(), sort_keys=True)
        json2 = json.dumps(graph2.to_evaluator_dict(), sort_keys=True)
        self.assertEqual(json1, json2)

        # Different seed produces different fake edges
        graph3 = generate_document_map(self.answer_key, mode=MapMode.NOISY, seed=999, num_fake_edges=2)
        json3 = json.dumps(graph3.to_evaluator_dict(), sort_keys=True)
        self.assertNotEqual(json1, json3)

    def test_compatibility_with_existing_environment(self) -> None:
        """Verifies generator works seamlessly with experiments/world_pt_001 directly."""
        existing_world = Path("experiments/world_pt_001")
        if existing_world.exists() and (existing_world.parent / "answer_key.json").exists():
            graph = generate_document_map(existing_world, mode=MapMode.COMPLETE)
            self.assertEqual(len(graph.nodes), 5)
            self.assertTrue(graph.is_solution_reachable())

    # -------------------------------------------------------------------------
    # Requirement 7: Connectivity, reachability, and cycle handling
    # -------------------------------------------------------------------------

    def test_reachability_and_pathfinding(self) -> None:
        """Verifies BFS reachability and DFS pathfinding find the exact solution path."""
        graph = generate_document_map(self.answer_key, mode=MapMode.COMPLETE)

        start = "arquivo/inicio_missao.txt"
        terminal = "cofre/acesso_terminal.txt"

        self.assertTrue(graph.is_reachable(start, terminal))
        paths = graph.find_all_paths(start, terminal)

        self.assertEqual(len(paths), 1)
        expected_path = [
            "arquivo/inicio_missao.txt",
            "operacoes/registro_checkpoint.txt",
            "cofre/acesso_terminal.txt",
        ]
        self.assertEqual(paths[0], expected_path)

    def test_cycle_resilience_with_fake_edges(self) -> None:
        """Verifies that cyclic fake edges do not cause infinite recursion in reachability queries."""
        # Create a graph with an intentional cycle: A -> B -> A
        node_a = DocumentNode(id="A", folder="f", filename="a.txt")
        node_b = DocumentNode(id="B", folder="f", filename="b.txt")
        edge_ab = DirectedEdge(source="A", target="B", is_true_clue=True)
        edge_ba = DirectedEdge(source="B", target="A", is_true_clue=False, clue_type="fake")

        graph = DocumentGraph(
            nodes=[node_a, node_b],
            true_edges=[edge_ab],
            fake_edges=[edge_ba],
            seed=42,
        )

        self.assertTrue(graph.is_reachable("A", "B"))
        self.assertTrue(graph.is_reachable("B", "A"))
        # find_all_paths terminates properly without recursion overflow
        paths = graph.find_all_paths("A", "B")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], ["A", "B"])

    def test_map_description_modes_none_short_detailed(self) -> None:
        """Verifies semantic descriptions across none, short, and detailed modes with zero leaks."""
        for desc_mode in ["none", "short", "detailed"]:
            graph = generate_document_map(
                self.answer_key,
                mode=MapMode.COMPLETE,
                description_mode=desc_mode,
            )
            agent_dict = graph.to_agent_dict()
            self.assertEqual(len(agent_dict["nodes"]), 5)

            for node in agent_dict["nodes"]:
                if desc_mode == "none":
                    self.assertNotIn("description", node)
                else:
                    self.assertIn("description", node)
                    self.assertGreater(len(node["description"]), 0)

            agent_json = json.dumps(agent_dict)
            self.assertNotIn(self.answer_key["secret_code"], agent_json)
            self.assertNotIn("is_terminal", agent_json)
            self.assertNotIn("has_treasure", agent_json)

    def test_map_scaling_to_100_nodes(self) -> None:
        """Verifies DocumentGraph scales properly to 100 files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            w_dir = Path(tmp_dir) / "world_100"
            key = generate_world(seed=42, output_dir=w_dir, language="pt-BR", total_files=100)
            graph = generate_document_map(key, mode=MapMode.COMPLETE)
            self.assertEqual(len(graph.nodes), 100)
            self.assertTrue(graph.is_solution_reachable())
            agent_dict = graph.to_agent_dict()
            self.assertEqual(agent_dict["node_count"], 100)


if __name__ == "__main__":
    unittest.main()
