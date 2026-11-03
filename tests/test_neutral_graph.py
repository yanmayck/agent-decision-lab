"""Tests for TreasureGraph neutral world generation, distractor graphs, and anti-leakage safeguards."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from treasuregraph.generator import generate_world
from treasuregraph.map_generator import (
    compute_graph_statistics,
    generate_document_map,
    solve_topology_only,
)

FORBIDDEN_LEAK_WORDS = [
    "inicio",
    "missao",
    "checkpoint",
    "cofre",
    "terminal",
    "segredo",
    "tesouro",
    "start",
    "vault",
    "secret",
    "treasure",
    "mission",
]


class TestNeutralGraphGeneration(unittest.TestCase):
    """Test suite validating neutral identifiers, distractor graphs, and absence of leaks."""

    def test_neutral_identifiers_format(self):
        """Verifies that all files follow neutral doc_XXX.txt formatting."""
        for size in [5, 10, 20, 50, 100]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                w_dir = Path(tmp_dir) / "world"
                key = generate_world(seed=42, output_dir=w_dir, total_files=size, neutral_ids=True)

                files = sorted(list(w_dir.glob("*.txt")))
                self.assertEqual(len(files), size)
                for f in files:
                    self.assertRegex(f.name, r"^doc_\d{3}\.txt$")
                    # Ensure no subdirectories created
                    self.assertEqual(f.parent, w_dir)

    def test_absence_of_semantic_leakage_in_agent_map(self):
        """Verifies that node descriptions and identifiers never contain forbidden functional words."""
        for lang in ["pt-BR", "en"]:
            for size in [10, 100]:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    w_dir = Path(tmp_dir) / "world"
                    key = generate_world(
                        seed=42,
                        output_dir=w_dir,
                        language=lang,
                        total_files=size,
                        neutral_ids=True,
                    )

                    for desc_mode in ["short", "detailed"]:
                        doc_map = generate_document_map(key, description_mode=desc_mode)
                        agent_dict = doc_map.to_agent_dict(desc_mode)

                        for node in agent_dict["nodes"]:
                            nid = node["id"].lower()
                            desc = (node.get("description") or "").lower()
                            for forbidden in FORBIDDEN_LEAK_WORDS:
                                self.assertIsNone(
                                    re.search(r"\b" + re.escape(forbidden) + r"\b", nid),
                                    f"Leaked forbidden word '{forbidden}' in node ID '{nid}' (Lang: {lang})",
                                )
                                self.assertIsNone(
                                    re.search(r"\b" + re.escape(forbidden) + r"\b", desc),
                                    f"Leaked forbidden word '{forbidden}' in description '{desc}' (Lang: {lang})",
                                )

    def test_structural_richness_and_no_isolated_nodes(self):
        """Verifies distractor graph metrics: multiple components, 0 isolated nodes, cycles, and branches."""
        for size in [10, 100]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                w_dir = Path(tmp_dir) / "world"
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

                self.assertEqual(stats["node_count"], size)
                self.assertGreater(stats["edge_count"], size - 2)
                self.assertGreater(stats["connected_components_count"], 1, "Must have > 1 connected component")
                self.assertEqual(stats["isolated_nodes_count"], 0, "No isolated nodes permitted")
                self.assertGreater(stats["dead_ends_count"], 0, "Must have dead ends")
                self.assertGreater(stats["branching_nodes_count"], 0, "Must have branching nodes")
                self.assertGreater(stats["cycles_count"], 0, "Must have at least one cycle")

    def test_topological_sanity_ambiguity_solver(self):
        """Sanity test: An algorithm receiving ONLY topology and node IDs cannot uniquely deduce the path."""
        for size in [10, 100]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                w_dir = Path(tmp_dir) / "world"
                key = generate_world(
                    seed=42,
                    output_dir=w_dir,
                    language="pt-BR",
                    total_files=size,
                    neutral_ids=True,
                )

                doc_map = generate_document_map(key)
                agent_dict = doc_map.to_agent_dict()

                # Solver given only start node and topology
                candidate_paths = solve_topology_only(
                    agent_dict,
                    start_node_id=key["start_file"],
                    path_length=2,
                )

                # Crucial check: multiple paths of length 3 must exist from start node
                self.assertGreater(
                    len(candidate_paths),
                    1,
                    f"Topology must be ambiguous for N={size}. Found {len(candidate_paths)} candidate path(s).",
                )
                self.assertIn(key["solution_path"], candidate_paths)

                # Solver given no start node: total candidate paths across all components
                all_candidate_paths = solve_topology_only(agent_dict, start_node_id=None, path_length=2)
                self.assertGreater(len(all_candidate_paths), len(candidate_paths))

    def test_logical_verifiability_from_file_contents(self):
        """Verifies that an agent following text clues from doc_001.txt reliably reaches the secret code."""
        for lang in ["pt-BR", "en"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                w_dir = Path(tmp_dir) / "world"
                key = generate_world(
                    seed=42,
                    output_dir=w_dir,
                    language=lang,
                    total_files=10,
                    neutral_ids=True,
                )

                # 1. Read start file
                c1 = (w_dir / key["start_file"]).read_text(encoding="utf-8")
                pat1 = r"examine o documento (doc_\d{3}\.txt)" if lang == "pt-BR" else r"examine document (doc_\d{3}\.txt)"
                m1 = re.search(pat1, c1)
                self.assertIsNotNone(m1)
                step2 = m1.group(1)
                self.assertEqual(step2, key["solution_path"][1])

                # 2. Read step 2 file
                c2 = (w_dir / step2).read_text(encoding="utf-8")
                pat2 = r"autenticação está em (doc_\d{3}\.txt)" if lang == "pt-BR" else r"parameters is in (doc_\d{3}\.txt)"
                m2 = re.search(pat2, c2)
                self.assertIsNotNone(m2)
                step3 = m2.group(1)
                self.assertEqual(step3, key["solution_path"][2])

                # 3. Read step 3 file and extract secret code
                c3 = (w_dir / step3).read_text(encoding="utf-8")
                pat3 = r"(?:CÓDIGO SECRETO|SECRET CODE):\s*([A-Z0-9-]+)"
                m3 = re.search(pat3, c3)
                self.assertIsNotNone(m3)
                extracted_code = m3.group(1)
                self.assertEqual(extracted_code, key["secret_code"])

                # 4. Verify no distractor file contains the secret code
                for f in w_dir.glob("*.txt"):
                    if f.name != step3:
                        content = f.read_text(encoding="utf-8")
                        self.assertNotIn(key["secret_code"], content)

    def test_bilingual_and_seed_reproducibility(self):
        """Verifies deterministic output across calls with the same seed."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            w1 = Path(tmp1) / "world"
            w2 = Path(tmp2) / "world"

            key1 = generate_world(seed=123, output_dir=w1, total_files=20, neutral_ids=True)
            key2 = generate_world(seed=123, output_dir=w2, total_files=20, neutral_ids=True)

            self.assertEqual(key1["solution_path"], key2["solution_path"])
            self.assertEqual(key1["secret_code"], key2["secret_code"])
            self.assertEqual(key1["distractor_edges"], key2["distractor_edges"])

            for f in w1.glob("*.txt"):
                self.assertEqual(f.read_text(encoding="utf-8"), (w2 / f.name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
