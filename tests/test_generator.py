"""Tests for TreasureGraph world generator."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from treasuregraph.generator import generate_world


class TestTreasureGraphGenerator(unittest.TestCase):
    """Test suite for world generation, reproducibility, and clue chain resolution."""

    def resolve_next_step(self, clue_text: str) -> str:
        """Solves direct or semantic clues to determine the next relative file path."""
        # Check direct clue format
        direct_match = re.search(r"Next clue \(Direct\):.*?'([^']+\.txt)'", clue_text)
        if direct_match:
            return direct_match.group(1)

        # Check semantic clue format
        semantic_match = re.search(r"folder:\s*'([^']+)'.*?'([^']+\.txt)'", clue_text)
        if semantic_match:
            folder, filename = semantic_match.group(1), semantic_match.group(2)
            return f"{folder}/{filename}"

        raise ValueError(f"Unable to resolve next path from clue:\n{clue_text}")

    def extract_secret_code(self, vault_text: str) -> str:
        """Extracts the secret code from the final vault terminal file."""
        match = re.search(r"SECRET CODE:\s*(TREASURE-[A-Z0-9-]+)", vault_text)
        if match:
            return match.group(1)
        raise ValueError(f"Secret code not found in terminal text:\n{vault_text}")

    def test_environment_structure(self):
        """Verifies 5 .txt files across 3 folders, and answer_key.json outside output_dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            world_dir = tmp_path / "world"

            key = generate_world(seed=42, output_dir=world_dir)

            # Check folders in world_dir
            subdirs = [p for p in world_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(subdirs), 3, "Expected exactly 3 folders in world_dir.")

            # Check files in world_dir
            txt_files = list(world_dir.glob("*/*.txt"))
            self.assertEqual(len(txt_files), 5, "Expected exactly 5 .txt files in world.")

            # Verify distribution: all folders have at least 1 file
            counts = [len(list(d.glob("*.txt"))) for d in subdirs]
            self.assertCountEqual(counts, [2, 2, 1], "Expected distribution of files to be [2, 2, 1].")

            # Check answer_key.json exists outside world_dir
            key_file = tmp_path / "answer_key.json"
            self.assertTrue(key_file.exists(), "answer_key.json should be saved outside world_dir.")
            self.assertFalse(
                (world_dir / "answer_key.json").exists(),
                "answer_key.json must not be accessible inside world_dir.",
            )

            # Verify key matches return value
            saved_key = json.loads(key_file.read_text(encoding="utf-8"))
            self.assertEqual(saved_key["secret_code"], key["secret_code"])

    def test_following_clues_finds_treasure(self):
        """Simulates an agent following the chain of clues to discover the secret code."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            world_dir = tmp_path / "world"

            key = generate_world(seed=123, output_dir=world_dir)

            # 1. Start at the briefing file
            start_rel_path = key["start_file"]
            self.assertTrue((world_dir / start_rel_path).exists())

            step1_text = (world_dir / start_rel_path).read_text(encoding="utf-8")
            step2_rel_path = self.resolve_next_step(step1_text)

            # 2. Inspect step 2 file
            self.assertTrue((world_dir / step2_rel_path).exists())
            step2_text = (world_dir / step2_rel_path).read_text(encoding="utf-8")
            step3_rel_path = self.resolve_next_step(step2_text)

            # 3. Inspect final vault terminal
            self.assertTrue((world_dir / step3_rel_path).exists())
            step3_text = (world_dir / step3_rel_path).read_text(encoding="utf-8")
            found_code = self.extract_secret_code(step3_text)

            # Verify entire path and discovered secret code
            expected_path = key["solution_path"]
            self.assertEqual([start_rel_path, step2_rel_path, step3_rel_path], expected_path)
            self.assertEqual(found_code, key["secret_code"])

    def test_reproducibility(self):
        """Verifies that the same seed produces identical environments and answer keys."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            dir1 = Path(tmp1) / "world"
            dir2 = Path(tmp2) / "world"

            key1 = generate_world(seed=777, output_dir=dir1)
            key2 = generate_world(seed=777, output_dir=dir2)

            self.assertEqual(key1, key2)

            # Compare all generated files
            files1 = sorted([str(p.relative_to(dir1)) for p in dir1.glob("*/*.txt")])
            files2 = sorted([str(p.relative_to(dir2)) for p in dir2.glob("*/*.txt")])
            self.assertEqual(files1, files2)

            for rel_file in files1:
                content1 = (dir1 / rel_file).read_text(encoding="utf-8")
                content2 = (dir2 / rel_file).read_text(encoding="utf-8")
                self.assertEqual(content1, content2)

    def test_decoy_is_logically_solvable(self):
        """Verifies that the decoy file contains clear logical indicators of invalidity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            key = generate_world(seed=42, output_dir=world_dir)

            decoy_meta = next(d for d in key["distractors"] if d["type"] == "decoy")
            decoy_path = world_dir / decoy_meta["file"]
            self.assertTrue(decoy_path.exists())

            content = decoy_path.read_text(encoding="utf-8")
            self.assertIn("REVOKED / UNVERIFIED", content)
            self.assertIn(decoy_meta["fake_code"], content)
            self.assertNotEqual(decoy_meta["fake_code"], key["secret_code"])


if __name__ == "__main__":
    unittest.main()
