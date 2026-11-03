"""Tests for TreasureGraph world generator."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from treasuregraph.generator import SECTORS, generate_world


class TestTreasureGraphGenerator(unittest.TestCase):
    """Test suite for world generation, semantic clue properties, and multilingual support."""

    def resolve_next_step(self, clue_text: str, world_dir: Path, language: str) -> str:
        """Resolves direct or semantic clues in English or Portuguese."""
        # 1. Direct clue check
        direct_match = re.search(
            r"(?:Next clue \(Direct\):|Próxima pista \(Direta\):).*?'([^']+\.txt)'",
            clue_text,
        )
        if direct_match:
            return direct_match.group(1)

        # 2. Semantic clue check
        semantic_match = re.search(
            r"(?:Next clue \(Semantic\):\s*(?:Proceed to|Access)|Próxima pista \(Semântica\):\s*(?:Prossiga até|Acesse))\s+(.+?)\s+(?:and\s+(?:examine|inspect)\s+(?:the document detailing|the console record storing)|e\s+(?:examine|inspecione)\s+(?:o documento detalhando|o registro de console armazenando))\s+(.+?)\.",
            clue_text,
        )
        if not semantic_match:
            raise ValueError(f"Unable to parse clue text:\n{clue_text}")

        target_sector_desc = semantic_match.group(1).strip()
        target_purpose = semantic_match.group(2).strip()

        # Map descriptor to folder in current language
        matching_folders = [
            SECTORS[k][language]["folder"]
            for k in SECTORS
            if SECTORS[k][language]["descriptor"] == target_sector_desc
        ]
        if len(matching_folders) != 1:
            raise ValueError(f"Ambiguous or unknown sector descriptor: '{target_sector_desc}'")

        target_folder = matching_folders[0]
        folder_path = world_dir / target_folder

        # Within candidate folder, find matching file by purpose
        candidates = list(folder_path.glob("*.txt"))
        matching_files = []
        for cand in candidates:
            content = cand.read_text(encoding="utf-8")
            if target_purpose in content:
                matching_files.append(cand)

        if len(matching_files) != 1:
            raise ValueError(
                f"Expected exactly 1 matching file in '{target_folder}' for purpose '{target_purpose}', found {len(matching_files)}"
            )

        return f"{target_folder}/{matching_files[0].name}"

    def extract_secret_code(self, vault_text: str) -> str:
        """Extracts the secret code from the final vault terminal file."""
        match = re.search(r"(?:SECRET CODE|CÓDIGO SECRETO):\s*([A-Z0-9-]+)", vault_text)
        if match:
            return match.group(1)
        raise ValueError(f"Secret code not found in terminal text:\n{vault_text}")

    def test_environment_structure_both_languages(self):
        """Verifies 5 .txt files across 3 folders for both 'en' and 'pt-BR'."""
        for lang in ["en", "pt-BR"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                world_dir = tmp_path / "world"

                key = generate_world(seed=42, output_dir=world_dir, language=lang)

                subdirs = [p for p in world_dir.iterdir() if p.is_dir()]
                self.assertEqual(len(subdirs), 3, f"Expected 3 folders for {lang}.")

                txt_files = list(world_dir.glob("*/*.txt"))
                self.assertEqual(len(txt_files), 5, f"Expected 5 .txt files for {lang}.")

                counts = [len(list(d.glob("*.txt"))) for d in subdirs]
                self.assertCountEqual(counts, [2, 2, 1])

                key_file = tmp_path / "answer_key.json"
                self.assertTrue(key_file.exists())
                self.assertFalse((world_dir / "answer_key.json").exists())

                saved_key = json.loads(key_file.read_text(encoding="utf-8"))
                self.assertEqual(saved_key["secret_code"], key["secret_code"])
                self.assertEqual(saved_key["language"], lang)

    def test_default_language_is_en(self):
        """Verifies backwards compatibility when language is omitted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            key = generate_world(seed=42, output_dir=world_dir)
            self.assertEqual(key["language"], "en")
            start_content = (world_dir / key["start_file"]).read_text(encoding="utf-8")
            self.assertIn("AGENT BRIEFING", start_content)

    def test_semantic_clues_never_reveal_exact_names(self):
        """Verifies semantic clues never leak folder or file names in en or pt-BR."""
        for lang in ["en", "pt-BR"]:
            for seed in range(30):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    world_dir = Path(tmp_dir) / "world"
                    key = generate_world(seed=seed, output_dir=world_dir, language=lang)

                    for step in key["clue_chain"]:
                        if step["type"] == "semantic":
                            source_file = world_dir / step["file"]
                            content = source_file.read_text(encoding="utf-8")

                            target_path = step["leads_to"]
                            target_folder, target_filename = target_path.split("/")
                            target_stem = target_filename.replace(".txt", "")

                            clue_match = re.search(
                                r"(?:Next clue \(Semantic\):|Próxima pista \(Semântica\):).+",
                                content,
                            )
                            self.assertIsNotNone(clue_match)
                            clue_line = clue_match.group(0).lower()

                            self.assertNotIn(
                                target_folder.lower(),
                                clue_line,
                                f"Lang {lang}, Seed {seed}: Folder name '{target_folder}' leaked in clue!",
                            )
                            self.assertNotIn(
                                target_filename.lower(),
                                clue_line,
                                f"Lang {lang}, Seed {seed}: Filename '{target_filename}' leaked in clue!",
                            )
                            self.assertNotIn(
                                target_stem.lower(),
                                clue_line,
                                f"Lang {lang}, Seed {seed}: File stem '{target_stem}' leaked in clue!",
                            )

    def test_following_clues_finds_treasure_both_languages(self):
        """Simulates clue-following solver in both English and Portuguese across seeds."""
        for lang in ["en", "pt-BR"]:
            for seed in [42, 123, 777]:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    world_dir = Path(tmp_dir) / "world"
                    key = generate_world(seed=seed, output_dir=world_dir, language=lang)

                    # 1. Start file
                    start_path = key["start_file"]
                    step1_text = (world_dir / start_path).read_text(encoding="utf-8")
                    step2_path = self.resolve_next_step(step1_text, world_dir, lang)

                    # 2. Step 2 file
                    step2_text = (world_dir / step2_path).read_text(encoding="utf-8")
                    step3_path = self.resolve_next_step(step2_text, world_dir, lang)

                    # 3. Final vault
                    step3_text = (world_dir / step3_path).read_text(encoding="utf-8")
                    found_code = self.extract_secret_code(step3_text)

                    self.assertEqual(
                        [start_path, step2_path, step3_path],
                        key["solution_path"],
                        f"Lang {lang}, Seed {seed}: Solution path mismatch.",
                    )
                    self.assertEqual(
                        found_code,
                        key["secret_code"],
                        f"Lang {lang}, Seed {seed}: Discovered code mismatch.",
                    )

    def test_reproducibility_both_languages(self):
        """Verifies same seed generates byte-identical outputs in both languages."""
        for lang in ["en", "pt-BR"]:
            with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
                dir1 = Path(tmp1) / "world"
                dir2 = Path(tmp2) / "world"

                key1 = generate_world(seed=500, output_dir=dir1, language=lang)
                key2 = generate_world(seed=500, output_dir=dir2, language=lang)

                self.assertEqual(key1, key2)

                files1 = sorted([str(p.relative_to(dir1)) for p in dir1.glob("*/*.txt")])
                files2 = sorted([str(p.relative_to(dir2)) for p in dir2.glob("*/*.txt")])
                self.assertEqual(files1, files2)

                for rel_file in files1:
                    c1 = (dir1 / rel_file).read_text(encoding="utf-8")
                    c2 = (dir2 / rel_file).read_text(encoding="utf-8")
                    self.assertEqual(c1, c2)

    def test_decoy_is_logically_solvable_both_languages(self):
        """Verifies decoy notice has revocation advisory in both languages."""
        for lang in ["en", "pt-BR"]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                world_dir = Path(tmp_dir) / "world"
                key = generate_world(seed=42, output_dir=world_dir, language=lang)

                decoy_meta = next(d for d in key["distractors"] if d["type"] == "decoy")
                content = (world_dir / decoy_meta["file"]).read_text(encoding="utf-8")

                if lang == "pt-BR":
                    self.assertIn("REVOGADO / NÃO VERIFICADO", content)
                else:
                    self.assertIn("REVOKED / UNVERIFIED", content)

                self.assertIn(decoy_meta["fake_code"], content)
                self.assertNotEqual(decoy_meta["fake_code"], key["secret_code"])


if __name__ == "__main__":
    unittest.main()
