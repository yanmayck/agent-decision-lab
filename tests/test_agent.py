"""Tests for BaselineAgent using simulated LLMs without real API calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from treasuregraph.agent import (
    GEMINI_OPENAI_ENDPOINT,
    BaselineAgent,
    extract_thought_signature,
    extract_thought_summary,
    load_dotenv,
    safe_resolve_path,
)
from treasuregraph.generator import generate_world


class TestBaselineAgent(unittest.TestCase):
    """Test suite for BaselineAgent, sandboxing, and simulated execution."""

    def test_safe_resolve_path_security(self):
        """Verifies sandbox path traversal prevention."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "sandbox"
            base_dir.mkdir()
            secret_file = Path(tmp_dir) / "answer_key.json"
            secret_file.write_text("SECRET GABARITO", encoding="utf-8")

            # Normal path inside
            valid = safe_resolve_path(base_dir, "archive/file.txt")
            self.assertEqual(valid, (base_dir / "archive/file.txt").resolve())

            # Traversal attempts
            with self.assertRaises(PermissionError):
                safe_resolve_path(base_dir, "../answer_key.json")

            with self.assertRaises(PermissionError):
                safe_resolve_path(base_dir, "..\\answer_key.json")

            with self.assertRaises(PermissionError):
                safe_resolve_path(base_dir, "../../")

            with self.assertRaises(PermissionError):
                safe_resolve_path(base_dir, "C:/Windows/System32")

    def test_google_gemini_endpoint_auto_detection(self):
        """Verifies that Gemini models automatically configure Google's API endpoint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            world_dir.mkdir()

            agent = BaselineAgent(
                world_dir=world_dir,
                model="gemini-3.5-flash-lite",
                api_key="fake-test-key",
                llm_client=lambda msgs, tools: {},
            )
            self.assertEqual(agent.api_base, GEMINI_OPENAI_ENDPOINT)
            self.assertEqual(agent.model, "gemini-3.5-flash-lite")

    def test_dotenv_loader(self):
        """Verifies that load_dotenv reads variables safely."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text(
                "TEST_TG_VAR=hello_world\n# Comment\nINVALID_LINE\nTEST_TG_QUOTED='quoted_value'\n",
                encoding="utf-8",
            )
            load_dotenv(env_file)
            self.assertEqual(os.environ.get("TEST_TG_VAR"), "hello_world")
            self.assertEqual(os.environ.get("TEST_TG_QUOTED"), "quoted_value")

    def test_extract_thought_summary_and_signature(self):
        """Validates extraction of thought summaries and thought signatures across formats."""
        # 1. OpenAI reasoning_content format
        msg1 = {
            "role": "assistant",
            "reasoning_content": "Pensando: devo inspecionar o arquivo antes de submeter.",
            "tool_calls": [],
        }
        self.assertEqual(
            extract_thought_summary(msg1),
            "Pensando: devo inspecionar o arquivo antes de submeter.",
        )
        self.assertIsNone(extract_thought_signature(msg1))

        # 2. Google extra_content format with thought and signature
        msg2 = {
            "role": "assistant",
            "extra_content": {
                "google": {
                    "thought": "Analisando as pastas disponíveis...",
                    "thought_signature": "cryptographic_sig_xyz",
                }
            },
        }
        self.assertEqual(
            extract_thought_summary(msg2),
            "Analisando as pastas disponíveis...",
        )
        self.assertEqual(extract_thought_signature(msg2), "cryptographic_sig_xyz")

        # 3. Message with ONLY thought_signature (no plain text thoughts, like gemini-3.5-flash-lite)
        msg3 = {
            "role": "assistant",
            "extra_content": {
                "google": {
                    "thought_signature": "signature_only_abc",
                }
            },
        }
        self.assertIsNone(extract_thought_summary(msg3))
        self.assertEqual(extract_thought_signature(msg3), "signature_only_abc")

    def test_simulated_llm_with_thought_summary_and_signature_preservation(self):
        """Verifies that thought summaries and signatures are recorded and preserved across turns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            key = generate_world(seed=42, output_dir=world_dir, language="pt-BR")

            sent_contexts = []

            def mock_thinking_llm(messages: list[dict], tools: list[dict]) -> dict:
                sent_contexts.append(list(messages))
                turn = len(sent_contexts)
                if turn == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "reasoning_content": "Passo 1: Vou listar o diretório raiz para mapear o ambiente.",
                                    "extra_content": {
                                        "google": {"thought_signature": "sig_turn_1"}
                                    },
                                    "tool_calls": [
                                        {
                                            "id": "c1",
                                            "type": "function",
                                            "function": {
                                                "name": "list_directory",
                                                "arguments": json.dumps({"path": "."}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 170},
                    }
                else:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "reasoning_content": "Passo 2: Código descoberto, submetendo agora.",
                                    "extra_content": {
                                        "google": {"thought_signature": "sig_turn_2"}
                                    },
                                    "tool_calls": [
                                        {
                                            "id": "c2",
                                            "type": "function",
                                            "function": {
                                                "name": "submit_answer",
                                                "arguments": json.dumps({"code": key["secret_code"]}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 150, "completion_tokens": 10, "total_tokens": 200},
                    }

            agent = BaselineAgent(
                world_dir=world_dir,
                model="gemini-3.5-flash-lite",
                max_steps=5,
                llm_client=mock_thinking_llm,
                reasoning_effort="high",
            )

            res = agent.run()

            self.assertEqual(res["status"], "submitted")
            self.assertEqual(res["total_steps"], 2)

            step1 = res["trajectory"][0]
            self.assertEqual(
                step1["thought_summary"],
                "Passo 1: Vou listar o diretório raiz para mapear o ambiente.",
            )
            self.assertEqual(step1["observable_decision"], {"action": "list_directory", "arguments": {"path": "."}})
            self.assertEqual(step1["thought_signature"], "sig_turn_1")
            self.assertEqual(step1["tokens"]["reasoning_tokens"], 50)  # 170 - (100 + 20) = 50

            # Verify that thought_signature was preserved in the history sent to turn 2
            turn2_messages = sent_contexts[1]
            assistant_msg_turn1 = next(m for m in turn2_messages if m.get("role") == "assistant")
            self.assertIn("extra_content", assistant_msg_turn1)
            self.assertEqual(
                assistant_msg_turn1["extra_content"]["google"]["thought_signature"],
                "sig_turn_1",
            )

    def test_thought_summary_is_null_when_api_omits_it(self):
        """Verifies that thought_summary is strictly null when API does not provide plain text thoughts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            generate_world(seed=42, output_dir=world_dir)

            def mock_gemini_without_text_thoughts(messages: list[dict], tools: list[dict]) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "extra_content": {
                                    "google": {"thought_signature": "opaque_cryptographic_signature"}
                                },
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_answer",
                                            "arguments": json.dumps({"code": "TEST-CODE"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 150},
                }

            agent = BaselineAgent(
                world_dir=world_dir,
                llm_client=mock_gemini_without_text_thoughts,
                max_steps=2,
            )

            res = agent.run()
            step1 = res["trajectory"][0]
            # Must be None, never hallucinated
            self.assertIsNone(step1["thought_summary"])
            self.assertEqual(step1["thought_signature"], "opaque_cryptographic_signature")
            self.assertEqual(step1["observable_decision"]["action"], "submit_answer")
            self.assertEqual(step1["tokens"]["reasoning_tokens"], 55)

    def test_sandbox_blocks_agent_access_to_answer_key(self):
        """Verifies agent cannot read answer_key.json even if attempting path traversal."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            key = generate_world(seed=42, output_dir=world_dir, language="en")
            answer_key_path = Path(tmp_dir) / "answer_key.json"
            self.assertTrue(answer_key_path.exists())

            attempt_idx = 0

            def mock_attacker_llm(messages: list[dict], tools: list[dict]) -> dict:
                nonlocal attempt_idx
                if attempt_idx == 0:
                    attempt_idx += 1
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": "exploit_1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": json.dumps({"path": "../answer_key.json"}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                    }
                else:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": "finish",
                                            "type": "function",
                                            "function": {
                                                "name": "submit_answer",
                                                "arguments": json.dumps({"code": "GAVE_UP"}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 120, "completion_tokens": 10, "total_tokens": 130},
                    }

            agent = BaselineAgent(
                world_dir=world_dir,
                llm_client=mock_attacker_llm,
                max_steps=5,
            )

            res = agent.run()
            first_action = res["trajectory"][0]
            self.assertEqual(first_action["action"], "read_file")
            self.assertEqual(first_action["status"], "error")
            self.assertIn("Security Error", first_action["result"])
            self.assertNotIn("TREASURE", first_action["result"])

    def test_max_steps_enforcement(self):
        """Verifies that the agent halts when max_steps is reached."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            generate_world(seed=42, output_dir=world_dir)

            def mock_loop_llm(messages: list[dict], tools: list[dict]) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "loop_call",
                                        "type": "function",
                                        "function": {
                                            "name": "list_directory",
                                            "arguments": json.dumps({"path": "."}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                }

            agent = BaselineAgent(
                world_dir=world_dir,
                llm_client=mock_loop_llm,
                max_steps=4,
            )

            res = agent.run()
            self.assertEqual(res["status"], "max_steps_reached")
            self.assertEqual(res["total_steps"], 4)
            self.assertIsNone(res["submitted_code"])


if __name__ == "__main__":
    unittest.main()
