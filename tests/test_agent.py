"""Tests for BaselineAgent using simulated LLMs without real API calls."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from treasuregraph.agent import BaselineAgent, safe_resolve_path
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

    def test_simulated_llm_completes_mission_and_logs_metrics(self):
        """Validates tool execution, code submission, and action metrics recording."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            key = generate_world(seed=42, output_dir=world_dir, language="en")

            # Sequence of actions to simulate
            plan_sequence = [
                ("list_directory", {"path": "."}),
                ("list_directory", {"path": "archive"}),
                ("read_file", {"path": "archive/mission_start.txt"}),
                ("read_file", {"path": "operations/checkpoint_record.txt"}),
                ("read_file", {"path": "vault/terminal_access.txt"}),
                ("submit_answer", {"code": key["secret_code"]}),
            ]
            call_idx = 0

            def mock_llm(messages: list[dict], tools: list[dict]) -> dict:
                nonlocal call_idx
                tool_name, tool_args = plan_sequence[call_idx]
                call_idx += 1
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": f"call_{call_idx}",
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(tool_args),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 150 + call_idx * 20,
                        "completion_tokens": 30,
                        "total_tokens": 180 + call_idx * 20,
                    },
                }

            agent = BaselineAgent(
                world_dir=world_dir,
                model="mock-gpt",
                max_steps=10,
                llm_client=mock_llm,
            )

            result = agent.run()

            self.assertEqual(result["status"], "submitted")
            self.assertEqual(result["submitted_code"], key["secret_code"])
            self.assertEqual(result["total_steps"], 6)

            # Check that every action has all required metric fields
            for item in result["trajectory"]:
                self.assertIn("step", item)
                self.assertIn("action", item)
                self.assertIn("arguments", item)
                self.assertIn("result", item)
                self.assertIn("status", item)
                self.assertEqual(item["status"], "success")
                self.assertIn("duration_seconds", item)
                self.assertGreaterEqual(item["duration_seconds"], 0.0)
                self.assertIn("tokens", item)
                self.assertGreater(item["tokens"]["total_tokens"], 0)
                self.assertIn("timestamp", item)

    def test_sandbox_blocks_agent_access_to_answer_key(self):
        """Verifies agent cannot read answer_key.json even if attempting path traversal."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            world_dir = Path(tmp_dir) / "world"
            key = generate_world(seed=42, output_dir=world_dir, language="en")
            answer_key_path = Path(tmp_dir) / "answer_key.json"
            self.assertTrue(answer_key_path.exists())

            # Simulated agent trying to read answer_key.json directly
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
