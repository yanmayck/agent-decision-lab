"""Baseline Agent for TreasureGraph.

Provides a minimal, natural ReAct-style agent with sandboxed environment tools:
- list_directory
- read_file
- submit_answer

Records detailed action metrics (arguments, results, tokens, duration) without
pre-planning, optimization, or prior knowledge of the environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import urllib.error
import urllib.request

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List all files and subdirectories in a directory relative to the environment root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to inspect. Use '.' or '' for root.",
                        "default": ".",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the complete text content of a file relative to the environment root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file to read (e.g. 'archive/briefing.txt').",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the secret treasure code once discovered to finish the mission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The secret code discovered in the environment.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]

DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous AI operative exploring a sandboxed facility.\n"
    "Your objective is to find the hidden secret code and submit it using submit_answer(code=...).\n"
    "You have tools to list directories and read files. Explore carefully.\n"
    "Once you discover the secret code, call submit_answer immediately to complete the mission."
)


def safe_resolve_path(world_dir: Path, requested_path: str | Path) -> Path:
    """Resolves requested_path securely, preventing sandbox escapes and symlink traversal.

    Args:
        world_dir: Canonical root directory for the sandbox.
        requested_path: User/agent-provided path.

    Returns:
        Resolved canonical Path inside world_dir.

    Raises:
        PermissionError: If path attempts to escape world_dir or symlink points outside.
    """
    canonical_root = world_dir.resolve()
    raw_str = str(requested_path).strip()

    # Reject empty or root designations safely
    if not raw_str or raw_str in (".", "./", ".\\"):
        return canonical_root

    # Guard against drive letters or absolute roots
    target = Path(raw_str)
    if target.is_absolute() or (len(raw_str) > 1 and raw_str[1] == ":"):
        resolved = target.resolve()
    else:
        resolved = (canonical_root / target).resolve()

    # Ensure resolved path is strictly within canonical_root
    try:
        resolved.relative_to(canonical_root)
    except ValueError:
        raise PermissionError(
            f"Access denied: '{requested_path}' escapes the authorized sandbox directory."
        )

    # Check symlinks (if symlink exists on disk, its real destination must also be inside root)
    if resolved.is_symlink():
        real_dest = resolved.resolve(strict=False)
        try:
            real_dest.relative_to(canonical_root)
        except ValueError:
            raise PermissionError(
                f"Access denied: Symlink '{requested_path}' points outside authorized sandbox."
            )

    return resolved


class BaselineAgent:
    """Baseline Agent without planning, reflection, or prior knowledge."""

    def __init__(
        self,
        world_dir: str | Path,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        max_steps: int = 15,
        llm_client: Callable[..., dict[str, Any]] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Initializes the baseline agent.

        Args:
            world_dir: Path to the environment directory.
            model: Model name. Falls back to TREASURE_MODEL or OPENAI_MODEL env var.
            api_key: API Key. Falls back to TREASURE_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.
            api_base: Base URL for OpenAI-compatible endpoint. Falls back to TREASURE_API_BASE.
            max_steps: Maximum number of tool actions allowed.
            llm_client: Optional callable to override HTTP requests (useful for mocking/testing).
            system_prompt: Baseline instructions for the agent.
        """
        self.world_dir = Path(world_dir).resolve()
        if not self.world_dir.exists():
            raise FileNotFoundError(f"Environment directory '{self.world_dir}' does not exist.")

        self.model = (
            model
            or os.getenv("TREASURE_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        self.api_key = (
            api_key
            or os.getenv("TREASURE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or ""
        )
        self.api_base = (
            api_base
            or os.getenv("TREASURE_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.max_steps = max_steps
        self.llm_client = llm_client
        self.system_prompt = system_prompt

        # Trajectory log
        self.trajectory: list[dict[str, Any]] = []
        self.submitted_code: str | None = None
        self.finished = False

    # -------------------------------------------------------------------------
    # Environment Tools (Strictly Sandboxed)
    # -------------------------------------------------------------------------

    def list_directory(self, path: str = ".") -> str:
        """Lists entries in a directory relative to the world root."""
        try:
            target = safe_resolve_path(self.world_dir, path)
            if not target.exists():
                return f"Error: Directory '{path}' does not exist."
            if not target.is_dir():
                return f"Error: Path '{path}' is a file, not a directory."

            entries = []
            for item in sorted(target.iterdir()):
                rel = item.relative_to(self.world_dir)
                kind = "directory" if item.is_dir() else "file"
                entries.append(f"{rel.as_posix()}{'/' if item.is_dir() else ''} ({kind})")

            if not entries:
                return "Directory is empty."
            return "\n".join(entries)
        except PermissionError as e:
            return f"Security Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def read_file(self, path: str) -> str:
        """Reads text content of a file inside the sandbox."""
        try:
            target = safe_resolve_path(self.world_dir, path)
            if not target.exists():
                return f"Error: File '{path}' does not exist."
            if not target.is_file():
                return f"Error: Path '{path}' is a directory, not a file."

            return target.read_text(encoding="utf-8")
        except PermissionError as e:
            return f"Security Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def submit_answer(self, code: str) -> str:
        """Submits the discovered treasure code."""
        cleaned = code.strip()
        self.submitted_code = cleaned
        self.finished = True
        return f"Code '{cleaned}' submitted successfully. Mission concluded."

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Executes one of the three tools and returns (result_text, is_error)."""
        if tool_name == "list_directory":
            path_arg = arguments.get("path", ".")
            result = self.list_directory(path_arg)
            is_error = result.startswith("Error") or result.startswith("Security Error")
            return result, is_error
        elif tool_name == "read_file":
            path_arg = arguments.get("path", "")
            result = self.read_file(path_arg)
            is_error = result.startswith("Error") or result.startswith("Security Error")
            return result, is_error
        elif tool_name == "submit_answer":
            code_arg = arguments.get("code", "")
            result = self.submit_answer(code_arg)
            return result, False
        else:
            return f"Error: Unknown tool '{tool_name}'. Available: list_directory, read_file, submit_answer.", True

    # -------------------------------------------------------------------------
    # LLM Interaction
    # -------------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Calls the configured LLM or custom simulated client."""
        if self.llm_client is not None:
            return self.llm_client(messages, TOOLS_SCHEMA)

        if not self.api_key:
            raise ValueError(
                "API key not configured. Set TREASURE_API_KEY, OPENAI_API_KEY, "
                "or pass llm_client for simulated execution."
            )

        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "tool_choice": "auto",
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM API request failed with status {e.code}: {err_body}")

    # -------------------------------------------------------------------------
    # Execution Loop
    # -------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Runs the baseline agent loop until submit_answer or max_steps reached.

        Returns:
            Dictionary containing trajectory, metrics, submitted_code, and status.
        """
        self.trajectory = []
        self.submitted_code = None
        self.finished = False

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": "Begin your investigation. Explore the environment to find and submit the secret code.",
            },
        ]

        step_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        while step_count < self.max_steps and not self.finished:
            # Call LLM
            llm_response = self._call_llm(messages)

            # Extract token usage
            usage = llm_response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            choice = llm_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            # Append assistant message to context
            messages.append(message)

            if not tool_calls:
                # Agent answered with text without tools; prompt it to use tools
                messages.append(
                    {
                        "role": "user",
                        "content": "Please use list_directory, read_file, or submit_answer to take action.",
                    }
                )
                step_count += 1
                continue

            for tool_call in tool_calls:
                if self.finished or step_count >= self.max_steps:
                    break

                step_count += 1
                call_id = tool_call.get("id", f"call_{step_count}")
                func_meta = tool_call.get("function", {})
                tool_name = func_meta.get("name", "")

                raw_args = func_meta.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {"raw": raw_args}
                else:
                    args = raw_args

                # Execute tool with timing
                t0 = time.perf_counter()
                result_text, is_error = self.execute_tool(tool_name, args)
                duration = time.perf_counter() - t0

                # Record action metric
                action_record = {
                    "step": step_count,
                    "action": tool_name,
                    "arguments": args,
                    "result": result_text,
                    "status": "error" if is_error else "success",
                    "duration_seconds": round(duration, 5),
                    "tokens": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.trajectory.append(action_record)

                # Feed tool result back to LLM context
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": result_text,
                    }
                )

        if self.finished:
            status = "submitted"
        elif step_count >= self.max_steps:
            status = "max_steps_reached"
        else:
            status = "stopped"

        total_duration = sum(item["duration_seconds"] for item in self.trajectory)

        return {
            "status": status,
            "submitted_code": self.submitted_code,
            "total_steps": len(self.trajectory),
            "total_duration_seconds": round(total_duration, 4),
            "total_tokens": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "trajectory": self.trajectory,
        }
