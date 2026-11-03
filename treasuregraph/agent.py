"""Baseline Agent for TreasureGraph.

Provides a minimal, natural ReAct-style agent with sandboxed environment tools:
- list_directory
- read_file
- submit_answer

Supports Google AI (Gemini) natively via its OpenAI-compatible endpoint, as well
as OpenAI and compatible local/remote models. Zero external dependencies.
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

from treasuregraph.instrumentation import (
    PricingConfig,
    ResourceTracker,
    estimate_tokens,
)

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
                        "description": "Relative path to the file to read (e.g. 'arquivo/inicio_missao.txt').",
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

GEMINI_OPENAI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai"


def extract_thought_summary(message_or_candidate: dict[str, Any]) -> str | None:
    """Extracts plain-text thought summary if provided by the model/endpoint.

    Returns None (null) if the API does not expose thought text. Never synthesizes thoughts.
    """
    if not isinstance(message_or_candidate, dict):
        return None

    # 1. Standard OpenAI reasoning_content (DeepSeek, o-series, newer formats)
    if "reasoning_content" in message_or_candidate and message_or_candidate["reasoning_content"]:
        return str(message_or_candidate["reasoning_content"]).strip() or None

    # 2. Direct thought / thoughts field in message
    if "thought" in message_or_candidate and isinstance(message_or_candidate["thought"], str):
        return message_or_candidate["thought"].strip() or None
    if "thoughts" in message_or_candidate and isinstance(message_or_candidate["thoughts"], str):
        return message_or_candidate["thoughts"].strip() or None

    # 3. Google extra_content in OpenAI-compatible endpoint
    extra = message_or_candidate.get("extra_content", {})
    if isinstance(extra, dict):
        google_data = extra.get("google", {})
        if isinstance(google_data, dict):
            if "thought" in google_data and isinstance(google_data["thought"], str):
                return google_data["thought"].strip() or None
            if "thoughts" in google_data and isinstance(google_data["thoughts"], str):
                return google_data["thoughts"].strip() or None

    # 4. Native Gemini parts with thought=True
    content = message_or_candidate.get("content", {})
    if isinstance(content, dict):
        parts = content.get("parts", [])
        if isinstance(parts, list):
            thought_texts = []
            for p in parts:
                if isinstance(p, dict) and p.get("thought") is True and p.get("text"):
                    thought_texts.append(p["text"])
            if thought_texts:
                return "\n".join(thought_texts).strip()

    return None


def extract_thought_signature(message_or_candidate: dict[str, Any]) -> str | None:
    """Extracts cryptographic thought signature to preserve context across turns."""
    if not isinstance(message_or_candidate, dict):
        return None

    # OpenAI-compatible Google format
    extra = message_or_candidate.get("extra_content", {})
    if isinstance(extra, dict):
        google_data = extra.get("google", {})
        if isinstance(google_data, dict) and "thought_signature" in google_data:
            return google_data["thought_signature"]

    # Native Gemini format
    content = message_or_candidate.get("content", {})
    if isinstance(content, dict):
        parts = content.get("parts", [])
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and p.get("thoughtSignature"):
                    return p["thoughtSignature"]

    return None


def load_dotenv(dotenv_path: Path | str = ".env") -> None:
    """Loads key-value pairs from .env into os.environ without overriding existing vars."""
    env_file = Path(dotenv_path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


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
        reasoning_effort: str | None = None,
        include_thoughts: bool | None = None,
        read_cost: int = 0,
        resource_tracker: ResourceTracker | None = None,
        pricing_config: PricingConfig | None = None,
        document_map: Any = None,
    ) -> None:
        """Initializes the baseline agent.

        Args:
            world_dir: Path to the environment directory.
            model: Model name. Falls back to TREASURE_MODEL, GEMINI_MODEL, or 'gemini-2.0-flash'.
            api_key: API Key. Falls back to GEMINI_API_KEY, GOOGLE_API_KEY, TREASURE_API_KEY, or OPENAI_API_KEY.
            api_base: Base URL. Auto-configures for Google AI (Gemini) if gemini model/key is detected.
            max_steps: Maximum number of tool actions allowed.
            llm_client: Optional callable to override HTTP requests (useful for mocking/testing).
            system_prompt: Baseline instructions for the agent.
            reasoning_effort: Optional reasoning effort level ('low', 'medium', 'high').
            include_thoughts: Optional boolean to request thought summaries from models that support it.
            read_cost: Optional cost per read_file action presented to the agent.
            resource_tracker: Optional ResourceTracker instance for recording metrics.
            pricing_config: Optional PricingConfig for USD cost estimation.
            document_map: Optional DocumentGraph or sanitized map dict for topological exploration.
        """
        # Load local .env if available
        load_dotenv()

        self.world_dir = Path(world_dir).resolve()
        if not self.world_dir.exists():
            raise FileNotFoundError(f"Environment directory '{self.world_dir}' does not exist.")

        self.pricing_config = pricing_config
        self.resource_tracker = resource_tracker or ResourceTracker(pricing=self.pricing_config)
        self.document_map = document_map

        self.model = (
            model
            or os.getenv("TREASURE_MODEL")
            or os.getenv("GEMINI_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gemini-2.0-flash"
        )

        self.api_key = (
            api_key
            or os.getenv("TREASURE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )

        # Automatic endpoint routing: Google AI endpoint for Gemini, OpenAI endpoint otherwise
        if api_base:
            self.api_base = api_base
        elif os.getenv("TREASURE_API_BASE"):
            self.api_base = os.environ["TREASURE_API_BASE"]
        elif os.getenv("OPENAI_BASE_URL"):
            self.api_base = os.environ["OPENAI_BASE_URL"]
        elif (
            self.model.lower().startswith("gemini")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        ):
            self.api_base = GEMINI_OPENAI_ENDPOINT
        else:
            self.api_base = "https://api.openai.com/v1"

        self.max_steps = max_steps
        self.llm_client = llm_client
        self.read_cost = read_cost

        # Configure system prompt and tools schema with cost information when applicable
        if self.document_map is not None and system_prompt == DEFAULT_SYSTEM_PROMPT:
            self.system_prompt = (
                "You are an autonomous AI operative exploring a sandboxed facility.\n"
                "Your objective is to find the hidden secret code and submit it using submit_answer(code=...).\n"
                "You have tools to inspect the document map (get_map), list directories, and read files. Explore carefully.\n"
                "Once you discover the secret code, call submit_answer immediately to complete the mission."
            )
        else:
            self.system_prompt = system_prompt

        if self.read_cost > 0:
            cost_directive = f"\nNota de custo operacional: Cada chamada de read_file possui um custo explícito de {self.read_cost} unidade(s). Minimize custos desnecessários."
            self.system_prompt += cost_directive

        self.tools_schema = json.loads(json.dumps(TOOLS_SCHEMA))
        if self.document_map is not None:
            self.tools_schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_map",
                        "description": "Retrieve the topological network map of documents and known directed connections between them.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            )

        if self.read_cost > 0:
            for tool_item in self.tools_schema:
                if tool_item.get("function", {}).get("name") == "read_file":
                    tool_item["function"]["description"] += f" (Custo operacional: {self.read_cost} unidade(s) por leitura)."

        self.reasoning_effort = (
            reasoning_effort or os.getenv("TREASURE_REASONING_EFFORT")
        )
        self.include_thoughts = (
            include_thoughts
            if include_thoughts is not None
            else os.getenv("TREASURE_INCLUDE_THOUGHTS", "false").lower() in ("true", "1", "yes")
        )

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

    def get_map(self) -> str:
        """Returns the sanitized topological map of documents."""
        if self.document_map is None:
            return "Error: No topological map available in this environment."
        if hasattr(self.document_map, "to_agent_dict"):
            data = self.document_map.to_agent_dict()
        else:
            data = self.document_map
        return json.dumps(data, indent=2, ensure_ascii=False)

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Executes one of the available tools and returns (result_text, is_error)."""
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
        elif tool_name == "get_map":
            result = self.get_map()
            is_error = result.startswith("Error")
            return result, is_error
        elif tool_name == "submit_answer":
            code_arg = arguments.get("code", "")
            result = self.submit_answer(code_arg)
            return result, False
        else:
            return f"Error: Unknown tool '{tool_name}'. Available: list_directory, read_file, get_map, submit_answer.", True

    # -------------------------------------------------------------------------
    # LLM Interaction
    # -------------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Calls the configured LLM or custom simulated client."""
        if self.llm_client is not None:
            return self.llm_client(messages, TOOLS_SCHEMA)

        if not self.api_key:
            raise ValueError(
                "API key not configured. Set GEMINI_API_KEY, GOOGLE_API_KEY, "
                "or TREASURE_API_KEY in your environment or .env file."
            )

        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "api-key": self.api_key,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": self.tools_schema,
            "tool_choice": "auto",
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                if e.code in (503, 429) and attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                try:
                    err_json = json.loads(err_body)
                    err_msg = err_json.get("error", {}).get("message", err_body)
                except Exception:
                    err_msg = err_body
                raise RuntimeError(f"Google AI / LLM API request failed ({e.code}): {err_msg}")
            except (TimeoutError, urllib.error.URLError) as e:
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise RuntimeError(f"Google AI / LLM API connection error: {e}")

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

        run_start_time = time.perf_counter()
        self.resource_tracker.start()
        step_count = 0
        api_calls_count = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_tool_cost = 0

        while step_count < self.max_steps and not self.finished:
            # Call LLM
            api_calls_count += 1
            llm_response = self._call_llm(messages)

            # Record API call usage in resource tracker
            usage = llm_response.get("usage", {})
            self.resource_tracker.record_api_call(usage)

            # Extract token usage
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            reasoning_tokens = (
                usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                or max(0, total_tokens - (prompt_tokens + completion_tokens))
            )
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_reasoning_tokens += reasoning_tokens

            choice = llm_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            # Extract thought summary and thought signature (never fabricated)
            thought_summary = extract_thought_summary(message)
            thought_signature = extract_thought_signature(message)

            # Append assistant message to context (preserves thought_signatures across turns)
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

            is_parallel_batch = len(tool_calls) > 1

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

                # Cost accounting
                action_cost = self.read_cost if tool_name == "read_file" else 0
                total_tool_cost += action_cost

                # Execute tool with timing
                t0 = time.perf_counter()
                result_text, is_error = self.execute_tool(tool_name, args)
                duration = time.perf_counter() - t0

                # Record in decoupled resource tracker
                tool_record = self.resource_tracker.record_tool_execution(
                    step=step_count,
                    action=tool_name,
                    arguments=args,
                    result_text=result_text,
                    duration_seconds=duration,
                    cost_units=action_cost,
                    is_parallel=is_parallel_batch,
                )

                # Record action metric
                action_record = {
                    "step": step_count,
                    "thought_summary": thought_summary,
                    "observable_decision": {
                        "action": tool_name,
                        "arguments": args,
                    },
                    "action": tool_name,
                    "arguments": args,
                    "result": result_text,
                    "result_bytes": tool_record.result_bytes,
                    "tool_result_tokens": tool_record.estimated_tool_result_tokens,
                    "status": "error" if is_error else "success",
                    "duration_seconds": round(duration, 5),
                    "cost": action_cost,
                    "is_parallel": is_parallel_batch,
                    "tokens": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "total_tokens": prompt_tokens + completion_tokens + reasoning_tokens,
                    },
                    "thought_signature": thought_signature,
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

        self.resource_tracker.stop()
        metrics = self.resource_tracker.get_metrics_summary()

        total_duration = sum(item["duration_seconds"] for item in self.trajectory)
        total_elapsed = time.perf_counter() - run_start_time

        read_actions = [a for a in self.trajectory if a["action"] == "read_file"]
        parallel_reads = [a for a in read_actions if a.get("is_parallel")]
        map_actions = [a for a in self.trajectory if a["action"] == "get_map"]

        return {
            "status": status,
            "submitted_code": self.submitted_code,
            "total_steps": len(self.trajectory),
            "api_calls_count": api_calls_count,
            "read_actions_count": len(read_actions),
            "parallel_reads_count": len(parallel_reads),
            "sequential_reads_count": len(read_actions) - len(parallel_reads),
            "map_available": self.document_map is not None,
            "map_consulted": len(map_actions) > 0,
            "map_consulted_step": map_actions[0]["step"] if map_actions else None,
            "map_actions_count": len(map_actions),
            "total_tool_cost": total_tool_cost,
            "read_cost_configured": self.read_cost,
            "reasoning_effort": self.reasoning_effort,
            "total_duration_seconds": round(total_duration, 4),
            "total_elapsed_seconds": round(total_elapsed, 4),
            "total_tokens": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "reasoning_tokens": total_reasoning_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens + total_reasoning_tokens,
            },
            # Decoupled resource instrumentation metrics
            "resource_metrics": metrics,
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "reasoning_tokens": metrics["reasoning_tokens"],
            "tool_result_tokens": metrics["tool_result_tokens"],
            "tool_calls": metrics["tool_calls"],
            "api_calls": metrics["api_calls"],
            "estimated_cost_usd": metrics["estimated_cost_usd"],
            "wall_clock_seconds": metrics["wall_clock_seconds"],
            "trajectory": self.trajectory,
        }
