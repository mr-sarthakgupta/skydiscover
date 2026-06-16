"""Agentic code generator -- multi-turn tool-calling loop with read_file and search."""

import asyncio
import concurrent.futures
import fnmatch
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.llm.openai import is_openai_reasoning_model
from skydiscover.llm.responses_utils import (
    convert_messages_to_responses_input,
    extract_responses_output,
)
from skydiscover.utils.code_utils import build_repo_map

logger = logging.getLogger(__name__)
_agentic_log = logging.getLogger("skydiscover.agentic_trace")
EXTERNAL_RESEARCH_TOOLS = {"web_search", "research_papers", "hf_papers", "fetch_webpage"}

_TOOL_SCHEMAS_PATH = Path(__file__).parent / "tool_schemas" / "agentic_tools.json"
with open(_TOOL_SCHEMAS_PATH, "r") as _f:
    TOOL_SCHEMAS = json.load(_f)
AVAILABLE_TOOL_NAMES = [
    schema.get("function", {}).get("name", "")
    for schema in TOOL_SCHEMAS
    if schema.get("function", {}).get("name")
]

# Responses API uses a flattened tool format (name/description/parameters at top level)
TOOL_SCHEMAS_RESPONSES = [
    {
        "type": "function",
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "parameters": t["function"]["parameters"],
    }
    for t in TOOL_SCHEMAS
]

_AGENTIC_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "context_builder"
    / "default"
    / "templates"
    / "agentic_system_message.txt"
)
with open(_AGENTIC_PROMPT_PATH, "r") as _f:
    _AGENTIC_SYSTEM_PROMPT = _f.read()


class AgenticGenerator:
    """
    V0 [simple version]: Multi-turn tool-calling agent that explores a codebase before generating code.

    Tools: read_file, search. When it stops calling tools, its text output
    is the final answer. Returns None if no output is produced (caller falls
    back to direct generation).
    """

    def __init__(self, llm_pool, config, trace_dir: Optional[str] = None):
        self.llm_pool = llm_pool
        self.config = config
        self.trace_dir = trace_dir

    async def generate(self, system_message: str, user_message: str) -> Optional[str]:
        """Run the agent loop. Returns generated text, or None on failure."""
        cfg = self.config
        files_read: set = set()
        conversation: List[Dict[str, Any]] = []
        trace_log: List[Dict[str, Any]] = []
        t0 = time.time()
        external_research_required = _requires_external_research(system_message)
        external_research_nudge_sent = False

        sys_prompt = f"{system_message}\n\n{_AGENTIC_SYSTEM_PROMPT}"
        logger.info("Agentic tools available: %s", ", ".join(AVAILABLE_TOOL_NAMES))
        repo_map = build_repo_map(
            cfg.codebase_root,
            max_depth=cfg.repo_map_max_depth,
            allowed_extensions=cfg.allowed_extensions,
            excluded_dirs=cfg.excluded_dirs,
        )

        user_parts = [user_message]
        if repo_map:
            user_parts.append(f"\n## Project structure\n```\n{repo_map}\n```")
        conversation.append({"role": "user", "content": "\n".join(user_parts)})

        for step in range(cfg.max_steps):
            if time.time() - t0 > cfg.overall_timeout:
                logger.warning("Agent timed out at step %d", step)
                break

            remaining = cfg.max_steps - step - 1
            if step > 0:
                elapsed = time.time() - t0
                time_left = max(0, cfg.overall_timeout - elapsed)
                step_note = _build_step_note(step, cfg.max_steps, remaining, time_left)
                if conversation and conversation[-1].get("role") == "user":
                    conversation[-1]["content"] += f"\n\n{step_note}"
                else:
                    conversation.append({"role": "user", "content": step_note})

            if _context_chars(sys_prompt, conversation) > cfg.max_context_chars:
                conversation.append(
                    {
                        "role": "user",
                        "content": "Context limit reached. Output your improved program now.",
                    }
                )

            try:
                assistant_msg = await asyncio.wait_for(
                    self._call_llm(sys_prompt, conversation),
                    timeout=cfg.per_step_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Step %d: LLM timed out", step)
                conversation.append(
                    {
                        "role": "user",
                        "content": "Timed out. Output your solution or try a simpler action.",
                    }
                )
                continue
            except Exception as e:
                logger.error("Step %d: LLM error: %s", step, e)
                break

            tool_calls = assistant_msg.get("tool_calls", [])
            text_content = assistant_msg.get("content", "").strip()
            conversation.append(assistant_msg)

            if not tool_calls:
                if text_content:
                    external_research_used = any(
                        entry.get("tool") in EXTERNAL_RESEARCH_TOOLS for entry in trace_log
                    )
                    if (
                        external_research_required
                        and not external_research_used
                        and not external_research_nudge_sent
                        and remaining > 0
                    ):
                        external_research_nudge_sent = True
                        logger.info(
                            "Agent attempted final response before external research; "
                            "requesting one targeted web_search or research_papers call."
                        )
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Before producing the final program for this scientific "
                                    "discovery task, make one targeted external research call "
                                    "using `web_search` or `research_papers`, unless the tool "
                                    "itself fails. Use the result to justify the next symbolic "
                                    "structure, then output the complete improved program."
                                ),
                            }
                        )
                        continue
                    logger.info(
                        "Agent produced text at step %d (%d files read)", step, len(files_read)
                    )
                    self._save_trace(trace_log, conversation, sys_prompt)
                    return text_content
                conversation.append(
                    {
                        "role": "user",
                        "content": "Use a tool to explore, or output your improved program.",
                    }
                )
                continue

            for tc in tool_calls:
                fn = tc.get("function", {})
                name, raw, tc_id = fn.get("name", ""), fn.get("arguments", "{}"), tc.get("id", "")

                try:
                    args = json.loads(raw)
                except (json.JSONDecodeError, TypeError) as e:
                    conversation.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Bad JSON: {e}"}
                    )
                    continue

                logger.info(
                    "Step %d: tool=%s args=%s",
                    step,
                    name,
                    {
                        k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                        for k, v in args.items()
                    },
                )

                result = await self._run_tool(name, args, files_read)
                logger.info("Step %d: tool=%s returned:\n%s", step, name, result.get("content", ""))
                conversation.append(
                    {"role": "tool", "tool_call_id": tc_id, "content": result["content"]}
                )
                trace_log.append({
                    "step": step, "tool": name, "args": args,
                    "result_len": len(result.get("content", "")),
                })

        self._save_trace(trace_log, conversation, sys_prompt)
        logger.warning("Agent loop ended without producing code")
        return None

    def _save_trace(
        self, trace_log: list, conversation: list, system_prompt: str
    ) -> None:
        """Persist the agentic trace to the run reference dir and codebase reference dir."""
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            serialized_conversation = _serialize_conversation(conversation)
            payload = {
                "timestamp": ts,
                "available_tools": AVAILABLE_TOOL_NAMES,
                "external_research_tool_names": sorted(EXTERNAL_RESEARCH_TOOLS),
                "external_research_tools_used": any(
                    entry.get("tool") in EXTERNAL_RESEARCH_TOOLS for entry in trace_log
                ),
                "tool_call_count": len(trace_log),
                "tool_calls": trace_log,
                "system_prompt": _serialize_message_content(system_prompt),
                "conversation": serialized_conversation,
                "model_inputs": _build_model_inputs(system_prompt, conversation),
            }
            saved_paths = []

            candidate_dirs = []
            if self.trace_dir:
                candidate_dirs.append(self.trace_dir)
            root = self.config.codebase_root
            if root:
                candidate_dirs.append(os.path.join(root, "reference"))

            for ref_dir in dict.fromkeys(candidate_dirs):
                os.makedirs(ref_dir, exist_ok=True)
                path = os.path.join(ref_dir, f"agentic_trace_{ts}.json")
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
                saved_paths.append(path)

            if saved_paths:
                _agentic_log.info("Agentic trace saved to %s", ", ".join(saved_paths))
        except Exception as e:
            logger.debug("Failed to save agentic trace: %s", e)

    async def _call_llm(
        self, system_message: str, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call a sampled LLM with tool schemas.

        Routes to Bedrock Converse API for BedrockLLM, otherwise uses
        Chat Completions with Responses API fallback.
        """
        model = self.llm_pool.models[
            self.llm_pool.random_state.choices(
                range(len(self.llm_pool.models)), weights=self.llm_pool.weights, k=1
            )[0]
        ]

        from skydiscover.llm.bedrock import BedrockLLM

        if isinstance(model, BedrockLLM):
            return await self._call_llm_bedrock(model, system_message, conversation)

        if not hasattr(model, "client"):
            raise RuntimeError(
                f"Agentic mode requires an OpenAI-compatible LLM ({type(model).__name__} has no .client)"
            )

        # If we already know this model needs the Responses API, skip Chat Completions
        if getattr(model, "_use_responses_api", False):
            return await self._call_llm_responses(model, system_message, conversation)

        messages = [{"role": "system", "content": system_message}] + conversation
        is_reasoning = is_openai_reasoning_model(model.model, getattr(model, "api_base", "") or "")

        params: Dict[str, Any] = {
            "model": model.model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }
        if is_reasoning:
            if model.max_tokens:
                params["max_completion_tokens"] = model.max_tokens
            if getattr(model, "reasoning_effort", None):
                params["reasoning_effort"] = model.reasoning_effort
        else:
            if model.temperature is not None:
                params["temperature"] = model.temperature
            if model.top_p is not None:
                params["top_p"] = model.top_p
            if model.max_tokens is not None:
                params["max_tokens"] = model.max_tokens

        from skydiscover.llm.cost_tracker import global_cost_tracker

        global_cost_tracker.check_budget()
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None, lambda: model.client.chat.completions.create(**params)
            )
        except Exception as exc:
            if "unsupported" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise
            logger.info("Chat Completions unsupported for agentic; falling back to Responses API")
            model._use_responses_api = True
            return await self._call_llm_responses(model, system_message, conversation)

        _record_openai_chat_usage(resp.usage, model.model)
        msg = resp.choices[0].message
        out: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return out

    async def _call_llm_responses(
        self,
        model,
        system_message: str,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call the LLM via the Responses API (Azure-compatible) with tool support."""
        is_reasoning = is_openai_reasoning_model(model.model, getattr(model, "api_base", "") or "")

        input_items = convert_messages_to_responses_input(conversation)

        resp_params: Dict[str, Any] = {
            "model": model.model,
            "input": input_items,
            "instructions": system_message,
            "tools": TOOL_SCHEMAS_RESPONSES,
            "tool_choice": "auto",
        }
        if is_reasoning:
            if model.max_tokens:
                resp_params["max_output_tokens"] = model.max_tokens
            if getattr(model, "reasoning_effort", None):
                resp_params["reasoning"] = {"effort": model.reasoning_effort}
        else:
            if model.temperature is not None:
                resp_params["temperature"] = model.temperature
            if model.max_tokens is not None:
                resp_params["max_output_tokens"] = model.max_tokens

        from skydiscover.llm.cost_tracker import global_cost_tracker

        global_cost_tracker.check_budget()
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: model.client.responses.create(**resp_params)
        )

        _record_responses_api_usage(resp, model.model)
        text, _, tool_calls = extract_responses_output(resp)
        out: Dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out

    # ------------------------------------------------------------------
    # Bedrock Converse API (tool use)
    # ------------------------------------------------------------------

    async def _call_llm_bedrock(
        self,
        model,
        system_message: str,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call Bedrock Converse API with native tool use support and retries."""
        from skydiscover.llm.cost_tracker import global_cost_tracker

        global_cost_tracker.check_budget()

        bedrock_messages = _conv_to_bedrock(conversation)
        tool_config = _bedrock_tool_config()
        cache_point = _bedrock_prompt_cache_point()
        if cache_point:
            tool_config["tools"].append(cache_point)
            _add_bedrock_conversation_cache_point(bedrock_messages, cache_point)

        params: Dict[str, Any] = {
            "modelId": model.model,
            "messages": bedrock_messages,
            "system": (
                [{"text": system_message}, cache_point]
                if cache_point
                else [{"text": system_message}]
            ),
            "toolConfig": tool_config,
        }

        inference_config: Dict[str, Any] = {}
        if model.max_tokens:
            inference_config["maxTokens"] = int(model.max_tokens)
        if model.temperature is not None:
            inference_config["temperature"] = float(model.temperature)
        if inference_config:
            params["inferenceConfig"] = inference_config

        retries, retry_delay, timeout = model._resolve_retry_options()

        for attempt in range(retries + 1):
            try:
                loop = asyncio.get_running_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: model.client.converse(**params)),
                    timeout=timeout,
                )
                break
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        "Bedrock agentic timeout attempt %d/%d, retrying...",
                        attempt + 1, retries + 1,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as exc:
                if attempt < retries:
                    logger.warning(
                        "Bedrock agentic error attempt %d/%d: %s, retrying...",
                        attempt + 1, retries + 1, exc,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise

        usage = response.get("usage", {})
        global_cost_tracker.record_usage(
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            cache_read_tokens=usage.get("cacheReadInputTokens", 0),
            cache_write_tokens=usage.get("cacheWriteInputTokens", 0),
            model=model.model,
        )

        return _parse_bedrock_response(response)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def _run_tool(self, name: str, args: Dict[str, Any], files_read: set) -> Dict[str, Any]:
        try:
            if name == "read_file":
                return self._tool_read_file(args, files_read)
            elif name == "search":
                return self._tool_search(args)
            elif name == "web_search":
                from skydiscover.llm.tools.web_search_tool import web_search_handler
                output, success = await web_search_handler(args)
                return {"content": output, "_error": not success}
            elif name in ("research_papers", "hf_papers"):
                from skydiscover.llm.tools.papers_tool import research_papers_handler
                output, success = await research_papers_handler(args)
                return {"content": output, "_error": not success}
            elif name == "fetch_webpage":
                from skydiscover.llm.tools.fetch_webpage_tool import fetch_webpage_handler
                output, success = await fetch_webpage_handler(
                    args, codebase_root=self.config.codebase_root
                )
                return {"content": output, "_error": not success}
            elif name == "run_command":
                from skydiscover.llm.tools.run_command_tool import run_command_handler
                if not getattr(self.config, "run_command_enabled", True):
                    return _err("run_command is disabled by configuration (agentic.run_command_enabled=false).")
                output, success = await run_command_handler(
                    args,
                    codebase_root=self.config.codebase_root,
                    run_command_default_timeout=getattr(self.config, "run_command_default_timeout", 30),
                    run_command_max_timeout=getattr(self.config, "run_command_max_timeout", 120),
                    run_command_max_output_chars=getattr(self.config, "run_command_max_output_chars", 20_000),
                    allow_unsafe_commands=getattr(self.config, "allow_unsafe_commands", False),
                )
                return {"content": output, "_error": not success}
            return _err(
                f"Unknown tool '{name}'. Available: read_file, search, web_search, research_papers, fetch_webpage, run_command."
            )
        except Exception as e:
            return _err(f"Tool '{name}' error: {e}")

    def _tool_read_file(self, args: Dict[str, Any], files_read: set) -> Dict[str, Any]:
        path = args.get("path", "")
        if not path:
            return _err("'path' is required.")

        root = self.config.codebase_root
        if not root:
            return _err("codebase_root not configured.")
        full = os.path.join(root, path) if not os.path.isabs(path) else path

        ok, resolved, err = _validate_path(
            full, root, self.config.allowed_extensions, self.config.excluded_dirs
        )
        if not ok:
            return _err(err)

        if resolved not in files_read and len(files_read) >= self.config.max_files_read:
            return _err(f"Read limit ({self.config.max_files_read}). Output your solution.")

        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return _err(f"Cannot read: {e}")

        total = len(lines)
        start = max(1, int(args.get("line_start") or 1)) - 1
        end = min(total, int(args.get("line_end") or total))
        content = "".join(lines[start:end])

        if len(content) > self.config.max_file_chars:
            half = self.config.max_file_chars // 2
            content = (
                content[:half]
                + f"\n\n... ({len(content) - self.config.max_file_chars} chars truncated) ...\n\n"
                + content[-half:]
            )

        files_read.add(resolved)
        rel = os.path.relpath(resolved, root)
        if rel.startswith("reference" + os.sep) or rel.startswith("reference/"):
            logger.info("read_file: loaded reference file %s (%d lines)", rel, total)
        numbered = [
            f"{i:4d} | {ln.rstrip(chr(10))}"
            for i, ln in enumerate(content.splitlines(True), start=start + 1)
        ]
        return {"content": f"{rel} (lines {start + 1}-{end} of {total})\n" + "\n".join(numbered)}

    def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        pattern = args.get("pattern", "")
        glob_pat = args.get("file_glob", "*.py")

        if not pattern:
            return _err("'pattern' is required.")
        if len(pattern) > self.config.max_regex_length:
            return _err(f"Pattern too long ({len(pattern)} > {self.config.max_regex_length}).")

        safety_err = _check_regex_safety(pattern)
        if safety_err:
            return _err(safety_err)

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return _err(f"Invalid regex: {e}")

        root = self.config.codebase_root
        if not root:
            return _err("codebase_root not configured.")
        excluded = set(self.config.excluded_dirs)
        allowed = set(self.config.allowed_extensions)
        matches: List[str] = []
        n_files = 0
        max_results = self.config.max_search_results

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in excluded]
            for fname in filenames:
                if not fnmatch.fnmatch(fname, glob_pat):
                    continue
                if os.path.splitext(fname)[1].lower() not in allowed:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    if os.path.getsize(fpath) > self.config.max_file_chars:
                        continue
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except Exception:
                    continue

                n_files += 1
                ok, hits, err = _safe_regex_search(compiled, text, self.config.regex_timeout)
                if not ok:
                    return _err(err)

                rel = os.path.relpath(fpath, root)
                for hit in hits:
                    matches.append(f"{rel}:{hit}")
                    if len(matches) >= max_results:
                        break
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        if not matches:
            return {"content": f"No matches for '{pattern}' in {n_files} files."}

        suffix = f"\n(capped at {max_results} results)" if len(matches) >= max_results else ""
        return {"content": "\n".join(matches) + suffix}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _record_openai_chat_usage(usage, model_name: str) -> None:
    """Record token usage from an OpenAI Chat Completions response."""
    if usage is None:
        return
    from skydiscover.llm.cost_tracker import global_cost_tracker

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cache_read = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    global_cost_tracker.record_usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cache_read_tokens=cache_read,
        model=model_name,
    )


def _record_responses_api_usage(response, model_name: str) -> None:
    """Record token usage from an OpenAI Responses API response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    from skydiscover.llm.cost_tracker import global_cost_tracker

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = 0
    details = getattr(usage, "input_tokens_details", None)
    if details:
        cache_read = getattr(details, "cached_tokens", 0) or 0
    global_cost_tracker.record_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        model=model_name,
    )


def _build_step_note(step: int, max_steps: int, remaining: int, time_left: float) -> str:
    """Build a concise step-counter message with progressive urgency."""
    time_str = f"{time_left:.0f}s" if time_left < 600 else f"{time_left / 60:.0f}min"
    if remaining <= 2:
        return (
            f"[Step {step + 1}/{max_steps} | {remaining} turns left | {time_str} remaining] "
            f"URGENT: You MUST output your final improved program NOW. "
            f"Do NOT call any more tools. Respond with your complete solution code."
        )
    if remaining <= 5:
        return (
            f"[Step {step + 1}/{max_steps} | {remaining} turns left | {time_str} remaining] "
            f"Time is running out. Finish your exploration and output your improved program."
        )
    if remaining <= 15:
        return (
            f"[Step {step + 1}/{max_steps} | {remaining} turns left | {time_str} remaining] "
            f"Start wrapping up — you must output a complete improved program before your turns run out."
        )
    return f"[Step {step + 1}/{max_steps} | {remaining} turns left | {time_str} remaining]"


def _requires_external_research(system_message: str) -> bool:
    """Detect benchmark prompts that explicitly ask for outside research."""
    lowered = system_message.lower()
    markers = (
        "internet and paper tools are available and valuable",
        "perform targeted research",
        "governing relationships",
        "external sources before proposing",
    )
    return any(marker in lowered for marker in markers)


def _err(msg: str) -> Dict[str, Any]:
    return {"content": msg, "_error": True}


def _context_chars(system: str, conversation: List[Dict[str, Any]]) -> int:
    n = len(system)
    for msg in conversation:
        n += len(msg.get("content", ""))
        for tc in msg.get("tool_calls", []):
            n += len(tc.get("function", {}).get("arguments", ""))
    return n


_SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".env.staging",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
        "credentials.json",
        "credentials.yaml",
        "service-account.json",
        "service_account.json",
        ".netrc",
        ".pgpass",
        ".my.cnf",
    }
)


def _validate_path(
    requested: str, root: str, allowed_ext: tuple, excluded_dirs: tuple
) -> Tuple[bool, str, str]:
    """Validate a file path. Returns (ok, resolved_path, error_message)."""
    try:
        resolved = os.path.realpath(requested)
    except (OSError, ValueError) as e:
        return False, "", f"Invalid path: {e}"

    root_abs = os.path.realpath(root)
    if not resolved.startswith(root_abs + os.sep) and resolved != root_abs:
        return False, "", "Path outside codebase root."

    try:
        rel = os.path.relpath(resolved, root_abs)
        for part in Path(rel).parts:
            if part in excluded_dirs:
                return False, "", f"Path in excluded directory '{part}'."
    except ValueError:
        pass

    basename = os.path.basename(resolved).lower()
    if basename in _SENSITIVE_FILENAMES:
        return False, "", f"Access denied: '{basename}' may contain secrets."

    if not os.path.isfile(resolved):
        parent_dir = os.path.dirname(resolved)
        if os.path.isdir(parent_dir):
            try:
                siblings = sorted(os.listdir(parent_dir))[:15]
                rel_dir = os.path.relpath(parent_dir, root_abs)
                return (
                    False,
                    "",
                    f"Not found: '{os.path.basename(resolved)}'. '{rel_dir}/' contains: {siblings}",
                )
            except OSError:
                pass
        return False, "", f"File not found: '{requested}'."

    ext = os.path.splitext(resolved)[1].lower()
    if ext not in allowed_ext:
        return False, "", f"Extension '{ext}' not allowed."

    return True, resolved, ""


def _serialize_message_content(content: str) -> str:
    """Truncate large message bodies for JSON-safe trace persistence."""
    if len(content) > 2000:
        return content[:1000] + f"\n...[{len(content)} chars total]...\n" + content[-500:]
    return content


def _build_model_inputs(
    system_prompt: str, conversation: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build the full chat payload sent to the model (system + conversation)."""
    return [
        {"role": "system", "content": _serialize_message_content(system_prompt)},
        *_serialize_conversation(conversation),
    ]


def _serialize_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Produce a JSON-safe copy of the conversation (truncate large tool results)."""
    out: List[Dict[str, Any]] = []
    for msg in conversation:
        entry: Dict[str, Any] = {"role": msg.get("role", "")}
        entry["content"] = _serialize_message_content(msg.get("content", ""))
        if "tool_calls" in msg:
            entry["tool_calls"] = msg["tool_calls"]
        if "tool_call_id" in msg:
            entry["tool_call_id"] = msg["tool_call_id"]
        out.append(entry)
    return out


# ------------------------------------------------------------------
# Bedrock Converse format converters
# ------------------------------------------------------------------


def _bedrock_tool_config() -> Dict[str, Any]:
    """Convert OpenAI tool schemas to Bedrock toolConfig format."""
    tools = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        params = dict(fn["parameters"])
        params.pop("additionalProperties", None)
        tools.append({
            "toolSpec": {
                "name": fn["name"],
                "description": fn["description"],
                "inputSchema": {"json": params},
            }
        })
    return {"tools": tools}


def _bedrock_prompt_cache_point() -> Optional[Dict[str, Any]]:
    """Return a Bedrock Converse cache checkpoint, unless disabled by env."""
    raw = os.environ.get("BEDROCK_PROMPT_CACHE_TTL", "1h").strip()
    if raw.lower() in {"", "0", "false", "off", "none"}:
        return None

    cache_point: Dict[str, Any] = {"type": "default"}
    if raw:
        cache_point["ttl"] = raw
    return {"cachePoint": cache_point}


def _add_bedrock_conversation_cache_point(
    bedrock_messages: List[Dict[str, Any]],
    cache_point: Dict[str, Any],
) -> None:
    """Mark the current conversation prefix for reuse by later agentic steps."""
    if os.environ.get("BEDROCK_CACHE_CONVERSATION", "1").lower() in {
        "0",
        "false",
        "off",
        "none",
    }:
        return
    if not bedrock_messages:
        return

    content = bedrock_messages[-1].setdefault("content", [])
    if isinstance(content, list):
        content.append(cache_point)


def _conv_to_bedrock(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format conversation to Bedrock Converse messages.

    Handles: user text, assistant text+tool_calls, and consecutive tool
    result messages (grouped into a single user message as toolResult blocks).
    """
    bedrock_msgs: List[Dict[str, Any]] = []
    i = 0
    while i < len(conversation):
        msg = conversation[i]
        role = msg.get("role")

        if role == "user":
            text = msg.get("content", "") or " "
            bedrock_msgs.append({"role": "user", "content": [{"text": text}]})
            i += 1

        elif role == "assistant":
            content: List[Dict[str, Any]] = []
            text = msg.get("content", "")
            if text:
                content.append({"text": text})
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                try:
                    tool_input = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}
                content.append({
                    "toolUse": {
                        "toolUseId": tc.get("id", f"tc_{i}"),
                        "name": fn.get("name", ""),
                        "input": tool_input,
                    }
                })
            bedrock_msgs.append({"role": "assistant", "content": content or [{"text": " "}]})
            i += 1

        elif role == "tool":
            tool_results: List[Dict[str, Any]] = []
            while i < len(conversation) and conversation[i].get("role") == "tool":
                tr = conversation[i]
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tr.get("tool_call_id", f"tc_{i}"),
                        "content": [{"text": tr.get("content", "")}],
                        "status": "success",
                    }
                })
                i += 1
            bedrock_msgs.append({"role": "user", "content": tool_results})

        else:
            i += 1

    return bedrock_msgs


def _parse_bedrock_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Bedrock Converse response to internal OpenAI-like format."""
    output = response.get("output", {}).get("message", {})
    content_blocks = output.get("content", [])

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append({
                "id": tu.get("toolUseId", ""),
                "type": "function",
                "function": {
                    "name": tu.get("name", ""),
                    "arguments": json.dumps(tu.get("input", {})),
                },
            })

    result: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


_NESTED_QUANTIFIER_RE = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*?]|\([^)]*[+*][^)]*\)\s*\{")

_MAX_SEARCH_LINE_LEN = 2000


def _check_regex_safety(pattern: str) -> Optional[str]:
    """Reject patterns with nested quantifiers that cause catastrophic backtracking."""
    if _NESTED_QUANTIFIER_RE.search(pattern):
        return "Nested quantifiers detected (e.g. '(a+)+'). Use a simpler pattern."
    return None


_REGEX_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="regex")


def _safe_regex_search(
    compiled: "re.Pattern", text: str, timeout: float = 2.0
) -> Tuple[bool, List[str], str]:
    """Regex search with thread-based timeout."""

    def do_search():
        return [
            f"{i}: {line}"
            for i, line in enumerate(text.splitlines(), 1)
            if len(line) <= _MAX_SEARCH_LINE_LEN and compiled.search(line)
        ]

    fut = _REGEX_EXECUTOR.submit(do_search)
    try:
        result = fut.result(timeout=timeout)
        return True, result, ""
    except concurrent.futures.TimeoutError:
        return False, [], f"Regex timed out ({timeout}s). Simplify the pattern."
