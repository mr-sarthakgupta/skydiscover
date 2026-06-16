"""AWS Bedrock LLM backend using the native Converse API."""

from __future__ import annotations

import asyncio
import configparser
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse

logger = logging.getLogger("skydiscover.llm")


def _content_to_text(content: Any) -> str:
    """Convert OpenAI-style message content into plain text for Bedrock."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("text") is not None:
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def _bedrock_role(role: str) -> str:
    """Map common chat roles to Bedrock Converse roles."""
    return "assistant" if role == "assistant" else "user"


def _region_from_api_base(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return None
    if api_base.startswith("bedrock:"):
        return api_base.split(":", 1)[1] or None
    return None


def _prompt_cache_point() -> Optional[Dict[str, Any]]:
    """Return a Bedrock Converse cache checkpoint, unless disabled by env."""
    raw = os.environ.get("BEDROCK_PROMPT_CACHE_TTL", "1h").strip()
    if raw.lower() in {"", "0", "false", "off", "none"}:
        return None

    cache_point: Dict[str, Any] = {"type": "default"}
    if raw:
        cache_point["ttl"] = raw
    return {"cachePoint": cache_point}


def _bedrock_api_key_from_aws_credentials(
    profile: Optional[str] = None,
    credentials_path: Optional[Path] = None,
) -> Optional[str]:
    """Return a Bedrock API key stored in aws_session_token, if present.

    Bedrock bearer API keys use an ABSK prefix. They are not AWS STS session
    tokens, but users sometimes place them in ~/.aws/credentials under
    aws_session_token because both are token-like values.
    """
    credentials_path = credentials_path or Path(
        os.environ.get("AWS_SHARED_CREDENTIALS_FILE", Path.home() / ".aws" / "credentials")
    )
    if not credentials_path.exists():
        return None

    parser = configparser.RawConfigParser()
    parser.read(credentials_path)
    section = profile or os.environ.get("AWS_PROFILE") or "default"
    if not parser.has_section(section):
        return None

    token = parser.get(section, "aws_session_token", fallback="").strip()
    if token.startswith("ABSK"):
        return token
    return None


def _ensure_bedrock_bearer_token(profile: Optional[str] = None) -> bool:
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return False

    token = _bedrock_api_key_from_aws_credentials(profile)
    if not token:
        return False

    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    logger.info(
        "Using Bedrock API key from aws_session_token as AWS_BEARER_TOKEN_BEDROCK"
    )
    return True


class BedrockLLM(LLMInterface):
    """LLM backend using AWS Bedrock Runtime's Converse API."""

    def __init__(self, model_cfg: Optional[LLMModelConfig] = None):
        if model_cfg is None:
            raise ValueError("BedrockLLM requires an LLMModelConfig")

        self.model = model_cfg.name
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base

        if not self.model:
            raise ValueError("Bedrock model config requires a model name")

        self.region = (
            _region_from_api_base(self.api_base)
            or os.environ.get("BEDROCK_AWS_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )

        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise ImportError(
                "AWS Bedrock support requires boto3. Install with "
                "`pip install boto3` or `uv sync --extra bedrock`."
            ) from exc

        session_kwargs: dict[str, str] = {}
        profile = os.environ.get("AWS_PROFILE")
        if profile:
            session_kwargs["profile_name"] = profile
        _ensure_bedrock_bearer_token(profile)
        session = boto3.Session(**session_kwargs)
        client_config = BotoConfig(
            connect_timeout=int(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "10")),
            read_timeout=int(os.environ.get("BEDROCK_READ_TIMEOUT", self.timeout or 300)),
            retries={"mode": "standard", "total_max_attempts": 1},
        )
        self.client = session.client(
            "bedrock-runtime",
            region_name=self.region,
            config=client_config,
        )

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        model_key = ("bedrock", self.region, self.model)
        if model_key not in logger._initialized_models:
            logger.info(f"AWS Bedrock LLM: {self.model} region={self.region}")
            logger._initialized_models.add(model_key)

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        if kwargs.get("image_output"):
            raise NotImplementedError("BedrockLLM does not support image_output")
        text = await self._generate_text(system_message, messages, **kwargs)
        return LLMResponse(text=text)

    async def _generate_text(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> str:
        params = self._build_converse_params(system_message, messages, **kwargs)
        retries, retry_delay, timeout = self._resolve_retry_options(**kwargs)

        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(self._call_api(params), timeout=timeout)
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout attempt {attempt + 1}/{retries + 1}, retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as exc:
                if attempt < retries:
                    logger.warning(
                        f"Error attempt {attempt + 1}/{retries + 1}: {exc}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    def _build_converse_params(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        bedrock_messages: list[dict[str, Any]] = []
        for message in messages:
            text = _content_to_text(message.get("content"))
            if not text:
                continue
            bedrock_messages.append(
                {
                    "role": _bedrock_role(str(message.get("role", "user"))),
                    "content": [{"text": text}],
                }
            )

        if not bedrock_messages:
            bedrock_messages.append({"role": "user", "content": [{"text": ""}]})

        inference_config: dict[str, Any] = {}
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            inference_config["maxTokens"] = int(max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            inference_config["temperature"] = float(temperature)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            inference_config["topP"] = float(top_p)

        params: Dict[str, Any] = {
            "modelId": self.model,
            "messages": bedrock_messages,
        }
        if system_message:
            system_blocks: list[dict[str, Any]] = [{"text": system_message}]
            cache_point = _prompt_cache_point()
            if cache_point:
                system_blocks.append(cache_point)
            params["system"] = system_blocks
        if inference_config:
            params["inferenceConfig"] = inference_config
        return params

    async def _call_api(self, params: Dict[str, Any]) -> str:
        from skydiscover.llm.cost_tracker import global_cost_tracker

        global_cost_tracker.check_budget()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: self.client.converse(**params))
        self._record_usage(response)
        return self._extract_text(response)

    def _record_usage(self, response: Dict[str, Any]) -> None:
        from skydiscover.llm.cost_tracker import global_cost_tracker

        usage = response.get("usage", {})
        global_cost_tracker.record_usage(
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            cache_read_tokens=usage.get("cacheReadInputTokens", 0),
            cache_write_tokens=usage.get("cacheWriteInputTokens", 0),
            model=self.model,
        )

    def _extract_text(self, response: Dict[str, Any]) -> str:
        content = response.get("output", {}).get("message", {}).get("content", [])
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        return "\n".join(parts)

    def _resolve_retry_options(self, **kwargs) -> Tuple[int, int, int]:
        retries = kwargs.get("retries", self.retries)
        if retries is None:
            retries = 0
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        if retry_delay is None:
            retry_delay = 2
        timeout = kwargs.get("timeout", self.timeout)
        if timeout is None:
            timeout = 300
        return retries, retry_delay, timeout
