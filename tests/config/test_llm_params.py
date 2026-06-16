"""Tests for LLM config: optional temperature/top_p and api_base routing."""

import os
from dataclasses import fields
from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import LLMConfig, LLMModelConfig

_OPENAI_DEFAULT_API_BASE: str = next(
    f.default for f in fields(LLMConfig) if f.name == "api_base"
)


class TestLLMConfigDefaults:
    def test_default_temperature(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.temperature == 0.7

    def test_default_top_p_is_none(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.top_p is None

    def test_explicit_none_temperature(self):
        cfg = LLMConfig(name="test-model", temperature=None)
        assert cfg.temperature is None

    def test_explicit_none_top_p(self):
        cfg = LLMConfig(name="test-model", top_p=None)
        assert cfg.top_p is None

    def test_both_none(self):
        cfg = LLMConfig(name="test-model", temperature=None, top_p=None)
        assert cfg.temperature is None
        assert cfg.top_p is None


class TestApiBaseRouting:
    def test_unknown_model_preserves_local_api_base(self):
        local = "http://localhost:11434/v1"
        cfg = LLMConfig(
            name="my-custom-local-model",
            api_base=local,
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == local

    def test_unknown_model_gets_openai_default(self):
        cfg = LLMConfig(
            name="my-custom-local-model",
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == _OPENAI_DEFAULT_API_BASE

    def test_mixed_providers_with_local_api_base(self):
        cfg = LLMConfig(
            api_base="http://localhost:11434/v1",
            models=[
                LLMModelConfig(name="anthropic/claude-3-sonnet"),
                LLMModelConfig(name="my-local-model"),
            ],
        )
        assert cfg.models[0].api_base == "https://api.anthropic.com/v1/"
        assert cfg.models[1].api_base == "http://localhost:11434/v1"

    def test_bedrock_model_uses_native_provider(self):
        cfg = LLMConfig(
            models=[
                LLMModelConfig(
                    name="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                )
            ],
        )
        assert cfg.models[0].name == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert cfg.models[0].api_base == "bedrock"

    def test_bedrock_model_preserves_explicit_us_east_1_region(self):
        cfg = LLMConfig(
            api_base="bedrock:us-east-1",
            models=[
                LLMModelConfig(
                    name="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                )
            ],
        )
        assert cfg.models[0].api_base == "bedrock:us-east-1"

    def test_bedrock_does_not_fall_back_to_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        cfg = LLMConfig(
            models=[
                LLMModelConfig(
                    name="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                )
            ],
        )
        assert cfg.models[0].api_key is None


class TestBedrockLLMParams:
    def _make_llm(self, temperature=0.7, top_p=0.95):
        from skydiscover.llm.bedrock import BedrockLLM

        llm = BedrockLLM.__new__(BedrockLLM)
        llm.model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        llm.temperature = temperature
        llm.top_p = top_p
        llm.max_tokens = 1024
        llm.timeout = 10
        llm.retries = 0
        llm.retry_delay = 0
        llm.api_base = "bedrock"
        return llm

    def test_builds_converse_params(self):
        llm = self._make_llm(temperature=0.5, top_p=0.9)
        params = llm._build_converse_params(
            "sys",
            [{"role": "user", "content": "hello"}],
            max_tokens=512,
        )
        assert params["modelId"] == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert params["system"] == [{"text": "sys"}]
        assert params["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
        assert params["inferenceConfig"] == {
            "maxTokens": 512,
            "temperature": 0.5,
            "topP": 0.9,
        }

    def test_excludes_none_sampling_params(self):
        llm = self._make_llm(temperature=None, top_p=None)
        params = llm._build_converse_params(
            "sys",
            [{"role": "user", "content": "hello"}],
        )
        assert params["inferenceConfig"] == {"maxTokens": 1024}

    def test_detects_bedrock_api_key_in_session_token(self, tmp_path):
        from skydiscover.llm.bedrock import _bedrock_api_key_from_aws_credentials

        credentials = tmp_path / "credentials"
        credentials.write_text(
            "\n".join(
                [
                    "[default]",
                    "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
                    "aws_secret_access_key = example",
                    "aws_session_token = ABSKexample",
                ]
            )
        )

        assert (
            _bedrock_api_key_from_aws_credentials(credentials_path=credentials)
            == "ABSKexample"
        )

    def test_installs_bedrock_bearer_token_from_session_token(
        self, monkeypatch, tmp_path
    ):
        from skydiscover.llm.bedrock import _ensure_bedrock_bearer_token

        credentials = tmp_path / "credentials"
        credentials.write_text(
            "\n".join(
                [
                    "[default]",
                    "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
                    "aws_secret_access_key = example",
                    "aws_session_token = ABSKexample",
                ]
            )
        )
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))

        assert _ensure_bedrock_bearer_token() is True
        assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "ABSKexample"


class TestOpenAILLMParams:
    def _make_llm(self, temperature=0.7, top_p=0.95):
        from skydiscover.llm.openai import OpenAILLM

        cfg = LLMModelConfig(
            name="test-model",
            temperature=temperature,
            top_p=top_p,
            api_base="http://localhost:1234/v1",
            api_key="fake",
            timeout=10,
            retries=0,
            retry_delay=0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            llm = OpenAILLM(cfg)
        return llm

    @pytest.mark.asyncio
    async def test_params_include_temperature_and_top_p(self):
        llm = self._make_llm(temperature=0.5, top_p=0.9)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
            temperature=0.5,
            top_p=0.9,
        )
        params = llm._call_api.call_args[0][0]
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_params_exclude_none_top_p(self):
        llm = self._make_llm(top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "top_p" not in params
        assert "temperature" in params

    @pytest.mark.asyncio
    async def test_params_exclude_none_temperature(self):
        llm = self._make_llm(temperature=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" in params

    @pytest.mark.asyncio
    async def test_params_exclude_both_none(self):
        llm = self._make_llm(temperature=None, top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" not in params
