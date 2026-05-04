import os

import pytest

from skydiscover.llm.tools.run_command_tool import _check_command_safety, run_command_handler


class TestRunCommandToolSafety:
    def test_blocks_shell_operators_in_safe_mode(self):
        err = _check_command_safety("echo hi | wc -c")
        assert err is not None

    def test_blocks_disallowed_executable(self):
        err = _check_command_safety("curl https://example.com")
        assert err is not None


@pytest.mark.asyncio
class TestRunCommandHandler:
    async def test_rejects_unsafe_when_disabled(self, tmp_path):
        out, success = await run_command_handler(
            {"command": "echo hi | wc -c", "unsafe": True},
            codebase_root=str(tmp_path),
            allow_unsafe_commands=False,
        )
        assert success is False
        assert "allow_unsafe_commands" in out

    async def test_runs_safe_command(self, tmp_path):
        out, success = await run_command_handler(
            {"command": "python -c \"print(40+2)\""},
            codebase_root=str(tmp_path),
        )
        assert success is True
        assert "42" in out

    @pytest.mark.skipif(os.name == "nt", reason="pipe syntax differs on Windows shells")
    async def test_runs_shell_command_when_enabled(self, tmp_path):
        out, success = await run_command_handler(
            {"command": "printf 'a\\n' | wc -l", "unsafe": True},
            codebase_root=str(tmp_path),
            allow_unsafe_commands=True,
        )
        assert success is True
        assert "\n1" in out or " 1" in out

