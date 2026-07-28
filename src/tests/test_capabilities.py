"""V3.0 能力层与图改进测试。

覆盖范围：
    - capabilities/base.py（自动发现、注册、隔离）
    - capabilities/bootstrap.py（ToolsConfig、build_capability_map、引导）
    - capabilities/providers/*（Shell、文件、日志、SQLite、Artifact、Skill）
    - build_graph.py V3（normalize_entry_state、错误隔离、路由）
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from agent_core.capability_registry import Capability, clear_registry, list_capabilities, register
from agent_core.capabilities.base import (
    CapabilityProvider,
    ToolProviderConfigError,
    _PROVIDER_REGISTRY,
    all_registered_categories,
    get_provider_class,
    register_provider,
)
from agent_core.capabilities.bootstrap import (
    ToolsConfig,
    _reset_capabilities_for_testing,
    bootstrap_capabilities,
    build_capability_map,
    get_enabled_capabilities,
)
from agent_core.capabilities.providers.shell_provider import ShellCapabilityProvider
from agent_core.capabilities.providers.file_provider import FileCapabilityProvider
from agent_core.capabilities.providers.log_provider import LogCapabilityProvider
from agent_core.capabilities.providers.sqlite_provider import SqliteCapabilityProvider
from agent_core.capabilities.providers.artifact_provider import (
    ArtifactCapabilityProvider,
)
from agent_core.artifacts import ArtifactStore
from agent_core.telemetry import telemetry_task
from agent_core.capabilities.providers.skills_provider import (
    SkillsCapabilityProvider,
    SkillsToolConfig,
    _load_skill_file,
)
from agent_core.exceptions import (
    AgentCoreError,
    ExecutionError,
    PlanningError,
    ReflectionError,
)
from agent_core.graph.state import AgentState
from agent_core.graph.build_graph import (
    _fail_state,
    _normalize_entry_state,
    _route_after_executor,
    _route_after_planner,
    _route_after_reflector,
    _with_error_isolation,
    build_graph,
)
from agent_core.config import load_tools_config

# ── 辅助函数 ────────────────────────────────────────────────────────────────


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


# ── 测试夹具 ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(_PROVIDER_REGISTRY)
    _reset_capabilities_for_testing()
    yield
    _PROVIDER_REGISTRY.clear()
    _PROVIDER_REGISTRY.update(saved)
    _reset_capabilities_for_testing()


# ============================================================================
# base.py 测试
# ============================================================================


class TestAutoDiscovery:
    def test_default_providers_registered(self):
        categories = all_registered_categories()
        assert "shell" in categories
        assert "file" in categories
        assert "log" in categories
        assert "sqlite" in categories
        assert "artifact" in categories
        assert "skills" in categories

    def test_new_provider_via_decorator(self):
        @register_provider
        class _Fake(CapabilityProvider):
            category = "p1-test"
            def build(self, raw_config):
                return []

        assert "p1-test" in all_registered_categories()

    def test_duplicate_category_raises(self):
        @register_provider
        class _First(CapabilityProvider):
            category = "dup-test"
            def build(self, raw_config):
                return []

        with pytest.raises(ValueError, match="dup-test"):
            @register_provider
            class _Second(CapabilityProvider):
                category = "dup-test"
                def build(self, raw_config):
                    return []


class TestIsolatedFailure:
    def test_failing_provider_blocked_others_ok(self, caplog):
        class _HealthyTool(BaseTool):
            name: str = "healthy"
            description: str = "h"
            def _run(self, *args, **kwargs):
                return "ok"

        class _HealthyProvider(CapabilityProvider):
            category = "test-healthy"
            def build(self, raw_config):
                cap = Capability(name="healthy", description="h",
                                 input_schema={"type": "object", "properties": {}},
                                 handler=lambda: "ok")
                return [cap]

        class _FailingProvider(CapabilityProvider):
            category = "test-failing"
            def build(self, raw_config):
                raise ToolProviderConfigError("intentional")

        # 模拟通过 @register_provider 注册
        _PROVIDER_REGISTRY["test-healthy"] = _HealthyProvider
        _PROVIDER_REGISTRY["test-failing"] = _FailingProvider

        config = ToolsConfig(providers={"test-healthy": {}, "test-failing": {}})
        with caplog.at_level(logging.WARNING):
            cap_map = build_capability_map(config)

        assert "healthy" in cap_map
        assert any("test-failing" in r.message for r in caplog.records)


class TestBuildCapabilityMap:
    def test_enabled_tools_filtering(self):
        # 注册测试 Provider
        class _TestProvider(CapabilityProvider):
            category = "test-filter"
            def build(self, raw_config):
                return [
                    Capability(name="tool-a", description="a", input_schema={}, handler=lambda: "a"),
                    Capability(name="tool-b", description="b", input_schema={}, handler=lambda: "b"),
                ]
        _PROVIDER_REGISTRY["test-filter"] = _TestProvider

        config = ToolsConfig(
            enabled_tools=["tool-a"],
            providers={"test-filter": {}},
        )
        caps = get_enabled_capabilities(config)
        assert len(caps) == 1
        assert caps[0].name == "tool-a"

    def test_empty_enabled_tools_returns_all(self):
        class _AllProvider(CapabilityProvider):
            category = "test-all"
            def build(self, raw_config):
                return [Capability(name="all-tool", description="x", input_schema={}, handler=lambda: "x")]
        _PROVIDER_REGISTRY["test-all"] = _AllProvider

        config = ToolsConfig(providers={"test-all": {}})
        caps = get_enabled_capabilities(config)
        assert any(c.name == "all-tool" for c in caps)


class TestBootstrap:
    def test_bootstrap_registers_capabilities(self):
        class _BootProvider(CapabilityProvider):
            category = "test-boot"
            def build(self, raw_config):
                return [
                    Capability(
                        name="boot-tool",
                        description="A bootstrapped tool",
                        input_schema={
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                            "required": ["x"],
                        },
                        handler=lambda x: f"got {x}",
                    )
                ]
        _PROVIDER_REGISTRY["test-boot"] = _BootProvider

        clear_registry()
        config = ToolsConfig(providers={"test-boot": {}})
        bootstrap_capabilities(config)

        caps = list_capabilities()
        assert any(c.name == "boot-tool" for c in caps)

    def test_bootstrap_double_fails_fast(self):
        class _DoubleProvider(CapabilityProvider):
            category = "test-double"
            def build(self, raw_config):
                return [Capability(name="dt", description="d", input_schema={}, handler=lambda: "d")]
        _PROVIDER_REGISTRY["test-double"] = _DoubleProvider

        clear_registry()
        config = ToolsConfig(providers={"test-double": {}})
        bootstrap_capabilities(config)
        with pytest.raises(ValueError):
            bootstrap_capabilities(config)


# ============================================================================
# 内置 Provider 测试
# ============================================================================


class TestShellProvider:
    def test_build_returns_one_capability(self):
        provider = ShellCapabilityProvider()
        caps = provider.build({})
        assert len(caps) == 1
        assert caps[0].name == "execute_shell_command"

    def test_handler_allowed_cmd(self):
        provider = ShellCapabilityProvider()
        caps = provider.build({"allowed_commands": ["echo"]})
        result = json.loads(caps[0].handler(command="echo hello"))
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_handler_blocked_cmd(self):
        provider = ShellCapabilityProvider()
        caps = provider.build({"allowed_commands": ["echo"]})
        result = json.loads(caps[0].handler(command="rm -rf /"))
        assert result["success"] is False
        assert "not in the allowlist" in result["error"].lower()

    def test_invalid_config_raises(self):
        provider = SkillsCapabilityProvider()
        with pytest.raises(ToolProviderConfigError):
            provider.build({"skills_dir": None})


class TestFileProvider:
    def test_metadata_read_and_search(self, tmp_path):
        target = tmp_path / "notes.txt"
        target.write_text("alpha\nrunbook=RB-2048\nomega", encoding="utf-8")
        capabilities = {
            item.name: item
            for item in FileCapabilityProvider().build(
                {"allowed_roots": [str(tmp_path)]}
            )
        }
        metadata = json.loads(
            capabilities["get_file_metadata"].handler(file_path=str(target))
        )
        content = json.loads(
            capabilities["read_file"].handler(
                file_path=str(target), offset=6, length=15
            )
        )
        matches = json.loads(
            capabilities["search_file"].handler(
                file_path=str(target), query="RB-2048"
            )
        )
        assert metadata["size_bytes"] == target.stat().st_size
        assert "runbook" in content["content"]
        assert matches["matches"][0]["line_number"] == 2

    def test_rejects_path_outside_roots(self, tmp_path):
        capabilities = {
            item.name: item
            for item in FileCapabilityProvider().build(
                {"allowed_roots": [str(tmp_path / "allowed")]}
            )
        }
        result = json.loads(
            capabilities["read_file"].handler(file_path="/etc/hosts")
        )
        assert result["success"] is False
        assert "outside configured roots" in result["error"]


class TestLogProvider:
    def test_aggregate_search_and_window(self, tmp_path):
        log = tmp_path / "incident.log"
        log.write_text(
            "INFO code=START node=api\n"
            "WARN code=POOL_PRESSURE node=db\n"
            "ERROR code=DB_POOL_EXHAUSTED node=db\n",
            encoding="utf-8",
        )
        capabilities = {
            item.name: item
            for item in LogCapabilityProvider().build(
                {"allowed_roots": [str(tmp_path)]}
            )
        }
        aggregate = json.loads(
            capabilities["aggregate_log_errors"].handler(log_path=str(log))
        )
        search = json.loads(
            capabilities["search_log"].handler(
                log_path=str(log), query="DB_POOL_EXHAUSTED"
            )
        )
        window = json.loads(
            capabilities["get_log_window"].handler(
                log_path=str(log), line_number=3, before=1, after=0
            )
        )
        assert aggregate["levels"]["ERROR"] == 1
        assert search["matches"][0]["line_number"] == 3
        assert window["lines"][0]["line_number"] == 2


class TestSqliteProvider:
    def test_describe_select_and_reject_write(self, tmp_path):
        db = tmp_path / "test.sqlite"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE incidents (id TEXT, code TEXT)")
            connection.execute(
                "INSERT INTO incidents VALUES ('req-1', 'DB_POOL_EXHAUSTED')"
            )
        capabilities = {
            item.name: item
            for item in SqliteCapabilityProvider().build(
                {"allowed_roots": [str(tmp_path)]}
            )
        }
        described = json.loads(
            capabilities["describe_sqlite_table"].handler(
                db_path=str(db), table_name="incidents"
            )
        )
        selected = json.loads(
            capabilities["query_sqlite"].handler(
                db_path=str(db), query="SELECT * FROM incidents"
            )
        )
        selected_with_null_limit = json.loads(
            capabilities["query_sqlite"].handler(
                db_path=str(db),
                query="SELECT * FROM incidents",
                max_rows=None,
            )
        )
        rejected = json.loads(
            capabilities["query_sqlite"].handler(
                db_path=str(db), query="DELETE FROM incidents"
            )
        )
        assert [item["name"] for item in described["columns"]] == ["id", "code"]
        assert selected["rows"][0]["id"] == "req-1"
        assert selected_with_null_limit["rows"][0]["id"] == "req-1"
        assert rejected["success"] is False
        with sqlite3.connect(db) as connection:
            assert connection.execute("SELECT count(*) FROM incidents").fetchone()[0] == 1


class TestArtifactProvider:
    def test_read_tools_enforce_task_ownership(self, tmp_path):
        store = ArtifactStore(tmp_path)
        metadata = store.put(
            "prefix DB_POOL_EXHAUSTED suffix",
            owner_id="task-a",
            tool_name="search_log",
            summary="one database error",
        )
        capabilities = {
            item.name: item
            for item in ArtifactCapabilityProvider().build(
                {"storage_dir": str(tmp_path)}
            )
        }
        with telemetry_task("task-a"):
            summary = json.loads(
                capabilities["get_artifact_summary"].handler(
                    artifact_id=metadata.artifact_id
                )
            )
            search = json.loads(
                capabilities["search_artifact"].handler(
                    artifact_id=metadata.artifact_id,
                    query="DB_POOL_EXHAUSTED",
                )
            )
            chunk = json.loads(
                capabilities["retrieve_artifact"].handler(
                    artifact_id=metadata.artifact_id,
                    offset=7,
                    length=17,
                )
            )
        with telemetry_task("task-b"):
            denied = json.loads(
                capabilities["get_artifact_summary"].handler(
                    artifact_id=metadata.artifact_id
                )
            )
        assert summary["summary"] == "one database error"
        assert search["match_count"] == 1
        assert "DB_POOL" in chunk["content"]
        assert denied["success"] is False


class TestSkillsProvider:
    def test_string_skills_dir_from_toml_is_converted(self, tmp_path):
        provider = SkillsCapabilityProvider()
        caps = provider.build({"skills_dir": str(tmp_path)})
        assert caps == []

    def test_missing_dir_returns_empty(self, caplog):
        provider = SkillsCapabilityProvider()
        with caplog.at_level(logging.WARNING):
            caps = provider.build({"skills_dir": Path("/no/such/dir")})
        assert caps == []

    def test_good_and_bad_yaml(self, tmp_path, caplog):
        sd = tmp_path / "skills"
        sd.mkdir()
        _write_yaml(sd / "good.yaml", {
            "name": "good",
            "description": "A good skill",
            "kind": "shell_template",
            "command_template": "echo hello",
        })
        _write_yaml(sd / "bad.yaml", {
            "description": "Missing name",
            "kind": "shell_template",
            "command_template": "echo fail",
        })

        provider = SkillsCapabilityProvider()
        with caplog.at_level(logging.WARNING):
            caps = provider.build({"skills_dir": sd})
        assert len(caps) == 1
        assert caps[0].name == "good"

    def test_real_yaml_files_load(self):
        skills_dir = Path(__file__).resolve().parent.parent / "agent_core" / "capabilities" / "skills"
        if not skills_dir.exists():
            pytest.skip("skills dir not found")
        provider = SkillsCapabilityProvider()
        caps = provider.build({"skills_dir": skills_dir})
        assert len(caps) >= 1


# ============================================================================
# build_graph.py V3 测试
# ============================================================================


class TestNormalizeEntryState:
    def test_missing_fields_filled(self):
        state: AgentState = {"task_goal": "test"}
        state = _normalize_entry_state(state)  # type: ignore[arg-type]
        assert state["plan_steps"] == []
        assert state["execution_log"] == []
        assert state["current_step_index"] == 0
        assert state["current_iteration"] == 0
        assert state["max_iterations"] == 10
        assert state["status"] == "planning"

    def test_existing_fields_preserved(self):
        state: AgentState = {
            "task_goal": "t",
            "plan_steps": ["already set"],
            "current_step_index": 1,
            "execution_log": [{"step": "s", "result": "r", "tool_used": None}],
            "reflection_notes": ["n"],
            "status": "executing",
            "max_iterations": 5,
            "current_iteration": 2,
        }
        state = _normalize_entry_state(state)  # type: ignore[arg-type]
        assert state["plan_steps"] == ["already set"]
        assert state["current_step_index"] == 1
        assert state["max_iterations"] == 5


class TestErrorIsolation:
    def test_planner_error_converted_to_failed(self):
        def _bad_planner(state: AgentState) -> AgentState:
            raise PlanningError("bad plan")

        wrapped = _with_error_isolation(_bad_planner, "planner")
        state: AgentState = {
            "task_goal": "test", "plan_steps": [], "current_step_index": 0,
            "execution_log": [], "reflection_notes": [], "status": "planning",
            "max_iterations": 10, "current_iteration": 0,
        }
        result = wrapped(state)
        assert result["status"] == "failed"
        assert any("planner" in n for n in result["reflection_notes"])

    def test_executor_error_converted_with_config(self):
        def _bad_executor(state: AgentState, config) -> AgentState:
            raise ExecutionError("exec fail")

        wrapped = _with_error_isolation(_bad_executor, "executor")
        state: AgentState = {
            "task_goal": "test", "plan_steps": ["s"], "current_step_index": 0,
            "execution_log": [], "reflection_notes": [], "status": "executing",
            "max_iterations": 10, "current_iteration": 0,
        }
        result = wrapped(state, {"configurable": {"thread_id": "t"}})
        assert result["status"] == "failed"

    def test_non_agent_core_error_propagates(self):
        def _bad_node(state: AgentState) -> AgentState:
            raise RuntimeError("unexpected")

        wrapped = _with_error_isolation(_bad_node, "test")
        state: AgentState = {
            "task_goal": "test", "plan_steps": [], "current_step_index": 0,
            "execution_log": [], "reflection_notes": [], "status": "planning",
            "max_iterations": 10, "current_iteration": 0,
        }
        with pytest.raises(RuntimeError):
            wrapped(state)


class TestRoutingV3:
    def test_route_after_planner_failed(self):
        state: AgentState = {
            "task_goal": "t", "plan_steps": [], "current_step_index": 0,
            "execution_log": [], "reflection_notes": [], "status": "failed",
            "max_iterations": 10, "current_iteration": 0,
        }
        from langgraph.graph import END
        assert _route_after_planner(state) == END

    def test_route_after_executor_failed(self):
        state: AgentState = {
            "task_goal": "t", "plan_steps": ["s"], "current_step_index": 0,
            "execution_log": [], "reflection_notes": [], "status": "failed",
            "max_iterations": 10, "current_iteration": 0,
        }
        from langgraph.graph import END
        assert _route_after_executor(state) == END


class TestBuildGraphV3:
    def test_graph_has_normalize_state_node(self):
        g = build_graph()
        nodes = g.get_graph().nodes
        node_names = {n for n in nodes}  # type: ignore[var-annotated]
        assert "normalize_state" in node_names
        assert "planner" in node_names
        assert "executor" in node_names
        assert "reflector" in node_names


# ============================================================================
# load_tools_config 集成测试
# ============================================================================


class TestLoadToolsConfig:
    def test_from_toml(self, tmp_path):
        f = tmp_path / "t.toml"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            '[tools]\n'
            'enabled_tools = ["execute_shell_command"]\n\n'
            '[tools.providers.shell]\n'
            'allowed_commands = ["ls", "cat"]\n'
        )
        cfg = load_tools_config(config_file=f)
        assert cfg.enabled_tools == ["execute_shell_command"]
        assert cfg.providers["shell"]["allowed_commands"] == ["ls", "cat"]

    def test_defaults(self, monkeypatch, tmp_path):
        # 避免自动探测仓库中的真实 agent_config.toml。
        monkeypatch.chdir(tmp_path)
        cfg = load_tools_config()
        assert cfg.enabled_tools == []
        assert cfg.providers == {}


# ============================================================================
# AgentState 类型
# ============================================================================


def test_agent_state_backward_compat():
    """确保 AgentState 仍包含手工构造它的测试所需的全部字段。"""
    state: AgentState = {
        "task_goal": "t",
        "plan_steps": [],
        "current_step_index": 0,
        "execution_log": [],
        "reflection_notes": [],
        "status": "planning",
        "max_iterations": 10,
        "current_iteration": 0,
    }
    # 只验证构造过程不会失败
    assert state["task_goal"] == "t"
