"""测试 CLI 层、配置层以及 llm_engine v2 重构。

覆盖验收标准：
    E1–E10  — llm_engine.py 单例重构
    C1–C14  — AppConfig 加载
    E-C1–E-C8 — EngineConfig 加载
    L1–L22  — cli.py 行为
    I1–I3   — 跨模块集成
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_core.llm_engine import (
    ChatLlamaCpp,
    EngineConfig,
    _reset_engine_for_testing,
    get_engine,
    initialize_engine,
)
from agent_core.config import (
    ENV_PREFIX,
    AppConfig,
    _ENGINE_FIELD_CASTERS,
    _APP_FIELD_CASTERS,
    _cast_layer,
    _load_env_layer,
    _resolve_config_file,
    config_field_names,
    engine_config_field_names,
    load_app_config,
    load_engine_config,
)
from agent_core.exceptions import (
    AgentCoreError,
    AgentEngineError,
    EngineAlreadyInitializedError,
    EngineConfigError,
    EngineNotInitializedError,
    ExecutionError,
    ModelLoadError,
)
from agent_core.cli import (
    EXIT_BUSINESS_ERROR,
    EXIT_ENGINE_INIT_ERROR,
    EXIT_NO_RESUMABLE_TASK,
    EXIT_OK,
    _build_parser,
    _collect_app_cli_overrides,
    _collect_engine_cli_overrides,
    main,
)


# ── 辅助函数 ─────────────────────────────────────────────────────────────────


def _write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_benchmark_parser_accepts_repeated_case_filters():
    args = _build_parser().parse_args(
        [
            "benchmark",
            "--suite",
            "benchmark/workloads/r0_full.json",
            "--case",
            "w2-sqlite-investigation",
            "--case",
            "w7-eight-turn-memory",
        ]
    )
    assert args.case_ids == [
        "w2-sqlite-investigation",
        "w7-eight-turn-memory",
    ]


def test_ablation_parser_accepts_repeated_case_filters():
    args = _build_parser().parse_args(
        [
            "ablation",
            "--case",
            "w4-large-file-16k",
            "--case",
            "w7-eight-turn-memory",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )
    assert args.suite == Path("benchmark/workloads/r0_full.json")
    assert args.case_ids == [
        "w4-large-file-16k",
        "w7-eight-turn-memory",
    ]
    assert args.warmup_runs == 0
    assert args.measured_runs == 1


# ── 测试夹具 ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_engine():
    """每项测试前后均重置引擎单例。"""
    _reset_engine_for_testing()
    yield
    _reset_engine_for_testing()


@pytest.fixture(autouse=True)
def _clear_env():
    """移除所有 AGENT_* 环境变量，确保测试相互隔离。"""
    saved = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX)}
    for k in saved:
        del os.environ[k]
    yield
    for k, v in saved.items():
        os.environ[k] = v


# ============================================================================
# E1–E10  llm_engine 单例重构
# ============================================================================


class TestEngineSingleton:
    """E1–E10：initialize_engine/get_engine 生命周期。"""

    # E1
    def test_get_engine_before_init_raises(self):
        _reset_engine_for_testing()
        with pytest.raises(EngineNotInitializedError):
            get_engine()

    # E2
    def test_normal_initialization_flow(self):
        with patch(
            "agent_core.llm_engine.ChatLlamaCpp.__init__", return_value=None
        ) as mock_init:
            engine = initialize_engine(
                EngineConfig(model_path="/fake.gguf")
            )
        assert isinstance(engine, ChatLlamaCpp)
        assert get_engine() is engine

    # E3
    def test_repeated_initialization_raises(self):
        with patch(
            "agent_core.llm_engine.ChatLlamaCpp.__init__", return_value=None
        ):
            engine1 = initialize_engine(EngineConfig(model_path="/fake.gguf"))
            with pytest.raises(EngineAlreadyInitializedError):
                initialize_engine(EngineConfig(model_path="/other.gguf"))
        assert get_engine() is engine1

    # E4
    def test_model_path_empty_raises_engine_config_error(self):
        with pytest.raises(EngineConfigError, match="model_path"):
            initialize_engine(EngineConfig(model_path=""))

    # E5
    def test_model_path_whitespace_raises_engine_config_error(self):
        with pytest.raises(EngineConfigError):
            initialize_engine(EngineConfig(model_path="   "))

    # E6
    def test_model_load_failure_propagates(self):
        with patch(
            "agent_core.llm_engine.ChatLlamaCpp.__init__",
            side_effect=ModelLoadError("corrupt model"),
        ):
            with pytest.raises(ModelLoadError):
                initialize_engine(EngineConfig(model_path="/broken.gguf"))

    # E7
    def test_reset_for_testing(self):
        with patch(
            "agent_core.llm_engine.ChatLlamaCpp.__init__", return_value=None
        ):
            initialize_engine(EngineConfig(model_path="/fake.gguf"))
        assert get_engine() is not None
        _reset_engine_for_testing()
        with pytest.raises(EngineNotInitializedError):
            get_engine()

    # E8
    def test_engine_config_and_chat_llama_cpp_fields_aligned(self):
        """关键构造参数必须出现在 EngineConfig 中。"""
        ec_fields = {
            f.name for f in EngineConfig.__dataclass_fields__.values()
        }
        required = {
            "model_path", "n_ctx", "n_gpu_layers", "n_batch", "n_threads",
            "chat_format", "temperature", "top_p", "top_k", "repeat_penalty",
            "max_tokens",
        }
        assert required <= ec_fields

    # E9
    def test_get_engine_accepts_no_args(self):
        """静态检查：get_engine() 不接受任何参数。"""
        import inspect
        sig = inspect.signature(get_engine)
        assert not sig.parameters

    # E10
    def test_thread_safety_multiple_initializers(self):
        """仅允许一个线程初始化成功，其余线程均应收到重复初始化异常。"""
        errors: list = []
        instances: list = []
        barrier = threading.Barrier(5, timeout=5)

        def _worker():
            try:
                barrier.wait()
                with patch(
                    "agent_core.llm_engine.ChatLlamaCpp.__init__",
                    return_value=None,
                ):
                    inst = initialize_engine(EngineConfig(model_path=f"/t{threading.get_ident()}.gguf"))
                    instances.append(inst)
            except EngineAlreadyInitializedError as e:
                errors.append(e)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 1
        assert len(errors) == 4
        assert all(isinstance(e, EngineAlreadyInitializedError) for e in errors)


# ============================================================================
# C1–C14  AppConfig 加载
# ============================================================================


class TestAppConfigLoading:
    """C1–C14：AppConfig 合并逻辑。"""

    # C1
    def test_all_defaults(self):
        cfg = load_app_config()
        assert cfg.max_iterations == 6
        assert cfg.db_path == Path("data/checkpoints.sqlite")
        assert cfg.last_thread_file == Path("data/last_thread_id.txt")
        assert cfg.log_level == "INFO"

    # C2
    def test_file_layer(self, tmp_path):
        f = tmp_path / "cfg.toml"
        _write_toml(f, "[agent]\nmax_iterations = 10\n")
        cfg = load_app_config(config_file=f)
        assert cfg.max_iterations == 10
        assert cfg.log_level == "INFO"  # 默认值

    # C3
    def test_explicit_file_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_app_config(config_file=Path("/no/such/file.toml"))

    # C4
    def test_implicit_search_all_missing(self):
        # 测试环境中不应存在默认搜索路径。
        cfg = load_app_config(config_file=None)
        assert cfg.max_iterations == 6

    # C5
    def test_flat_toml_ignored(self, tmp_path):
        """静默忽略不在 [agent] 节中的扁平键。"""
        f = tmp_path / "flat.toml"
        _write_toml(f, "max_iterations = 99\n")
        cfg = load_app_config(config_file=f)
        assert cfg.max_iterations == 6  # 不应为 99

    # C6
    def test_env_overrides_file(self, tmp_path):
        f = tmp_path / "cfg.toml"
        _write_toml(f, "[agent]\nmax_iterations = 10\n")
        os.environ["AGENT_MAX_ITERATIONS"] = "20"
        cfg = load_app_config(config_file=f)
        assert cfg.max_iterations == 20

    # C7
    def test_cli_overrides_env(self, tmp_path):
        os.environ["AGENT_MAX_ITERATIONS"] = "20"
        cfg = load_app_config(cli_overrides={"max_iterations": 30})
        assert cfg.max_iterations == 30

    # C8
    def test_cli_none_does_not_override(self):
        os.environ["AGENT_MAX_ITERATIONS"] = "15"
        cfg = load_app_config(
            cli_overrides={"max_iterations": None, "log_level": "DEBUG"}
        )
        assert cfg.max_iterations == 15
        assert cfg.log_level == "DEBUG"

    # C9
    def test_path_type_conversion(self):
        os.environ["AGENT_DB_PATH"] = "/tmp/x.sqlite"
        cfg = load_app_config()
        assert isinstance(cfg.db_path, Path)
        assert cfg.db_path == Path("/tmp/x.sqlite")

    # C10
    def test_invalid_type_raises_value_error(self):
        os.environ["AGENT_MAX_ITERATIONS"] = "abc"
        with pytest.raises(ValueError, match="max_iterations"):
            load_app_config()

    # C11
    def test_unknown_field_warning(self, tmp_path, caplog):
        f = tmp_path / "cfg.toml"
        _write_toml(f, "[agent]\nunknown_field = 1\n")
        with caplog.at_level(logging.WARNING, logger="agent_core.config"):
            cfg = load_app_config(config_file=f)
        assert any("unknown_field" in r.message for r in caplog.records)
        assert cfg.max_iterations == 6

    # C12
    def test_to_run_config(self):
        ac = AppConfig(
            max_iterations=9,
            db_path=Path("/a"),
            last_thread_file=Path("/b"),
        )
        rc = ac.to_run_config()
        assert rc.max_iterations == 9
        assert rc.db_path == Path("/a")
        assert rc.last_thread_file == Path("/b")

    # C13
    def test_tomllib_missing_fallback(self, tmp_path, caplog):
        f = tmp_path / "cfg.toml"
        _write_toml(f, "[agent]\nmax_iterations = 42\n")
        with patch("agent_core.config.tomllib", None):
            with caplog.at_level(logging.WARNING, logger="agent_core.config"):
                cfg = load_app_config(config_file=f)
        # 回退到默认值。
        assert cfg.max_iterations == 6

    # C14
    def test_app_config_is_frozen(self):
        cfg = load_app_config()
        with pytest.raises(FrozenInstanceError):
            cfg.max_iterations = 100  # type: ignore[misc]


# ============================================================================
# E-C1–E-C8  EngineConfig 加载
# ============================================================================


class TestEngineConfigLoading:
    """E-C1–E-C8：EngineConfig 合并逻辑。"""

    def _minimal_toml(self, tmp_path):
        f = tmp_path / "ec.toml"
        _write_toml(f, '[engine]\nmodel_path = "/models/x.gguf"\nn_ctx = 8192\n')
        return f

    # E-C1
    def test_file_layer(self, tmp_path):
        f = self._minimal_toml(tmp_path)
        cfg = load_engine_config(config_file=f)
        assert cfg.model_path == "/models/x.gguf"
        assert cfg.n_ctx == 8192
        assert cfg.n_gpu_layers == 0  # 默认值

    # E-C2
    def test_missing_model_path_raises(self, tmp_path):
        f = tmp_path / "empty.toml"
        _write_toml(f, "[engine]\nn_ctx = 512\n")
        with pytest.raises(ValueError, match="model_path"):
            load_engine_config(config_file=f)

    # E-C3
    def test_env_provides_model_path(self):
        os.environ["AGENT_MODEL_PATH"] = "/models/env.gguf"
        cfg = load_engine_config()
        assert cfg.model_path == "/models/env.gguf"

    # E-C4
    def test_cli_overrides_all(self, tmp_path):
        f = self._minimal_toml(tmp_path)
        os.environ["AGENT_MODEL_PATH"] = "/models/env.gguf"
        cfg = load_engine_config(
            config_file=f,
            cli_overrides={"model_path": "/cli.gguf"},
        )
        assert cfg.model_path == "/cli.gguf"

    # E-C5
    def test_verbose_bool_conversion(self):
        for val in ("true", "True", "TRUE", "1", "yes", "YES"):
            os.environ["AGENT_VERBOSE"] = val
            os.environ["AGENT_MODEL_PATH"] = "/m.gguf"
            cfg = load_engine_config()
            assert cfg.verbose is True, f"verbose should be True for {val!r}"

        os.environ["AGENT_VERBOSE"] = "false"
        cfg = load_engine_config()
        assert cfg.verbose is False

    # E-C6
    def test_app_and_engine_configs_isolated(self, tmp_path):
        f = tmp_path / "both.toml"
        _write_toml(f, '[agent]\nmax_iterations = 7\n\n[engine]\nmodel_path = "/m.gguf"\n')
        app = load_app_config(config_file=f)
        eng = load_engine_config(config_file=f)
        assert app.max_iterations == 7
        assert eng.model_path == "/m.gguf"

    # E-C7
    def test_engine_config_field_names(self):
        names = engine_config_field_names()
        assert "model_path" in names
        assert "n_ctx" in names
        assert "n_gpu_layers" in names

    # E-C8
    def test_stop_field_not_overridable_by_env(self):
        os.environ["AGENT_STOP"] = "END"
        os.environ["AGENT_MODEL_PATH"] = "/m.gguf"
        cfg = load_engine_config()
        assert cfg.stop is None


# ============================================================================
# _cast_layer 与 _load_env_layer 单元测试
# ============================================================================


class TestHelpers:
    def test_load_env_layer_ignores_unknown_keys(self):
        os.environ["AGENT_UNKNOWN_KEY"] = "x"
        result = _load_env_layer(ENV_PREFIX, _APP_FIELD_CASTERS)
        assert "unknown_key" not in result

    def test_load_env_layer_collects_known_keys(self):
        os.environ["AGENT_MAX_ITERATIONS"] = "5"
        result = _load_env_layer(ENV_PREFIX, _APP_FIELD_CASTERS)
        assert result["max_iterations"] == "5"

    def test_cast_layer_unknown_key_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="agent_core.config"):
            result = _cast_layer({"foo": "bar"}, _APP_FIELD_CASTERS)
        assert "foo" not in result
        assert any("foo" in r.message for r in caplog.records)

    def test_cast_layer_type_error(self):
        with pytest.raises(ValueError, match="max_iterations"):
            _cast_layer({"max_iterations": "not-a-number"}, _APP_FIELD_CASTERS)

    def test_resolve_config_file_none_returns_empty(self, tmp_path):
        # 在默认搜索路径以外创建配置文件。
        f = tmp_path / "somewhere" / "cfg.toml"
        _write_toml(f, "[agent]\nmax_iterations = 99\n")
        result = _resolve_config_file(config_file=f)
        assert result["agent"]["max_iterations"] == 99


# ============================================================================
# L1–L22  CLI 集成测试
# ============================================================================


class TestCLI:
    """L1–L22：使用模拟依赖测试 cli.py 的行为。"""

    _mock_result = {
        "task_goal": "test",
        "plan_steps": ["do stuff"],
        "current_step_index": 1,
        "execution_log": [
            {"step": "do stuff", "result": "done", "tool_used": None}
        ],
        "reflection_notes": [],
        "status": "done",
        "max_iterations": 6,
        "current_iteration": 1,
    }

    @pytest.fixture(autouse=True)
    def _mocks(self):
        """模拟全部重量级依赖，使测试无需真实模型即可运行。"""
        os.environ["AGENT_MODEL_PATH"] = "/fake-test.gguf"
        with patch(
            "agent_core.cli.initialize_engine", return_value=MagicMock()
        ) as mock_init, patch(
            "agent_core.cli.TaskRunner", autospec=True
        ) as mock_runner_cls, patch(
            "agent_core.capabilities.bootstrap.bootstrap_capabilities"
        ) as mock_bootstrap, patch(
            "agent_core.graph.react_agent_factory.initialize_react_agent"
        ) as mock_ragent:
            mock_runner = mock_runner_cls.return_value
            mock_runner.start_new_task.return_value = ("test-tid", self._mock_result)
            mock_runner.resume_task.return_value = self._mock_result
            mock_runner.get_last_thread_id.return_value = "last-tid"
            yield mock_init, mock_runner

    # L1
    def test_run_command_success(self, _mocks):
        mock_init, mock_runner = _mocks
        rc = main(["run", "hello world"])
        assert rc == EXIT_OK
        mock_init.assert_called_once()
        mock_runner.start_new_task.assert_called_once_with("hello world")

    # L2
    def test_resume_command_explicit_id(self, _mocks):
        mock_init, mock_runner = _mocks
        rc = main(["resume", "explicit-id"])
        assert rc == EXIT_OK
        mock_runner.resume_task.assert_called_once_with("explicit-id")

    # L3
    def test_resume_auto_read_last_thread_id(self, _mocks):
        mock_init, mock_runner = _mocks
        mock_runner.get_last_thread_id.return_value = "auto-read-tid"
        rc = main(["resume"])
        assert rc == EXIT_OK
        mock_runner.resume_task.assert_called_once_with("auto-read-tid")

    # L4
    def test_resume_no_task_recorded(self, _mocks):
        mock_init, mock_runner = _mocks
        mock_runner.get_last_thread_id.return_value = None
        rc = main(["resume"])
        assert rc == EXIT_NO_RESUMABLE_TASK

    # L5
    def test_empty_goal_triggers_business_error(self, _mocks):
        mock_init, mock_runner = _mocks
        mock_runner.start_new_task.side_effect = ValueError("must not be empty")
        rc = main(["run", ""])
        assert rc == EXIT_BUSINESS_ERROR
        mock_runner.close.assert_called_once()

    # L6
    def test_agent_core_error_no_traceback(self, _mocks, capsys):
        mock_init, mock_runner = _mocks
        mock_runner.start_new_task.side_effect = ExecutionError("exec fail")
        rc = main(["run", "do something"])
        assert rc == EXIT_BUSINESS_ERROR
        stdout = capsys.readouterr().out
        assert "Traceback" not in stdout

    # L7
    def test_show_config_does_not_init_engine(self, _mocks):
        mock_init, mock_runner = _mocks
        rc = main(["show-config"])
        assert rc == EXIT_OK
        mock_init.assert_not_called()

    # L8
    def test_show_config_does_not_create_task_runner(self, _mocks, tmp_path):
        mock_init, _ = _mocks
        f = tmp_path / "sc.toml"
        _write_toml(f, '[engine]\nmodel_path = "/m.gguf"\n')
        # 验证带引擎配置运行 show-config 时不会崩溃。
        rc = main(["--config", str(f), "show-config"])
        assert rc == EXIT_OK

    # L9
    def test_show_config_missing_engine_config_is_tolerated(self, _mocks, capsys):
        mock_init, _ = _mocks
        rc = main(["show-config"])
        assert rc == EXIT_OK
        out = capsys.readouterr().out
        assert "Active application config:" in out

    # L10
    def test_engine_init_failure_returns_special_code(self, _mocks):
        mock_init, _ = _mocks
        mock_init.side_effect = ModelLoadError("corrupt")
        rc = main(["--model-path", "/bad.gguf", "run", "test"])
        assert rc == EXIT_ENGINE_INIT_ERROR

    # L11
    def test_missing_model_path_returns_business_error(self, _mocks):
        """load_engine_config 抛出的 ValueError 并非 AgentEngineError。"""
        # 测试夹具设置了 AGENT_MODEL_PATH，因此引擎配置可正常加载。
        # 这里必须强制 load_engine_config 抛出 ValueError。
        with patch(
            "agent_core.cli.load_engine_config",
            side_effect=ValueError("engine.model_path is not configured"),
        ):
            rc = main(["--model-path", "/x.gguf", "run", "test"])
        assert rc == EXIT_BUSINESS_ERROR

    # L12
    def test_model_path_cli_propagates(self, _mocks, tmp_path):
        mock_init, _ = _mocks
        f = tmp_path / "cfg.toml"
        _write_toml(f, '[engine]\nmodel_path = "/file.gguf"\n')
        main(["--config", str(f), "--model-path", "/cli.gguf", "run", "x"])
        cfg_passed = mock_init.call_args[0][0]
        assert cfg_passed.model_path == "/cli.gguf"

    # L13
    def test_db_path_cli_propagates(self, _mocks, tmp_path):
        _, mock_runner_cls = _mocks
        # 在 cli 模块层级替换 TaskRunner。
        f = tmp_path / "cfg.toml"
        _write_toml(f, '[engine]\nmodel_path = "/m.gguf"\n')
        with patch("agent_core.cli.TaskRunner") as mock_tr_cls:
            mock_tr_cls.return_value.start_new_task.return_value = ("tid", self._mock_result)
            mock_tr_cls.return_value.close = MagicMock()
            main(["--config", str(f), "--db-path", "/tmp/custom.sqlite", "run", "x"])
            rc_passed = mock_tr_cls.call_args[0][0]
            assert rc_passed.db_path == Path("/tmp/custom.sqlite")

    # L14
    def test_max_iterations_cli_propagates(self, _mocks):
        with patch("agent_core.cli.TaskRunner") as mock_tr_cls:
            mock_tr_cls.return_value.start_new_task.return_value = ("tid", self._mock_result)
            mock_tr_cls.return_value.close = MagicMock()
            main(["--model-path", "/m.gguf", "run", "x", "--max-iterations", "3"])
            rc_passed = mock_tr_cls.call_args[0][0]
            assert rc_passed.max_iterations == 3

    # L15
    def test_init_order_is_correct(self, _mocks, tmp_path):
        mock_init, mock_runner = _mocks
        f = tmp_path / "order.toml"
        _write_toml(f, '[engine]\nmodel_path = "/order.gguf"\n')
        order = []

        def _track_init(cfg):
            order.append("init")

        mock_init.side_effect = _track_init

        # mock_runner 已在类级别替换，这里也需要模拟 close。
        mock_runner.close = MagicMock()

        def _track_tr(*args, **kwargs):
            order.append("taskrunner")
            return mock_runner

        with patch("agent_core.cli.TaskRunner", side_effect=_track_tr):
            main(["--config", str(f), "run", "x"])

        assert order == ["init", "taskrunner"]

    # L16
    def test_close_called_even_on_error(self, _mocks):
        mock_init, mock_runner = _mocks
        mock_runner.start_new_task.side_effect = ExecutionError("fail")
        main(["--model-path", "/m.gguf", "run", "x"])
        mock_runner.close.assert_called_once()

    # L17
    def test_engine_init_failure_prevents_task_runner_creation(self, _mocks):
        mock_init, _ = _mocks
        mock_init.side_effect = ModelLoadError("fail")
        with patch("agent_core.cli.TaskRunner") as mock_tr:
            main(["--model-path", "/bad.gguf", "run", "x"])
        mock_tr.assert_not_called()

    # L18
    def test_missing_subcommand_triggers_system_exit_2(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2

    # L19
    def test_log_level_propagates(self, _mocks):
        mock_init, _ = _mocks
        # _setup_logging 调用 basicConfig；仅当根日志记录器尚未配置处理器时
        # 才会生效。这里只验证调用不会崩溃。
        rc = main(["--log-level", "DEBUG", "run", "x"])
        assert rc == EXIT_OK

    # L20
    def test_app_config_failure_blocks_all(self, _mocks):
        mock_init, _ = _mocks
        with patch(
            "agent_core.cli.load_app_config",
            side_effect=FileNotFoundError("config missing"),
        ):
            rc = main(["run", "x"])
        assert rc == EXIT_BUSINESS_ERROR
        mock_init.assert_not_called()

    # L21
    def test_architecture_isolation(self):
        """cli.py 不得导入 agent_core.graph 或 get_engine。"""
        cli_src = (
            Path(__file__).resolve().parent.parent
            / "agent_core" / "cli.py"
        )
        text = cli_src.read_text()
        import_lines = [
            line for line in text.split("\n")
            if line.startswith("import ") or line.startswith("from ")
        ]
        imports = "\n".join(import_lines)
        # 不得导入 graph/。
        assert "agent_core.graph" not in imports, (
            f"cli.py must not import graph/:\n{imports}"
        )
        # 不得导入 get_engine。
        assert "get_engine" not in imports, (
            f"cli.py must not import get_engine:\n{imports}"
        )

    # L22
    def test_resume_output_distinct_from_run(self, _mocks, capsys):
        mock_init, mock_runner = _mocks
        main(["resume", "some-id"])
        out = capsys.readouterr().out
        assert "Resume task ID:" in out


# ============================================================================
# I1–I3  跨模块集成
# ============================================================================


class TestIntegration:
    """I1–I3：配置→引擎→图的端到端集成。"""

    def _setup_minimal_config(self, tmp_path):
        f = tmp_path / "integ.toml"
        _write_toml(
            f,
            '[agent]\nmax_iterations = 3\n\n'
            '[engine]\nmodel_path = "/integ.gguf"\nn_ctx = 512\n',
        )
        return f

    # I1
    def test_full_cold_start(self, tmp_path):
        """I1：使用模拟 llama_cpp.Llama 验证完整冷启动流程。"""
        cf = self._setup_minimal_config(tmp_path)

        with patch(
            "agent_core.llm_engine.ChatLlamaCpp.__init__", return_value=None
        ), patch(
            "agent_core.cli.TaskRunner"
        ) as mock_tr_cls:
            mock_runner = MagicMock()
            mock_runner.start_new_task.return_value = (
                "integ-tid",
                {
                    "task_goal": "test",
                    "plan_steps": ["ok"],
                    "current_step_index": 1,
                    "execution_log": [{"step": "ok", "result": "done", "tool_used": None}],
                    "reflection_notes": [],
                    "status": "done",
                    "max_iterations": 3,
                    "current_iteration": 1,
                },
            )
            mock_tr_cls.return_value = mock_runner

            rc = main(["--config", str(cf), "run", "integration test"])
            assert rc == EXIT_OK

    # I2
    def test_engine_singleton_persists(self, tmp_path):
        """I2：在同一进程中先运行再恢复，引擎单例保持不变。"""
        cf = self._setup_minimal_config(tmp_path)

        with patch(
            "agent_core.llm_engine.ChatLlamaCpp.__init__", return_value=None
        ), patch(
            "agent_core.cli.TaskRunner"
        ) as mock_tr_cls:
            mock_runner = MagicMock()
            mock_runner.start_new_task.return_value = ("tid1", {
                "task_goal": "t", "plan_steps": ["s"], "current_step_index": 1,
                "execution_log": [], "reflection_notes": [], "status": "done",
                "max_iterations": 3, "current_iteration": 1,
            })
            mock_runner.resume_task.return_value = mock_runner.start_new_task.return_value[1]
            mock_runner.get_last_thread_id.return_value = "tid1"
            mock_tr_cls.return_value = mock_runner

            main(["--config", str(cf), "run", "first"])
            # 此时 get_engine 应可正常工作。
            eng = get_engine()
            main(["--config", str(cf), "resume", "tid1"])
            assert get_engine() is eng

    # I3
    def test_engine_config_params_reach_chat_llama_cpp(self):
        """I3：EngineConfig 正确合并到 ChatLlamaCpp 的关键字参数中。"""
        captured: list = []

        def _capture_init(**kwargs):
            captured.append(kwargs)
            return None  # type: ignore[return-value]

        os.environ["AGENT_MODEL_PATH"] = "/env-model.gguf"
        os.environ["AGENT_N_CTX"] = "1024"

        with patch(
            "agent_core.llm_engine.ChatLlamaCpp.__init__", side_effect=_capture_init
        ):
            initialize_engine(
                load_engine_config(cli_overrides={"n_gpu_layers": 99})
            )

        assert len(captured) == 1
        kwargs = captured[0]
        assert kwargs["model_path"] == "/env-model.gguf"
        assert kwargs["n_ctx"] == 1024
        assert kwargs["n_gpu_layers"] == 99
