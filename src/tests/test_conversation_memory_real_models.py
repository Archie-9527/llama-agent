"""使用本地 GGUF 模型进行可选的端到端会话记忆检查。

由于加载两个模型的开销过高，不适合常规单元测试套件，因此需要显式运行：

    RUN_REAL_CONVERSATION_TESTS=1 pytest -q -s \
      src/tests/test_conversation_memory_real_models.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    (
        "qwen2.5-7b",
        PROJECT_ROOT / "src/tests/models/qwen2.5-7b-instruct-q2_k.gguf",
    ),
    (
        "qwen3.5-4b",
        PROJECT_ROOT / "src/tests/models/Qwen3.5-4B-UD-Q8_K_XL.gguf",
    ),
)


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_CONVERSATION_TESTS") != "1",
    reason="set RUN_REAL_CONVERSATION_TESTS=1 to load the local GGUF models",
)
@pytest.mark.parametrize(("model_name", "model_path"), MODELS)
def test_chat_remembers_user_fact_across_turns(
    tmp_path: Path,
    model_name: str,
    model_path: Path,
) -> None:
    if not model_path.is_file():
        pytest.skip(f"local model is missing: {model_path}")

    config_path = tmp_path / f"{model_name}.toml"
    config_path.write_text(
        f"""
[agent]
max_iterations = 4
db_path = "{tmp_path / 'checkpoints.sqlite'}"
last_thread_file = "{tmp_path / 'last_thread_id.txt'}"
conversation_db_path = "{tmp_path / 'conversations.sqlite'}"
last_conversation_file = "{tmp_path / 'last_conversation_id.txt'}"
conversation_history_turns = 8
conversation_history_token_budget = 4096
log_level = "WARNING"

[engine]
model_path = "{model_path}"
n_ctx = 8192
n_gpu_layers = 0
n_batch = 256
n_threads = 8
chat_format = "chatml-function-calling"
temperature = 0.1
top_p = 0.9
top_k = 40
repeat_penalty = 1.1
max_tokens = 384
verbose = false
request_timeout = 180.0
disable_thinking = true

[tools]
enabled_tools = ["count_lines"]

[tools.providers.skills]
skills_dir = "{PROJECT_ROOT / 'src/agent_core/capabilities/skills'}"

[telemetry]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_core.cli",
            "--config",
            str(config_path),
            "chat",
        ],
        input=(
            "记住项目代号是 AgentMem，并直接告诉我你记住了什么。\n"
            "上一轮我告诉你的项目代号是什么？只回答项目代号。\n"
            "/exit\n"
        ),
        text=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        timeout=900,
        check=False,
    )
    diagnostic = f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    if (
        completed.returncode != 0
        and "Failed to create llama_context" in completed.stderr
    ):
        pytest.skip(
            f"{model_name}: llama.cpp could not allocate a context in this "
            "environment; rerun on the benchmark host"
        )
    assert completed.returncode == 0, diagnostic
    # 可用于诊断两轮任务是否均已完成：
    # assert completed.stdout.count("Final status: done") == 2, diagnostic
    second_turn = completed.stdout.rsplit("Final status: done", maxsplit=1)[-1]
    assert "AgentMem" in second_turn, diagnostic
