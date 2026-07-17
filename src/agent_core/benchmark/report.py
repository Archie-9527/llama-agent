"""Generate a compact, deterministic Markdown report from structured data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(
    run_dir: Path, manifest: dict[str, Any], summary: dict[str, Any]
) -> Path:
    duration = summary["duration_ms"]
    lines = [
        f"# {manifest['round']} Benchmark 报告",
        "",
        "## 实验信息",
        "",
        f"- Run ID：`{manifest['run_id']}`",
        f"- Suite：`{manifest['suite']['name']}`",
        f"- 模型：`{manifest['model']['path']}`",
        f"- 模型 SHA-256：`{manifest['model']['sha256']}`",
        f"- Python：`{manifest['environment']['python']}`",
        f"- OS：`{manifest['environment']['platform']}`",
        "",
        "## 功能结果",
        "",
        f"- 测量样本：{summary['sample_count']}",
        f"- 通过：{summary['passed_count']}",
        f"- 失败：{summary['failed_count']}",
        f"- Task Success Rate：{summary['task_success_rate']:.2%}",
        "",
        "## 延迟与内存",
        "",
        f"- 端到端平均耗时：{_fmt(duration['mean'], ' ms')}",
        f"- 端到端 P95：{_fmt(duration['p95'], ' ms')}",
        f"- 峰值 RSS：{_mib(summary['peak_rss_bytes'])}",
        f"- 峰值 GPU 进程显存：{_mib(summary['peak_gpu_process_bytes'])}",
        f"- 峰值逻辑 KV tokens：{_fmt(summary['peak_logical_kv_tokens'])}",
        f"- 推理调用次数：{summary['inference_call_count']}",
        f"- 累计 input tokens：{summary['input_tokens']['total']}",
        f"- 累计 output tokens：{summary['output_tokens']['total']}",
        f"- 工具输出字节：{summary['tool_output_bytes']}",
        f"- Checkpoint 文件总量：{_mib(summary['checkpoint_bytes'])}",
        f"- Conversation 文件总量：{_mib(summary['conversation_bytes'])}",
        "",
        "## 分类结果",
        "",
        "| 分类 | 通过 | 总数 |",
        "|---|---:|---:|",
    ]
    for category, values in sorted(summary["categories"].items()):
        lines.append(f"| {category} | {values['passed']} | {values['total']} |")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "这是未启用内存优化的功能基线。KV 逻辑 token、KV 预分配容量和"
            "进程 RSS 是不同指标，不在报告中相互替代。失败样本保留在原始数据中。",
            "",
        ]
    )
    output = run_dir / "report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _fmt(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.2f}{suffix}"


def _mib(value: float | None) -> str:
    return "N/A" if value is None else f"{value / 1024**2:.2f} MiB"
