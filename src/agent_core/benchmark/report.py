"""Generate a compact, deterministic Markdown report from structured data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(
    run_dir: Path, manifest: dict[str, Any], summary: dict[str, Any]
) -> Path:
    duration = summary["duration_ms"]
    memory_enabled = any(manifest.get("memory_flags", {}).values())
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
        f"- Warmup样本：{summary.get('warmup_sample_count', 0)}",
        f"- Warmup通过：{summary.get('warmup_passed_count', 0)}",
        f"- Warmup失败：{summary.get('warmup_failed_count', 0)}",
        "",
        "## 延迟与内存",
        "",
        f"- 端到端平均耗时：{_fmt(duration['mean'], ' ms')}",
        f"- 端到端 P95（nearest-rank）：{_fmt(duration['p95'], ' ms')}",
        f"- 峰值 RSS：{_mib(summary['peak_rss_bytes'])}",
        f"- 峰值 GPU 进程显存：{_mib(summary['peak_gpu_process_bytes'])}",
        f"- 峰值逻辑 KV tokens：{_fmt(summary['peak_logical_kv_tokens'])}",
        f"- 推理调用次数：{summary['inference_call_count']}",
        f"- 累计 input tokens：{summary['input_tokens']['total']}",
        f"- 累计 output tokens：{summary['output_tokens']['total']}",
        f"- 工具输出字节：{summary['tool_output_bytes']}",
        f"- R1 虚拟化工具结果数：{summary.get('virtualized_tool_result_count', 0)}",
        f"- R1 外置原始字节：{summary.get('externalized_tool_output_bytes', 0)}",
        f"- R1 模型内联字节：{summary.get('virtualized_inline_bytes', 0)}",
        f"- R1 减少内联字节：{summary.get('artifact_bytes_saved', 0)}",
        f"- R1 工具结果压缩率：{summary.get('artifact_reduction_ratio', 0.0):.2%}",
        f"- 测量样本 Artifact 磁盘占用：{_mib(summary.get('artifact_storage_bytes'))}",
        f"- 测量样本 Checkpoint 文件总量：{_mib(summary['checkpoint_bytes'])}",
        f"- 测量样本 Conversation 文件总量：{_mib(summary['conversation_bytes'])}",
        f"- R2 上下文预算事件数：{summary.get('context_budget_event_count', 0)}",
        f"- R2 上下文投影前 tokens：{summary.get('context_tokens_before', 0)}",
        f"- R2 上下文投影后 tokens：{summary.get('context_tokens_after', 0)}",
        f"- R2 上下文投影压缩率：{summary.get('context_reduction_ratio', 0.0):.2%}",
        f"- R2 状态压缩事件数：{summary.get('context_compaction_count', 0)}",
        f"- R2 状态压缩减少字节：{summary.get('context_compaction_bytes_saved', 0)}",
        f"- R2 Conversation上下文选择次数：{summary.get('context_recall_count', 0)}",
        f"- R2 额外召回历史轮数：{summary.get('context_recalled_turns', 0)}",
        f"- R2 最终选择历史轮数：{summary.get('context_selected_turns', 0)}",
        f"- R2 注入历史 tokens：{summary.get('context_recalled_tokens', 0)}",
        f"- 测量样本 ContextStore 文件总量：{_mib(summary.get('context_storage_bytes'))}",
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
            (
                "本轮已启用至少一个内存优化开关。"
                if memory_enabled
                else "这是未启用内存优化的功能基线。"
            )
            + "KV 逻辑 token、KV 预分配容量和进程 RSS 是不同指标，不在报告中"
            "相互替代。失败样本保留在原始数据中。",
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
