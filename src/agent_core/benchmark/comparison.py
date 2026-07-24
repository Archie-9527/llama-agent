"""Cross-round R0/R1/R2 comparison report and dependency-free SVG charts."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any


ROUND_ORDER = ("R0", "R1", "R2")
ROUND_COLORS = {
    "R0": "#64748b",
    "R1": "#2563eb",
    "R2": "#16a34a",
}


def write_comparison_report(
    output_dir: Path,
    run_dirs: dict[str, Path],
) -> Path:
    """Create structured comparison data, Markdown and SVG charts."""
    ordered = {
        round_name: Path(run_dirs[round_name])
        for round_name in ROUND_ORDER
    }
    manifests = {
        name: _read_json(path / "manifest.json")
        for name, path in ordered.items()
    }
    summaries = {
        name: _read_json(path / "summary.json")
        for name, path in ordered.items()
    }
    warnings = _comparison_warnings(manifests, summaries)
    case_metrics = {
        name: _case_metrics(path)
        for name, path in ordered.items()
    }
    failures = {
        name: _failure_summary(path)
        for name, path in ordered.items()
    }
    data = {
        "rounds": {
            name: {
                "run_dir": str(path),
                "manifest": manifests[name],
                "summary": summaries[name],
                "cases": case_metrics[name],
                "failures": failures[name],
            }
            for name, path in ordered.items()
        },
        "warnings": warnings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    _write_charts(charts_dir, summaries)

    lines = _report_lines(ordered, manifests, summaries, case_metrics, failures, warnings)
    report = output_dir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _comparison_warnings(
    manifests: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    reference = manifests["R0"]
    for name in ("R1", "R2"):
        current = manifests[name]
        for field, label in (
            ("model", "模型"),
            ("engine_config", "推理参数"),
        ):
            reference_value = reference[field]
            current_value = current[field]
            if field == "model":
                reference_value = reference_value.get("sha256")
                current_value = current_value.get("sha256")
            if current_value != reference_value:
                warnings.append(f"{name} 的{label}与 R0 不一致，不能严格对比。")
        for key in ("name", "warmup_runs", "measured_runs", "case_filter"):
            if current["suite"].get(key) != reference["suite"].get(key):
                warnings.append(
                    f"{name} 的 Suite 字段 {key} 与 R0 不一致。"
                )
    if any(
        manifest.get("environment", {}).get("dirty_worktree")
        for manifest in manifests.values()
    ):
        warnings.append(
            "至少一轮运行时工作区不是干净状态；正式报告应在固定 commit 上重跑。"
        )
    source_hashes = {
        manifest.get("environment", {}).get("source_tree_sha256")
        for manifest in manifests.values()
        if manifest.get("environment", {}).get("source_tree_sha256")
    }
    if len(source_hashes) > 1:
        warnings.append(
            "R0/R1/R2 的源码树指纹不一致，三轮不是同一份实现，不能严格对比。"
        )
    for previous, current in (("R0", "R1"), ("R1", "R2")):
        if (
            summaries[current]["task_success_rate"]
            < summaries[previous]["task_success_rate"]
        ):
            warnings.append(
                f"{current} 成功率低于 {previous}；该轮总体延迟、Token 和"
                "存储下降可能受到提前失败影响，不能直接视为优化收益。"
            )
    return list(dict.fromkeys(warnings))


def _case_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    results = [
        item
        for item in _read_jsonl(run_dir / "task_results.jsonl")
        if item.get("measured", True)
    ]
    measured_keys = {
        (str(item["case_id"]), f"rep-{int(item.get('repetition', 0)):03d}")
        for item in results
    }
    inference = [
        item
        for item in _read_jsonl(run_dir / "inference_events.jsonl")
        if (item.get("case_id"), item.get("sample")) in measured_keys
        and item.get("event") == "inference_completed"
    ]
    kv_rows = [
        item
        for item in _read_csv(run_dir / "kv_metrics.csv")
        if (item.get("case_id"), item.get("sample")) in measured_keys
    ]
    output: dict[str, dict[str, Any]] = {}
    for case_id in dict.fromkeys(str(item["case_id"]) for item in results):
        case_results = [item for item in results if item["case_id"] == case_id]
        case_inference = [
            item for item in inference if item.get("case_id") == case_id
        ]
        case_kv = [item for item in kv_rows if item.get("case_id") == case_id]
        sample_count = len(case_results)
        output[case_id] = {
            "sample_count": sample_count,
            "passed_count": sum(
                bool(item.get("evaluation", {}).get("passed"))
                for item in case_results
            ),
            "success_rate": (
                sum(
                    bool(item.get("evaluation", {}).get("passed"))
                    for item in case_results
                )
                / sample_count
                if sample_count
                else 0.0
            ),
            "duration_mean_ms": _mean(
                [float(item.get("duration_ms", 0)) for item in case_results]
            ),
            "inference_calls_per_sample": (
                len(case_inference) / sample_count if sample_count else 0.0
            ),
            "input_tokens_per_sample": (
                sum(int(item.get("input_tokens", 0)) for item in case_inference)
                / sample_count
                if sample_count
                else 0.0
            ),
            "output_tokens_per_sample": (
                sum(int(item.get("output_tokens", 0)) for item in case_inference)
                / sample_count
                if sample_count
                else 0.0
            ),
            "peak_logical_kv_tokens": max(
                (
                    float(item["logical_tokens"])
                    for item in case_kv
                    if item.get("logical_tokens") not in ("", None)
                ),
                default=0.0,
            ),
        }
    return output


def _failure_summary(run_dir: Path) -> dict[str, dict[str, Any]]:
    failures = [
        item
        for item in _read_jsonl(run_dir / "task_results.jsonl")
        if item.get("measured", True)
        and not item.get("evaluation", {}).get("passed")
    ]
    output: dict[str, dict[str, Any]] = {}
    for case_id in dict.fromkeys(str(item["case_id"]) for item in failures):
        case_failures = [item for item in failures if item["case_id"] == case_id]
        false_checks = Counter(
            key
            for item in case_failures
            for key, passed in item.get("evaluation", {})
            .get("checks", {})
            .items()
            if not passed
        )
        error = next(
            (
                str(item.get("error"))
                for item in case_failures
                if item.get("error")
            ),
            "",
        )
        output[case_id] = {
            "count": len(case_failures),
            "statuses": sorted(
                {str(item.get("status")) for item in case_failures}
            ),
            "false_checks": dict(false_checks),
            "failed_turn_indexes": sorted(
                {
                    int(item["failed_turn_index"])
                    for item in case_failures
                    if item.get("failed_turn_index") is not None
                }
            ),
            "example_error": error[:1000],
        }
    return output


def _report_lines(
    run_dirs: dict[str, Path],
    manifests: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    case_metrics: dict[str, dict[str, dict[str, Any]]],
    failures: dict[str, dict[str, dict[str, Any]]],
    warnings: list[str],
) -> list[str]:
    lines = [
        "# R0–R2 自动消融测试报告",
        "",
        "## 实验范围",
        "",
        f"- Suite：`{manifests['R0']['suite']['name']}`",
        f"- 模型 SHA-256：`{manifests['R0']['model']['sha256']}`",
        f"- 测量重复：{manifests['R0']['suite']['measured_runs']}",
        f"- Warmup：{manifests['R0']['suite']['warmup_runs']}",
        "- R0：关闭全部内存优化。",
        "- R1：仅开启工具输出 Artifact 虚拟化。",
        "- R2：在 R1 上增加生命周期上下文管理。",
        "",
        "各轮原始结果：",
        "",
    ]
    for name in ROUND_ORDER:
        lines.append(f"- {name}：`{run_dirs[name]}`")
    if warnings:
        lines.extend(["", "## 可比性与警告", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "## 总体结果",
            "",
            "| 指标 | R0 | R1 | R2 |",
            "|---|---:|---:|---:|",
            _summary_row(
                "任务成功率",
                summaries,
                lambda item: f"{item['task_success_rate']:.2%}",
            ),
            _summary_row(
                "平均端到端耗时",
                summaries,
                lambda item: _ms(item["duration_ms"]["mean"]),
            ),
            _summary_row(
                "P95 端到端耗时",
                summaries,
                lambda item: _ms(item["duration_ms"]["p95"]),
            ),
            _summary_row(
                "推理调用次数",
                summaries,
                lambda item: str(item["inference_call_count"]),
            ),
            _summary_row(
                "累计输入 Token",
                summaries,
                lambda item: f"{item['input_tokens']['total']:,}",
            ),
            _summary_row(
                "峰值逻辑 KV Token",
                summaries,
                lambda item: _number(item["peak_logical_kv_tokens"]),
            ),
            _summary_row(
                "峰值 RSS",
                summaries,
                lambda item: _mib(item["peak_rss_bytes"]),
            ),
            _summary_row(
                "峰值 GPU 进程显存",
                summaries,
                lambda item: _mib(item["peak_gpu_process_bytes"]),
            ),
            _summary_row(
                "Checkpoint 总量",
                summaries,
                lambda item: _mib(item["checkpoint_bytes"]),
            ),
            _summary_row(
                "持久化存储总量",
                summaries,
                lambda item: _mib(
                    item.get(
                        "total_persistent_storage_bytes",
                        item["checkpoint_bytes"],
                    )
                ),
            ),
            "",
            "![任务成功率](charts/success_rate.svg)",
            "",
            "![平均端到端耗时](charts/latency_mean.svg)",
            "",
            "![累计输入 Token](charts/input_tokens.svg)",
            "",
            "![峰值逻辑 KV Token](charts/peak_kv.svg)",
            "",
            "![峰值 GPU 进程显存](charts/gpu_memory.svg)",
            "",
            "![Checkpoint 总量](charts/checkpoint.svg)",
            "",
            "## 相对变化",
            "",
            "正数表示相对基准减少，成功率使用百分点；只有各轮成功率可比时，"
            "延迟与资源变化才可解释为优化收益。",
            "",
            "| 指标 | R1 相对 R0 | R2 相对 R1 | R2 相对 R0 |",
            "|---|---:|---:|---:|",
            _change_row(
                "成功率变化",
                summaries,
                lambda item: item["task_success_rate"],
                percentage_points=True,
            ),
            _change_row(
                "平均耗时减少",
                summaries,
                lambda item: item["duration_ms"]["mean"],
            ),
            _change_row(
                "输入 Token 减少",
                summaries,
                lambda item: item["input_tokens"]["total"],
            ),
            _change_row(
                "峰值逻辑 KV 减少",
                summaries,
                lambda item: item["peak_logical_kv_tokens"],
            ),
            _change_row(
                "GPU 进程显存减少",
                summaries,
                lambda item: item["peak_gpu_process_bytes"],
            ),
            _change_row(
                "Checkpoint 减少",
                summaries,
                lambda item: item["checkpoint_bytes"],
            ),
            "",
            "## 分 Case 结果",
            "",
            "| Case | Round | 通过 | 平均耗时 | 输入 Token/样本 | "
            "调用/样本 | 峰值逻辑 KV |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    case_order = list(
        dict.fromkeys(
            case_id
            for name in ROUND_ORDER
            for case_id in case_metrics[name]
        )
    )
    for case_id in case_order:
        for name in ROUND_ORDER:
            metrics = case_metrics[name].get(case_id)
            if metrics is None:
                continue
            lines.append(
                f"| {case_id} | {name} | "
                f"{metrics['passed_count']}/{metrics['sample_count']} | "
                f"{_ms(metrics['duration_mean_ms'])} | "
                f"{metrics['input_tokens_per_sample']:.0f} | "
                f"{metrics['inference_calls_per_sample']:.1f} | "
                f"{metrics['peak_logical_kv_tokens']:.0f} |"
            )

    lines.extend(
        [
            "",
            "## R1 工具输出虚拟化",
            "",
            "| 指标 | R0 | R1 | R2 |",
            "|---|---:|---:|---:|",
            _summary_row(
                "虚拟化结果数",
                summaries,
                lambda item: str(item["virtualized_tool_result_count"]),
            ),
            _summary_row(
                "外置原始字节",
                summaries,
                lambda item: f"{item['externalized_tool_output_bytes']:,}",
            ),
            _summary_row(
                "减少模型内联字节",
                summaries,
                lambda item: f"{item['artifact_bytes_saved']:,}",
            ),
            _summary_row(
                "工具结果压缩率",
                summaries,
                lambda item: f"{item['artifact_reduction_ratio']:.2%}",
            ),
            "",
            "## R2 生命周期上下文",
            "",
            "| 指标 | R0 | R1 | R2 |",
            "|---|---:|---:|---:|",
            _summary_row(
                "上下文投影压缩率",
                summaries,
                lambda item: f"{item['context_reduction_ratio']:.2%}",
            ),
            _summary_row(
                "状态压缩事件",
                summaries,
                lambda item: str(item["context_compaction_count"]),
            ),
            _summary_row(
                "状态压缩减少字节",
                summaries,
                lambda item: f"{item['context_compaction_bytes_saved']:,}",
            ),
            _summary_row(
                "收益不足跳过记录",
                summaries,
                lambda item: str(
                    item.get("context_compaction_skipped_count", 0)
                ),
            ),
            _summary_row(
                "被压缩原文总量",
                summaries,
                lambda item: f"{item.get('context_compaction_source_bytes', 0):,}",
            ),
            _summary_row(
                "额外召回历史轮数",
                summaries,
                lambda item: str(item.get("context_recalled_turns", 0)),
            ),
            _summary_row(
                "最终选择历史轮数",
                summaries,
                lambda item: str(item.get("context_selected_turns", 0)),
            ),
            _summary_row(
                "注入历史 Token",
                summaries,
                lambda item: str(item.get("context_recalled_tokens", 0)),
            ),
            "",
            "## 失败样本",
            "",
        ]
    )
    if not any(failures.values()):
        lines.append("三轮正式测量样本全部通过。")
    else:
        for name in ROUND_ORDER:
            if not failures[name]:
                lines.append(f"- {name}：无失败样本。")
                continue
            for case_id, detail in failures[name].items():
                checks = ", ".join(
                    f"{key}×{count}"
                    for key, count in detail["false_checks"].items()
                )
                turns = (
                    f"，失败轮次={detail['failed_turn_indexes']}"
                    if detail["failed_turn_indexes"]
                    else ""
                )
                error = (
                    f"，示例错误：{detail['example_error']}"
                    if detail["example_error"]
                    else ""
                )
                lines.append(
                    f"- {name}/{case_id}：{detail['count']} 个失败，"
                    f"status={detail['statuses']}，检查项={checks}"
                    f"{turns}{error}"
                )

    r0 = summaries["R0"]
    r1 = summaries["R1"]
    r2 = summaries["R2"]
    lines.extend(["", "## 自动结论", ""])
    if (
        r1["task_success_rate"] >= r0["task_success_rate"]
        and r1["input_tokens"]["total"] < r0["input_tokens"]["total"]
    ):
        lines.append(
            "- R1 在成功率不下降的前提下降低了输入 Token，满足有效优化的"
            "基本条件。"
        )
    else:
        lines.append(
            "- R1 未同时满足“成功率不下降且输入 Token 降低”，需要检查回归。"
        )
    if r2["task_success_rate"] < r1["task_success_rate"]:
        lines.append(
            "- R2 相比 R1 存在成功率回归；修复前不得用总体延迟下降作为最终"
            "优化结论。"
        )
    else:
        lines.append(
            "- R2 成功率未低于 R1，可结合分 Case 延迟、Token 和上下文指标"
            "评估增量收益。"
        )
    return lines


def _write_charts(
    charts_dir: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    chart_specs = (
        (
            "success_rate.svg",
            "Task Success Rate",
            "%",
            {
                name: summaries[name]["task_success_rate"] * 100
                for name in ROUND_ORDER
            },
            100.0,
        ),
        (
            "latency_mean.svg",
            "Mean End-to-End Latency",
            "s",
            {
                name: (summaries[name]["duration_ms"]["mean"] or 0) / 1000
                for name in ROUND_ORDER
            },
            None,
        ),
        (
            "input_tokens.svg",
            "Total Input Tokens",
            "tokens",
            {
                name: summaries[name]["input_tokens"]["total"]
                for name in ROUND_ORDER
            },
            None,
        ),
        (
            "peak_kv.svg",
            "Peak Logical KV Tokens",
            "tokens",
            {
                name: summaries[name]["peak_logical_kv_tokens"] or 0
                for name in ROUND_ORDER
            },
            None,
        ),
        (
            "gpu_memory.svg",
            "Peak GPU Process Memory",
            "MiB",
            {
                name: (summaries[name]["peak_gpu_process_bytes"] or 0)
                / 1024**2
                for name in ROUND_ORDER
            },
            None,
        ),
        (
            "checkpoint.svg",
            "Checkpoint Storage",
            "MiB",
            {
                name: summaries[name]["checkpoint_bytes"] / 1024**2
                for name in ROUND_ORDER
            },
            None,
        ),
    )
    for filename, title, unit, values, maximum in chart_specs:
        _write_bar_chart(
            charts_dir / filename,
            title=title,
            unit=unit,
            values=values,
            maximum=maximum,
        )


def _write_bar_chart(
    path: Path,
    *,
    title: str,
    unit: str,
    values: dict[str, float],
    maximum: float | None,
) -> None:
    width, height = 720, 420
    left, right, top, bottom = 90, 40, 70, 70
    chart_width = width - left - right
    chart_height = height - top - bottom
    upper = maximum or max(values.values(), default=1.0) * 1.15
    if upper <= 0:
        upper = 1.0
    bar_width = 100
    gap = (chart_width - bar_width * len(ROUND_ORDER)) / (
        len(ROUND_ORDER) + 1
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" '
        'font-family="sans-serif" font-size="22" font-weight="700">'
        f"{escape(title)}</text>",
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" '
        'stroke="#334155" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + chart_height}" stroke="#334155" stroke-width="2"/>',
    ]
    for tick in range(5):
        value = upper * tick / 4
        y = top + chart_height - chart_height * tick / 4
        svg.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" '
                f'x2="{left + chart_width}" y2="{y:.1f}" '
                'stroke="#e2e8f0" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" '
                'font-family="sans-serif" font-size="12" fill="#475569">'
                f"{escape(_compact_number(value))}</text>",
            ]
        )
    for index, name in enumerate(ROUND_ORDER):
        value = float(values.get(name, 0))
        x = left + gap * (index + 1) + bar_width * index
        bar_height = chart_height * value / upper
        y = top + chart_height - bar_height
        svg.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" '
                f'height="{bar_height:.1f}" rx="4" '
                f'fill="{ROUND_COLORS[name]}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" '
                'text-anchor="middle" font-family="sans-serif" '
                'font-size="13" font-weight="700">'
                f"{escape(_compact_number(value))}</text>",
                f'<text x="{x + bar_width / 2:.1f}" '
                f'y="{top + chart_height + 28}" text-anchor="middle" '
                'font-family="sans-serif" font-size="15">'
                f"{name}</text>",
            ]
        )
    svg.append(
        f'<text x="20" y="{top - 18}" font-family="sans-serif" '
        f'font-size="12" fill="#475569">{escape(unit)}</text>'
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _summary_row(
    label: str,
    summaries: dict[str, dict[str, Any]],
    formatter,
) -> str:
    return (
        f"| {label} | {formatter(summaries['R0'])} | "
        f"{formatter(summaries['R1'])} | {formatter(summaries['R2'])} |"
    )


def _change_row(
    label: str,
    summaries: dict[str, dict[str, Any]],
    getter,
    *,
    percentage_points: bool = False,
) -> str:
    values = {name: getter(summaries[name]) for name in ROUND_ORDER}
    pairs = (("R0", "R1"), ("R1", "R2"), ("R0", "R2"))
    rendered = [
        _change(
            values[baseline],
            values[current],
            percentage_points=percentage_points,
        )
        for baseline, current in pairs
    ]
    return f"| {label} | {' | '.join(rendered)} |"


def _change(
    baseline: float | int | None,
    current: float | int | None,
    *,
    percentage_points: bool,
) -> str:
    if baseline is None or current is None:
        return "N/A"
    if percentage_points:
        return f"{(float(current) - float(baseline)) * 100:+.2f} pp"
    if float(baseline) == 0:
        return "N/A"
    reduction = (float(baseline) - float(current)) / float(baseline)
    return f"{reduction:+.2%}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value / 1000:.2f}s"


def _mib(value: float | None) -> str:
    return "N/A" if value is None else f"{value / 1024**2:.2f} MiB"


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.0f}"


def _compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


__all__ = ["write_comparison_report"]
