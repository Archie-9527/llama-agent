# Llama Agent Baseline

基于 `llama-cpp-python`、LangChain 与 LangGraph 的本地工具智能体。当前版本包含
阶段一功能基线：Plan–Execute–Reflect–Finalize、多轮 Conversation、
checkpoint/resume、统一遥测和隔离进程 Benchmark；同时包含 R1 工具输出
虚拟化与 R2 生命周期感知上下文管理，两项优化均可独立关闭做消融实验。

完整架构、两阶段建设方案和 Benchmark 用例说明见
[`docs/整体系统设计说明书.md`](docs/整体系统设计说明书.md)。

## CLI

```bash
# 单个任务
llama-agent run "使用 count_lines 统计 src/agent_core/llm_engine.py 的行数"

# 新建或续接外部多轮会话
llama-agent continue "记住项目代号是 llama-agent"
llama-agent continue "上一轮的项目代号是什么"
llama-agent chat
llama-agent cli
llama-agent list-conversations

# 恢复中断任务
llama-agent resume [thread_id]

# 查看合并后的配置
llama-agent show-config
```

每个成功任务都通过 `AgentState.final_answer` 返回统一最终答案；失败任务通过
`AgentState.error` 返回结构化错误。

### 全屏交互式 CLI

`llama-agent cli` 使用与 `chat` 相同的持久化 Conversation 后端，但提供全屏
终端界面、运行阶段提示、工具执行卡片，以及 Token、KV、RSS/GPU 内存状态栏。
模型和 LangGraph 仍使用稳定的同步调用；等待期间界面显示滚动省略号，任务
完成后一次性展示原始 Thinking（可选）和最终回答。

```bash
llama-agent --config agent_config.toml cli
llama-agent --config agent_config.toml cli --conversation-id <id>
llama-agent --config agent_config.toml cli --show-thinking
```

常用交互命令：

```text
/help
/quit
/new
/resume <conversation_id>
/history
/conversations
/tools
/skills
/tool <name> <request>
/skill <name> <request>
/<capability> <request>
/thinking on|off
/stats
/config
/clear
```

`/tool`、`/skill` 和 `/<capability>` 不会绕过 Agent 直接执行本地代码，而是
把本轮约束为指定能力，继续经过 Tool Schema、LangGraph、Artifact 虚拟化与
Telemetry。以 `//` 开头可以向模型发送普通的 `/` 文本。

`tui.show_thinking` 只控制展示；只有 `engine.disable_thinking = false` 时，
Qwen 等模型通常才会实际生成 Thinking。TUI 日志默认写入
`data/llama-agent-tui.log`，避免日志破坏全屏布局。

Qwen3/Qwen3.5 默认可能生成 `<think>` 推理块。配置中的
`engine.disable_thinking = true` 会发送 `/no_think` 软开关，同时引擎会把仍然
返回的推理块保存在消息元数据中而不作为 CLI 答案打印。单步骤任务直接复用
Executor 已验证的自然语言结论，不再额外调用一次 Finalizer 模型。

## R0 Benchmark

```bash
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r0_smoke.json \
  --output-root benchmark/results \
  --round R0
```

快速连通性检查使用 `r0_smoke.json`；正式第一轮使用
`benchmark/workloads/r0_full.json`，覆盖本地文件、结构化日志、只读 SQLite、
16/64 KiB 工具输出、多工具证据链、失败恢复、八轮会话和工具证据跨轮引用。
网络 Provider 已移除，所有 fixture 均由 worker 在样本目录内确定性生成。

只回归一个用例时使用 `--case`：

```bash
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r0_full.json \
  --output-root benchmark/results \
  --round R0-regression \
  --case w2-sqlite-investigation
```

`--case` 可以重复使用，以 Suite 文件中的顺序执行多个用例：

```bash
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r0_full.json \
  --output-root benchmark/results \
  --round R0-regression \
  --case w2-sqlite-investigation \
  --case w5-multi-tool-incident \
  --case w6-readonly-recovery
```

快速复测上一轮全部失败类别时，可以使用只执行一次、不含 warmup 的
`benchmark/workloads/r0_failed_regression.json`：

```bash
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r0_failed_regression.json \
  --output-root benchmark/results \
  --round R0-regression
```

每个 warmup/测量样本在新的 Python 进程中加载模型并执行，避免 llama.cpp
分配器、KV 状态或上一个任务污染后续样本。输出目录包含：

```text
manifest.json
task_results.jsonl
inference_events.jsonl
tool_events.jsonl
lifecycle_events.jsonl
system_memory.csv
kv_metrics.csv
summary.json
failures.jsonl
report.md
cases/
```

`R0` 会拒绝任何已启用的 `[memory]` 优化开关，以确保基线没有混入后续优化。

## R1 工具输出虚拟化

R1 在统一工具执行边界检查结果大小。超过
`memory.artifact_inline_max_bytes` 的原始结果写入任务隔离的 ArtifactStore，
模型上下文只保留确定性摘要、首尾预览、内容哈希和 `artifact://` 引用；需要
更多证据时可调用 `search_artifact` 或 `retrieve_artifact` 按需读取。

默认配置保持 `artifact_virtualization = false`，因此可继续执行 R0。R1 可通过
环境变量临时开启，无需修改基线配置文件：

```bash
AGENT_MEMORY_ARTIFACT_VIRTUALIZATION=true \
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r1_artifact_virtualization.json \
  --output-root benchmark/results \
  --round R1
```

只回归按需检索链路：

```bash
AGENT_MEMORY_ARTIFACT_VIRTUALIZATION=true \
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r1_artifact_virtualization.json \
  --output-root benchmark/results \
  --round R1-regression \
  --case r1-artifact-on-demand-search
```

阈值、首尾预览和摘要长度也可分别通过
`AGENT_MEMORY_ARTIFACT_INLINE_MAX_BYTES`、
`AGENT_MEMORY_ARTIFACT_PREVIEW_CHARS`、
`AGENT_MEMORY_ARTIFACT_SUMMARY_CHARS` 覆盖。报告会额外给出外置字节、模型
内联字节、减少量、压缩率和 Artifact 磁盘成本。

若要做严格的 R0/R1 同用例对照，应保持 R0 的 workload 不变，只打开 R1
开关。例如可对 `r0_full.json` 中的 16/64 KiB 两个样本使用重复的 `--case`
运行，并将 `--round` 设为 `R1-ablation`；这样任务、fixture、模型和评价规则均
与 R0 一致。上面的 R1 专用 Suite 主要用于验证虚拟化描述和按需检索能力。

```bash
# R0：全部内存优化关闭
AGENT_MEMORY_ARTIFACT_VIRTUALIZATION=false \
AGENT_MEMORY_LIFECYCLE_CONTEXT=false \
AGENT_MEMORY_KV_LIFECYCLE=false \
AGENT_MEMORY_BRANCH_MANAGEMENT=false \
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r0_full.json \
  --output-root benchmark/results \
  --round R0-artifact-ablation \
  --case w4-large-file-16k \
  --case w4-large-file-64k

# R1：只打开工具输出虚拟化，其余条件完全相同
AGENT_MEMORY_ARTIFACT_VIRTUALIZATION=true \
AGENT_MEMORY_LIFECYCLE_CONTEXT=false \
AGENT_MEMORY_KV_LIFECYCLE=false \
AGENT_MEMORY_BRANCH_MANAGEMENT=false \
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r0_full.json \
  --output-root benchmark/results \
  --round R1-artifact-ablation \
  --case w4-large-file-16k \
  --case w4-large-file-64k
```

## R2 生命周期感知上下文管理

R2 在 R1 之上将上下文分为 PINNED、HOT、WARM、COLD 和 DEAD。当前目标、
明确记忆和当前计划受到保护；最近执行记录保留原文；较旧执行结果使用确定性
摘要替换并将原文归档到任务隔离的 `ContextStore`。多轮 Conversation 开启 R2
后不再把全部历史拼入 `task_goal`，而是选择最近轮、明确要求记住的轮次和与
当前问题相关的历史。

运行 R1/R2 同用例消融：

```bash
# R1：只启用 Artifact 虚拟化
AGENT_MEMORY_ARTIFACT_VIRTUALIZATION=true \
AGENT_MEMORY_LIFECYCLE_CONTEXT=false \
AGENT_MEMORY_KV_LIFECYCLE=false \
AGENT_MEMORY_BRANCH_MANAGEMENT=false \
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r2_lifecycle_context.json \
  --output-root benchmark/results \
  --round R1-lifecycle-ablation

# R2：在相同工作负载上增加生命周期上下文管理
AGENT_MEMORY_ARTIFACT_VIRTUALIZATION=true \
AGENT_MEMORY_LIFECYCLE_CONTEXT=true \
AGENT_MEMORY_KV_LIFECYCLE=false \
AGENT_MEMORY_BRANCH_MANAGEMENT=false \
llama-agent --config agent_config.toml benchmark \
  --suite benchmark/workloads/r2_lifecycle_context.json \
  --output-root benchmark/results \
  --round R2-lifecycle-context
```

R2 报告增加上下文投影前后 token、状态压缩字节、Conversation 召回次数和
ContextStore 磁盘占用。`R2` round 会拒绝未同时开启 R1 与 R2 的配置。

## 一键运行 R0–R2 消融实验

`ablation` 命令会对同一个 Suite、同一组 `--case`、同一模型和同一推理参数
依次执行三轮。三轮内存策略由程序固定，不读取外部
`AGENT_MEMORY_*` 开关作为实验分组：

- R0：关闭 Artifact 虚拟化和生命周期上下文；
- R1：仅开启 Artifact 虚拟化；
- R2：开启 Artifact 虚拟化和生命周期上下文；
- 三轮均关闭 `kv_lifecycle` 和 `branch_management`，当前不执行 R3。

建议先执行能够覆盖 R1/R2 主要链路的五个代表用例：

```bash
llama-agent --config agent_config.toml ablation \
  --suite benchmark/workloads/r0_full.json \
  --output-root benchmark/results \
  --case w4-large-file-16k \
  --case w4-large-file-64k \
  --case w5-multi-tool-incident \
  --case w7-eight-turn-memory \
  --case w8-tool-to-conversation-memory
```

上面的命令沿用 Suite 中的 `warmup_runs=1`、`measured_runs=5`，用于正式统计。
只想先确认代码链路时可追加
`--warmup-runs 0 --measured-runs 1`，速度约为正式五次测量版本的五分之一；
快速结果不能替代最终统计报告。

完整 Suite 去掉全部 `--case` 即可：

```bash
llama-agent --config agent_config.toml ablation \
  --suite benchmark/workloads/r0_full.json \
  --output-root benchmark/results
```

也可以直接执行：

```bash
.llm-env/bin/python scripts/run_r0_r2_ablation.py \
  --config agent_config.toml \
  --suite benchmark/workloads/r0_full.json \
  --output-root benchmark/results
```

自动化目录下会保留三轮各自完整的原始结果，并额外生成
`comparison_data.json`、`report.md` 和 `charts/*.svg`。报告包含总体与分 Case
成功率、端到端延迟、输入 Token、逻辑 KV、RSS/GPU 显存、Checkpoint、
R1 外置收益、R2 上下文压缩与召回指标，以及失败检查项。若模型、推理参数、
用例集合或成功率不满足严格可比条件，报告会明确给出警告。

## 遥测口径

- `logical_tokens`、`position_span_tokens`：KV 逻辑占用，不是显存字节。
- `state_size_bytes`：序列化状态大小，不等同于 KV Cache 物理内存。
- RSS/USS/VMS：整个进程内存，包括模型权重与计算缓冲区。
- NVIDIA 指标：安装 `.[gpu]` 后通过 NVML 采集进程和设备显存。
- llama.cpp 性能计数器分别记录 prompt eval 与 decode eval；非流式 Agent
  调用不伪造 TTFT，原始事件中的 `ttft_ms` 为 `null`。

普通 CLI 默认关闭遥测；Benchmark worker 会强制开启并将数据写入对应样本目录。

## 测试

```bash
pytest -q
```

不需要真实 GGUF 的阶段一回归测试：

```bash
pytest -q src/tests/test_stage1_baseline.py
```
