# llama-agent: 架构与集成设计报告

## 1. 项目架构

### 1.1 项目目录结构

```
llama-agent/                              # 工作区根目录
|-- models/                                # GGUF 模型文件
|   +-- Qwen3.5-4B-UD-Q8_K_XL.gguf       # 约 5.6 GB
|
|-- llm-env/                               # Python 虚拟环境 (3.14, editable 安装)
|
|-- llama.cpp/                             # 上游 C++ 推理引擎 (git submodule)
|-- llama-cpp-python/                      # Python ctypes 绑定层 (editable 安装)
|   +-- llama_cpp/
|       +-- llama.py                       # Llama 类: 公开 API
|       +-- llama_cpp.py                   # 对 libllama.dylib 的 ctypes 绑定
|       +-- llama_chat_format.py           # 聊天模板系统 + 格式注册表
|       +-- llama_tokenizer.py             # 分词器封装
|       +-- _internals.py                  # LlamaModel/LlamaContext/LlamaBatch/LlamaSampler
|       +-- server/                        # OpenAI 兼容的 FastAPI 服务
|       +-- tests/
|
|-- langchain/                             # langchain 单体仓库 (git clone)
|   +-- libs/
|       +-- core/langchain_core/           # 基础抽象层 (BaseChatModel, tools, runnables)
|       +-- langchain_v1/langchain/        # 新版 agent 工厂函数: create_agent()
|       +-- langchain/langchain_classic/   # 已废弃的 AgentExecutor
|       +-- partners/                      # 官方集成 (openai, ollama, anthropic 等)
|
|-- test-agent/                            # 现有概念验证脚本
|   +-- test-llama.py                      # 直接使用 llama-cpp-python
|   +-- 03-agent.py                        # 通过 ChatOpenAI -> 本地 server 的 agent
|
|-- llama_agent/                           # [新增] 统一 agent 项目
    +-- __init__.py
    +-- chat_model.py                      # LlamaCppChatModel: BaseChatModel 子类
    +-- agent.py                           # agent 工厂函数 + 默认工具
    +-- config.py                          # 配置项 (模型路径, 上下文长度, 采样默认值)
    +-- tools/                             # 内置工具集
    |   +-- __init__.py
    |   +-- filesystem.py                  # 文件读写/列表
    |   +-- shell.py                       # shell 命令执行
    +-- tests/
    |   +-- test_chat_model.py             # 单元测试: 聊天模型集成
    |   +-- test_agent.py                  # 集成测试: agent 循环与工具调用
    +-- README.md
    +-- requirements.txt
```

### 1.2 关键设计决策

**llama-cpp-python 和 langchain-core/langchain 均采用 editable 安装。**

项目已配置好此方式: `llama-cpp-python` 通过 scikit-build-core 的 editable 模式安装, langchain 各组件通过 pip editable 安装 (对应的 `_editable_impl_*` `.pth` 文件可在 `llm-env/lib/python3.14/site-packages/` 中看到)。这意味着:
- 在 agent 开发过程中可以直接调试和修改 `llama-cpp-python` 源码
- C++ 代码变更后无需手动重新构建 (editable hook 会自动触发 cmake 重编译)

**单一 venv, Python 3.14。**

所有依赖 (llama-cpp-python, langchain-core, langchain, langgraph) 共存于 `llm-env/` 中，无需多套环境。

**`llama_agent/` 包是集成层，不包含上游代码。**

它不复制任何上游项目的内容，仅从 `llama_cpp` 和 `langchain_core` 导入所需模块，仅提供胶水代码。

---

## 2. 集成分析

### 2.1 LangChain 的模型集成契约

LangChain v1 的 `create_agent()` 接受 `model` 参数，类型为 `str | BaseChatModel`。当传入 `BaseChatModel` 实例时，它会跳过 `init_chat_model()` 直接使用该实例。这意味着我们只需实现一个合法的 `BaseChatModel` 子类，agent 工厂函数就能直接接受，无需任何注册流程。

`BaseChatModel` (位于 `langchain_core/language_models/chat_models.py`) 要求实现:

| 方法 | 是否必须 | 说明 |
|---|---|---|
| `_generate(messages, stop, run_manager, **kwargs) -> ChatResult` | **是** | 核心方法: 接收 `List[BaseMessage]`, 返回 `ChatResult` |
| `_llm_type: str` (属性) | **是** | 标识字符串, 例如 `"llama-cpp"` |
| `_stream(messages, stop, run_manager, **kwargs) -> Iterator[ChatGenerationChunk]` | 否 | token 级别的流式输出。省略时自动用 `_generate` 包装 |
| `_agenerate(...)` / `_astream(...)` | 否 | 异步变体。省略时用 `run_in_executor` 包装同步版本 |
| `bind_tools(tools, **kwargs) -> Runnable` | 否 | 工具调用支持，对 agent 函数调用至关重要 |
| `_resolve_model_profile() -> ModelProfile \| None` | 否 | 从 profile 数据库自动加载模型能力信息 |

### 2.2 数据流: Messages -> ChatML -> Tokens -> Response

核心挑战在于 LangChain 的消息类型和 llama.cpp 期望的 prompt 格式之间的转换。以下是完整数据流:

```
LangChain Agent
    |
    | messages: [SystemMessage, HumanMessage, ...]
    v
LlamaCppChatModel._generate(messages, ...)
    |
    | (1) 将 messages 转换为 ChatML prompt 字符串
    |     SystemMessage(content="你是一个有用的助手")
    |     HumanMessage(content="统计 hello world 的单词数")
    |         |
    |         v
    |     "<|im_start|>system\n你是一个有用的助手<|im_end|>\n
    |      <|im_start|>user\n统计 hello world 的单词数<|im_end|>\n
    |      <|im_start|>assistant\n"
    |
    | (2) 调用 llama_cpp.Llama.create_completion(prompt=chatml_string, ...)
    |     Llama 类内部流程:
    |       - 通过 llama_tokenize (BPE 算法) 将 prompt 分词为 token ID 序列
    |       - 循环执行 llama_decode (eval -> sample -> append -> eval)
    |       - 通过 llama_token_to_piece 将每个采样出的 token 反分词为文本
    |
    | (3) 收到 completion 文本: "hello world test 共 4 个单词。"
    |
    | (4) 封装为 ChatResult:
    |     ChatResult(generations=[
    |         ChatGeneration(message=AIMessage(content="hello world test 共 4 个单词。"))
    |     ])
    v
LangChain Agent 继续循环 (执行工具, 将结果反馈给模型)
```

### 2.3 三种集成策略对比

#### 策略 A: 直接包装 Llama 实例 (推荐)

继承 `BaseChatModel`, 内部持有 `llama_cpp.Llama` 实例。`_generate()` 手动将 LangChain messages 转换为 ChatML 格式并调用 `llama.create_completion()`。

**优点:**
- 无需单独的服务进程，一切在进程内运行
- 完全控制采样参数、停止词、上下文管理
- 延迟最低 (无 HTTP 往返)
- 可直接访问 `tokenize()`/`detokenize()` 进行 token 计数
- 可利用 llama-cpp-python 已有的 chat format 系统

**缺点:**
- 需要自行编写 LangChain message -> ChatML 的转换逻辑 (也可委托给 llama-cpp-python 已有的 `llama_chat_format` 模块)
- 不直接支持异步 (可通过 `run_in_executor` 包装)

实现工作量: 约 200 行 Python。

#### 策略 B: OpenAI 兼容 Server

运行 `python -m llama_cpp.server`，将 `ChatOpenAI` 指向 `http://localhost:8000/v1`。此方案已在 `test-agent/03-agent.py` 中得到验证。

**优点:**
- 零 wrapper 代码
- 异步开箱即用 (ChatOpenAI 原生支持异步)
- Server 可在多个客户端/进程间共享
- 完整的 OpenAI API 兼容 (tools, streaming 等)

**缺点:**
- 需要管理独立的服务进程
- 每次推理调用有 HTTP 开销
- 对内部状态控制较弱 (跨请求的 KV cache 复用等)

#### 策略 C: 混合方案

以 `BaseChatModel` wrapper (策略 A) 为主要方式，将 server 方案 (策略 B) 作为多客户端或异步密集型场景的备选方案进行文档化说明。

### 2.4 工具调用分析

这是 agent 能力中最关键的环节。LangChain v1 的 `create_agent()` 通过 LangGraph 的 `ToolNode` 来实现工具调用。模型必须能够输出结构化内容，指明要调用哪个工具以及参数。

**llama-cpp-python 的 `create_chat_completion()` 如何处理 tools:**

`llama_chat_format.py` 中的 chat format 系统通过以下 handler 支持工具调用:
- `chatml-function-calling` (第 4130 行) — 基于 ChatML 的函数调用专用 handler
- `functionary` (第 1460 行) — 针对 Functionary 模型
- 通用的 JSON schema 模式 (通过 GBNF grammar 约束)

函数调用 handler 的工作流程:
1. 将 LangChain/OpenAI 的 tool 定义转换为 JSON schema
2. 将函数定义添加到 system prompt 中
3. 使用 GBNF grammar 约束模型输出为合法 JSON
4. 将 JSON 输出解析回工具调用结构

**对于我们的 wrapper，有两种工具调用实现方式:**

方式一: **委托给 `llama.create_chat_completion()`** — 将完整的 messages + tools 传递给 llama-cpp-python 已有的 chat completion handler。handler 负责格式化、grammar 约束和响应解析。我们的 `_generate()` 只需将解析出的 tool calls 转换回 LangChain 的 `AIMessage.tool_calls` 格式。

方式二: **手动 prompt 工程** — 自行将工具描述格式化到 system prompt 中，然后解析模型文本输出中的工具调用模式。更简单但可靠性较低。

强烈推荐方式一。关键设计洞察: **我们的 `_generate()` 内部可以调用 `llama.create_chat_completion(tools=..., ...)` 而非 `llama.create_completion()`**，从而直接复用 llama-cpp-python 已有的工具调用基础设施。

### 2.5 上下文窗口管理

Qwen3.5-4B 模型在 Llama 构造函数中配置的上下文窗口为 `n_ctx=4096` 个 token。在 agent 循环中，对话历史会随着每次工具调用和回复不断增长，因此需要:

1. **Token 计数**: 在发送给模型前使用 `llm.tokenize()` 统计 token 数量。
2. **截断策略**: 当对话超过 `n_ctx - max_tokens` 时，截断较早的消息 (但保留 system prompt 和最近的用户消息)。
3. **KV cache 复用**: `llama-cpp-python` 的 `generate()` 方法已自动检测 prompt 前缀匹配并复用缓存的 KV 状态，因此重复的 system prompt 几乎零开销。

这可以作为 `_generate()` 中的预处理步骤来实现。

### 2.6 流式输出

对于实时工具调用反馈，流式输出非常重要。我们的 `BaseChatModel` 应实现 `_stream()`:

```python
def _stream(
    self,
    messages: list[BaseMessage],
    stop: list[str] | None = None,
    run_manager: CallbackManagerForLLMRun | None = None,
    **kwargs: Any,
) -> Iterator[ChatGenerationChunk]:
    # 将 LangChain messages 转换为 ChatML 格式的 prompt 字符串
    prompt = self._messages_to_chatml(messages)
    # 调用 llama-cpp-python 的流式 completion
    for chunk in self._llm.create_completion(
        prompt=prompt,
        stream=True,
        **self._sampling_params,
    ):
        text = chunk["choices"][0]["text"]
        yield ChatGenerationChunk(message=AIMessageChunk(content=text))
```

这与 llama-cpp-python 已有的流式 `create_completion()` 完美对应。

### 2.7 总结: 需要构建的组件

| 组件 | 预估行数 | 说明 |
|---|---|---|
| `LlamaCppChatModel(BaseChatModel)` | ~200 | 核心 wrapper: messages->ChatML, completion->AIMessage, 流式输出, 工具调用 |
| `create_llama_agent()` | ~50 | 工厂函数: model + tools -> CompiledStateGraph |
| `config.py` | ~30 | 模型路径, n_ctx, 采样参数默认值 |
| 内置工具集 | ~100 | 文件操作, shell 执行 |
| 测试 | ~150 | 单元测试 + 集成测试 |

Wrapper 非常薄，因为两个库已经各司其职:
- llama-cpp-python 负责模型加载、推理、采样和聊天格式化
- LangChain (通过 LangGraph) 负责 agent 循环、工具执行和状态管理

我们的集成层只需要在两个类型系统之间做翻译 — 具体来说，就是将 LangChain 的 `BaseMessage` 类型与 llama-cpp-python 对话系统所期望的 ChatML prompt 格式互相转换。
