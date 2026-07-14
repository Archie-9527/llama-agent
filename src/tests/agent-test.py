# llama-agent integration test — agent with tool calling
# python agent-wrapper/tests/test_agent.py

import os
import sys
sys.path.insert(0, "agent-wrapper")

from langchain_core.tools import tool
from agent_core.agent import create_llama_agent

current_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = "Qwen3.5-4B-UD-Q8_K_XL.gguf"
model_path = os.path.join(current_dir, "models", model_filename)


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of English text.

    Args:
        text: The text to count words in.

    Returns:
        The number of words.
    """
    return len(text.split())


agent = create_llama_agent(
    model_path=model_path,
    tools=[word_count],
    system_prompt="You are a helpful assistant. Use tools when appropriate, reply concisely in English.",
    n_ctx=2048,
    n_gpu_layers=-1,
    temperature=0.0,
    verbose=False,
)

print("=== Agent invoke ===")
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Count the words in: hello world this is a test"}
    ]
})

# Print all messages to see the tool-calling flow
for msg in result["messages"]:
    role = msg.type
    content = msg.content if hasattr(msg, "content") else str(msg)
    print(f"[{role}] {content}")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  -> tool_call: {tc['name']}({tc['args']})")
