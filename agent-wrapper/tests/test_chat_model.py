# llama-agent integration test — basic chat model inference
# python agent-wrapper/tests/test_chat_model.py

import sys
sys.path.insert(0, "agent-wrapper")

from llama_agent.chat_model import LlamaCppChatModel
from langchain_core.messages import HumanMessage, SystemMessage

MODEL_PATH = "models/Qwen3.5-4B-UD-Q8_K_XL.gguf"

chat = LlamaCppChatModel(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,
    temperature=0.7,
    max_tokens=128,
    verbose=False,
)

messages = [
    SystemMessage(content="You are a helpful assistant. Reply concisely in English."),
    HumanMessage(content="Explain what attention mechanism is in one sentence."),
]

print("=== Non-streaming ===")
result = chat.invoke(messages)
print(f"Content: {result.content}")
print()

print("=== Streaming ===")
chat.streaming = True
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
print()
