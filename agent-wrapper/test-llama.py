# 01_raw_llama.py —— 跟通 create_completion 这条核心路径
from llama_cpp import Llama

llm = Llama(
    model_path="/Users/heart/Code/CPP-Project/llama-agent/models/Qwen3.5-4B-UD-Q8_K_XL.gguf",
    n_gpu_layers=-1,    # 全部层进 Metal
    n_ctx=2048,
    verbose=True,       # 看 Metal 加载日志
)

# 这是你之前要重点读源码的那个方法
output = llm.create_completion(
    "用一句话解释什么是大语言模型",
    max_tokens=512,
    # stop=["\n"],        # 留意 stop 词的处理
    temperature=0.1,
)
print(output["choices"][0]["text"])
