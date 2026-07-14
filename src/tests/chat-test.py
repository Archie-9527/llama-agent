from llama_cpp import Llama

def main():
    model_path = "/Users/heart/Code/Py-Project/llama-agent/src/tests/models/Qwen3.5-4B-UD-Q8_K_XL.gguf" 

    # 初始化模型
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,      # 上下文长度
        n_threads=8,     # CPU线程数，可按机器调整
        n_gpu_layers=0,  # 纯CPU；若GPU可改成 >0
        verbose=False
    )

    print("模型加载完成。输入 /exit 或 /quit 退出。")

    messages = [
        {"role": "system", "content": "你是一个简洁、友好的中文助手。"}
    ]

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in {"/exit", "/quit"}:
            print("已退出。")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            # Chat Completion
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stream=False
            )
            assistant_text = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            assistant_text = f"[生成失败] {e}"

        print(f"\n助手: {assistant_text}")
        messages.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    main()
