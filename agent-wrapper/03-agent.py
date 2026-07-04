# 03_agent.py —— 跟通 AgentExecutor 循环
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def word_count(text: str) -> int:
    """统计一段英文文本的单词数"""
    return len(text.split())

model = ChatOpenAI(
    model="local-qwen",
    base_url="http://localhost:8000/v1",   # 指向本地 llama-cpp server
    api_key="not-needed",                   # 本地服务随便填
    temperature=0,
)

agent = create_agent(model=model, tools=[word_count])

result = agent.invoke({"messages": [{"role": "user", "content": "Count words in: hello world test"}]})
print(result)