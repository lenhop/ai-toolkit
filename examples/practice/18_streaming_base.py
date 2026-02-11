
import sys
import os
import time
from dotenv import load_dotenv, dotenv_values
from ai_toolkit.models import ModelManager
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage

load_dotenv()
# Create model manager
manager = ModelManager()

# Create a model (auto-loads API key from environment - recommended)
model = manager.create_model(
    api_key=os.environ.get('QWEN_API_KEY'),
    provider="qwen",           # Provider: deepseek, qwen, or glm
    model="qwen-turbo",         # Model variant (optional)
    temperature=0.7,               # Randomness: 0.0-2.0
    max_tokens=2000                # Max response length
)


from langchain.agents import create_agent


# 1.Agent progress
def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
)

for chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")


# 2.LLM tokens
def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)
for token, metadata in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")


# 3.Custom updates
from langchain.agents import create_agent
from langgraph.config import get_stream_writer  


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()  
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"
):
    print(chunk)    


# 4.Stream multiple modes
from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
)

for stream_mode, chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"]
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}")
    print("\n")
    

# 5.Common patterns

## 5.1 Streaming tool calls
from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


agent = create_agent(model, tools=[get_weather])


def _render_message_chunk(token: AIMessageChunk) -> None:
    # print("token:", token)
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)
    # N.B. all content is available through token.content_blocks


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],  
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)  
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):  # `source` captures node name
                _render_completed_message(update["messages"][-1])


## 5.2 Streaming with human-in-the-loop
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI  # 补充模型初始化依赖
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt


# 定义天气工具
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# 初始化模型（替换为你自己的模型配置，这里用OpenAI示例）
model = ChatOpenAI(
    model="gpt-4o",  # 可替换为其他模型
    api_key="your-openai-api-key",  # 替换为你的API Key
    streaming=True  # 开启流式输出
)

# 初始化检查点（保存会话状态，支持中断恢复）
checkpointer = InMemorySaver()

# 创建带人机协同中间件的智能体
agent = create_agent(
    model,
    tools=[get_weather],
    middleware=[
        # 配置：调用get_weather工具时触发人工中断
        HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}),
    ],
    checkpointer=checkpointer,
)


# 自定义渲染函数：处理LLM生成的消息片段
def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|", flush=True)  # flush=True确保实时打印
    if token.tool_call_chunks:
        print(f"\n工具调用片段: {token.tool_call_chunks}")


# 自定义渲染函数：处理完成的消息（工具调用/工具返回）
def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"\n完整工具调用指令: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"\n工具返回结果: {message.content_blocks}")


# 自定义渲染函数：处理人工中断提示，并等待人工审批
def _handle_interrupt(interrupt: Interrupt) -> bool:
    """
    处理人工中断，返回是否同意继续执行
    :param interrupt: 中断对象
    :return: True=同意，False=拒绝
    """
    interrupts = interrupt.value
    print("\n===== 触发人工审批 =====")
    for request in interrupts["action_requests"]:
        print(f"审批提示: {request['description']}")
    
    # 等待人工输入审批结果
    while True:
        human_input = input("是否同意调用get_weather工具？(输入y同意/输入n拒绝): ").strip().lower()
        if human_input in ["y", "n"]:
            break
        print("输入无效，请输入y或n！")
    
    if human_input == "y":
        print("✅ 人工审批通过，继续执行工具调用...")
        return True
    else:
        print("❌ 人工审批拒绝，终止流程！")
        return False


# 主执行逻辑
if __name__ == "__main__":
    # 用户输入：查询波士顿和旧金山天气
    input_message = {
        "role": "user",
        "content": "Can you look up the weather in Boston and San Francisco?"
    }

    # 配置会话ID：保证中断后状态一致
    config = {"configurable": {"thread_id": "weather_query_123"}}
    interrupts = []
    approval_granted = True  # 标记是否通过人工审批

    # 流式执行智能体
    for stream_mode, data in agent.stream(
        {"messages": [input_message]},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        # 若已拒绝审批，直接终止循环
        if not approval_granted:
            break

        # 处理messages模式：实时打印LLM生成片段
        if stream_mode == "messages":
            token, metadata = data
            if isinstance(token, AIMessageChunk):
                _render_message_chunk(token)

        # 处理updates模式：完整节点（模型/工具/中断）
        if stream_mode == "updates":
            for source, update in data.items():
                # 处理模型/工具节点的完成消息
                if source in ("model", "tools"):
                    _render_completed_message(update["messages"][-1])
                
                # 处理人工中断节点：核心修改点
                if source == "__interrupt__":
                    interrupts.extend(update)
                    # 调用审批函数，获取审批结果
                    approval_granted = _handle_interrupt(update[0])
                    # 若拒绝，直接终止循环
                    if not approval_granted:
                        break

    # 流程结束提示
    print("\n===== 执行流程结束 =====")



# 5.3 Streaming from sub-agents
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


weather_agent = create_agent(
    model=model,
    tools=[get_weather],
    name="weather_agent",  
)


def call_weather_agent(query: str) -> str:
    """Query the weather agent."""
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].text


from copy import deepcopy

supervisor_model = deepcoy(model)

agent = create_agent(
    model=supervisor_model,
    tools=[call_weather_agent],
    name="supervisor",  
)


def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
current_agent = None
for _, stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    subgraphs=True,  
):
    if stream_mode == "messages":
        token, metadata = data
        if agent_name := metadata.get("lc_agent_name"):  
            if agent_name != current_agent:  
                print(f"🤖 {agent_name}: ")  
                current_agent = agent_name  
        if isinstance(token, AIMessage):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])


# 5.4 Disable streaming
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    streaming=False
)

