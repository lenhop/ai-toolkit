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


# 1.Custom tool description
# calculator is the name for agent invoking
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))


# 2.Advanced schema definition
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result


# 3.Accessing state
from langchain.tools import tool, ToolRuntime

# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")


# 4.Updating state
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

# Update the conversation history by removing all messages
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})


# 5.Context
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

agent = create_agent(
    model=model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="""You are a financial assistant. 
    You MUST use the `get_account_info` tool to answer all questions about user account information (including balance, account type, name).
    Do NOT say you don't have access - always call the tool first.""",  # 强制工具调用，禁用默认回复
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)

# 打印最终回复内容（便于查看结果）
print(result["messages"][-1].content)


# 6.Memory
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI

# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info by user_id."""  # 优化工具描述，更精准
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info (name/age/email etc.) with user_id as key."""  # 优化工具描述
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()

# 核心：补充精准的system prompt，引导智能体正确调用工具
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store,
    system_prompt="""You are a user information management assistant. Follow these rules strictly:
1. Tool usage rules:
   - When user asks to "save" user info: Call save_user_info tool, extract user_id (e.g., abc123) and user_info dict (name/age/email etc.) from user's message.
   - When user asks to "get/find/look up" user info: Call get_user_info tool with the specified user_id.
2. Parsing rules:
   - Extract user_id accurately (e.g., "userid: abc123" → user_id = "abc123").
   - Extract user attributes (name/age/email) into a dict (e.g., {"name":"Foo", "age":25, "email":"foo@langchain.dev"}).
3. Response rules:
   - After saving: Confirm "User info for {user_id} saved successfully".
   - After querying: Format user info as clear list (e.g., "- Name: Foo\n- Age: 25\n- Email: foo@langchain.dev").
   - Never refuse to call tools; always use the provided tools to handle user info requests.
""",
)

# First session: save user info
save_result = agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})
print("保存用户信息回复：", save_result["messages"][-1].content)

# Second session: get user info
get_result = agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
print("\n查询用户信息回复:", get_result["messages"][-1].content)


# 7.Stream writer
# This is useful for providing real-time feedback to users about what a tool is doing
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
