import sys
import os
import time
from dotenv import load_dotenv, dotenv_values
from ai_toolkit.models import ModelManager

load_dotenv()
# Create model manager
manager = ModelManager()

# Create a model (auto-loads API key from environment - recommended)
basic_model = manager.create_model(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    provider="deepseek",           # Provider: deepseek, qwen, or glm
    model="deepseek-chat",         # Model variant (optional)
    temperature=0.7,               # Randomness: 0.0-2.0
    max_tokens=2000                # Max response length
)

advanced_model = manager.create_model(
    api_key=os.environ.get('QWEN_API_KEY'),
    provider="qwen",           # Provider: deepseek, qwen, or glm
    model="qwen-turbo",         # Model variant (optional)
    temperature=0.7,               # Randomness: 0.0-2.0
    max_tokens=2000                # Max response length
)


# 1.tools
from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


tools = [search, get_weather]


# 2.dynamic models

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)

# 3.Tool error handling
"""
MIDDLEWARE OBJECTS SUMMARY:

wrap_model_call: Model selection middleware
- request: ModelRequest (state, model, messages)
- handler: Callable invoking model → ModelResponse

wrap_tool_call: Tool execution middleware  
- request: ToolCallRequest (tool_call dict, tool, state, runtime)
- handler: Callable executing tool → ToolMessage/Command

IMPORTS:
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langgraph.prebuilt.tool_node import ToolCallRequest, ModelRequest
from langchain_core.messages import ToolMessage

USE: Model routing, error handling, retry, caching, logging
"""


from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[handle_tool_errors]
)


# 4.System prompt
## 4.1
agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[handle_tool_errors]
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

## 4.2 Tool error handling
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
    model=basic_model,
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type": "text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    )
)

result = literary_agent.invoke(
    {"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]}
)


## 4.3 Dynamic system prompt
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)