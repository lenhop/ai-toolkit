import sys
import os
import time
from dotenv import load_dotenv, dotenv_values
from ai_toolkit.models import ModelManager
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool
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


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# 1. Structured output
## 1.1 ToolStrategy - custom output format using tools
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

print("=" * 60)
print("Example 1: ToolStrategy (Custom Output Format)")
print("=" * 60)

agent = create_agent(
    model=model,
    tools=[search],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

# IMPORTANT: agent.invoke() requires dict input with "messages" key
# WRONG: agent.invoke("text string")
# RIGHT: agent.invoke({"messages": [...]})
result = agent.invoke({
    "messages": [
        HumanMessage(content="Extract contact info from: John Doe, john@example.com, (555) 123-4567")
    ]
})

print(result["structured_response"])



## 1.2 ProviderStrategy - model provider's native structured output
from langchain.agents.structured_output import ProviderStrategy

print("=" * 60)
print("Example 2: ProviderStrategy (Native Structured Output)")
print("=" * 60)

# IMPORTANT: Qwen requires the word "json" in the prompt when using JSON mode
agent = create_agent(
    model=model,
    tools=[search],
    response_format=ProviderStrategy(ContactInfo),
    system_prompt="You are a helpful assistant that responds in JSON format."
)

# FIXED: Use correct dict format + add "json" to prompt for Qwen
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Extract contact info as JSON from: Jane Smith, jane@test.com, (555) 987-6543"}
    ]
})



# 2.Memory
# 2.1 Defining state via middleware
import os
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from typing import Any, Optional


# Define tools
@tool
def get_info(query: str) -> str:
    """Get information about a topic."""
    return f"Information about: {query}"

@tool
def analyze_data(data: str) -> str:
    """Analyze data and provide insights."""
    return f"Analysis of: {data}"

tools = [get_info, analyze_data]

# Define custom state schema
class CustomState(AgentState):
    """Custom state that extends AgentState with user preferences."""
    user_preferences: dict

# Define custom middleware
class CustomMiddleware(AgentMiddleware):
    """Middleware that can access and modify custom state."""
    
    state_schema = CustomState
    tools = []  # Can add middleware-specific tools here
    
    def before_model(
        self, 
        state: CustomState, 
        runtime
    ) -> Optional[dict[str, Any]]:
        """
        Called before the model is invoked.
        Can modify state or return updates.
        
        Args:
            state: Current agent state with user_preferences
            runtime: LangGraph runtime context
            
        Returns:
            Dict of state updates or None
        """
        # Access custom state
        prefs = state.get("user_preferences", {})
        style = prefs.get("style", "general")
        verbosity = prefs.get("verbosity", "normal")
        
        # Add system message based on preferences
        if style == "technical" and verbosity == "detailed":
            # Return state updates
            return {
                "messages": [
                    *state["messages"],
                    # Could add a system message here if needed
                ]
            }
        
        # Return None if no updates needed
        return None

# Create agent with custom middleware
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[CustomMiddleware()],
    system_prompt="You are a helpful assistant. Adapt your responses based on user preferences."
)


# Invoke agent with custom state
result = agent.invoke({
    "messages": [
        HumanMessage(content="I prefer technical explanations. Explain how neural networks work.")
    ],
    "user_preferences": {
        "style": "technical",
        "verbosity": "detailed"
    }
})

# Access results
print("Response:", result["messages"][-1].content)
print("\nUser preferences:", result.get("user_preferences"))


# 2.2 Defining state via state_schema
class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model=model,
    tools=tools,
    state_schema=CustomState,
    system_prompt="You are a helpful assistant. Adapt your responses based on user preferences."
)


# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})

# Access results
print("Response:", result["messages"][-1].content)
print("\nUser preferences:", result.get("user_preferences"))


# 3.streaming 
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values"):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
