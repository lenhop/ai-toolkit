# LangChain Agent Quick Reference

## Quick Start Template

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from ai_toolkit.models.model_manager import ModelManager

# 1. Create Model
model_manager = ModelManager()
model = model_manager.create_model(
    provider_name="deepseek",
    model_name="deepseek-chat",
    temperature=0.7,
    max_tokens=2000
)

# 2. Define Tools
@tool
def my_tool(input: str) -> str:
    """Tool description - model reads this to decide when to use it."""
    return f"Result: {input}"

tools = [my_tool]

# 3. Create Memory
checkpointer = MemorySaver()
config = {"configurable": {"thread_id": "conversation-1"}}

# 4. Create Agent
system_prompt = "You are a helpful assistant with access to tools."
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
    system_prompt=system_prompt
)

# 5. Run Agent
result = agent.invoke(
    {"messages": [HumanMessage(content="Your question here")]},
    config=config
)
print(result["messages"][-1].content)
```

## Component Cheat Sheet

### Models
```python
# Create model
model = model_manager.create_model(
    provider_name="deepseek",     # deepseek, qwen, glm
    model_name="deepseek-chat",
    temperature=0.7,              # 0.0-2.0 (lower = more deterministic)
    max_tokens=2000               # Max response length
)
```

### Messages
```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# System message - sets agent behavior
SystemMessage(content="You are a helpful assistant")

# Human message - user input
HumanMessage(content="What is 2 + 2?")

# AI message - model output (auto-generated)
AIMessage(content="2 + 2 equals 4")
```

### Tools
```python
from langchain_core.tools import tool

@tool
def tool_name(param: str) -> str:
    """
    Clear description of what the tool does.
    
    The model reads this docstring to decide when to use the tool.
    Be specific about when and how to use it.
    
    Args:
        param: Description of parameter
    
    Returns:
        Description of return value
    """
    # Tool implementation
    return f"Result: {param}"
```

**Tool Best Practices:**
- Clear, descriptive docstrings (model reads these!)
- Type hints are REQUIRED
- Return strings the model can interpret
- Handle errors gracefully

### Memory
```python
from langgraph.checkpoint.memory import MemorySaver

# In-memory storage (lost on restart)
checkpointer = MemorySaver()

# Configuration with thread_id
config = {"configurable": {"thread_id": "conversation-1"}}

# Same thread_id = shared memory
# Different thread_id = separate conversations
```

**Memory Types:**
- `MemorySaver` - In-memory (session-based)
- `SqliteSaver` - Persistent to disk
- `RedisSaver` - Distributed memory

### Agents
```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,                    # LLM for reasoning
    tools=tools,                    # List of available tools
    checkpointer=checkpointer,      # Memory for history
    system_prompt=system_prompt     # Agent instructions
)

# Invoke agent
result = agent.invoke(
    {"messages": [HumanMessage(content="Question")]},
    config=config  # Must include thread_id for memory
)

# Get response
response = result["messages"][-1].content
```

### Structured Output
```python
from pydantic import BaseModel, Field
from typing import Literal

# Define schema
class OutputSchema(BaseModel):
    field1: str = Field(description="Description")
    field2: int = Field(description="Description")
    field3: Literal["option1", "option2"] = Field(description="Description")

# Request structured output via prompt
query = """Your question here.
Respond in this exact JSON format:
{
    "field1": "value",
    "field2": 123,
    "field3": "option1"
}"""

result = model.invoke([
    SystemMessage(content="You provide structured JSON responses."),
    HumanMessage(content=query)
])

# Parse and validate
import json
parsed = json.loads(result.content)
validated = OutputSchema(**parsed)
```

## Common Patterns

### Pattern 1: Simple Question-Answer
```python
# No tools needed, just model
result = model.invoke([
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is Python?")
])
print(result.content)
```

### Pattern 2: Agent with Tools
```python
# Agent decides when to use tools
result = agent.invoke(
    {"messages": [HumanMessage(content="Calculate 123 * 456")]},
    config=config
)
print(result["messages"][-1].content)
```

### Pattern 3: Multi-turn Conversation
```python
# First message
result1 = agent.invoke(
    {"messages": [HumanMessage(content="What's the weather in Tokyo?")]},
    config=config  # thread_id: "conversation-1"
)

# Second message - agent remembers first
result2 = agent.invoke(
    {"messages": [HumanMessage(content="What if it's 5 degrees warmer?")]},
    config=config  # Same thread_id = shared memory
)
```

### Pattern 4: Separate Conversations
```python
# User 1
config_user1 = {"configurable": {"thread_id": "user-1"}}
result1 = agent.invoke({"messages": [HumanMessage(content="Hello")]}, config_user1)

# User 2 - separate memory
config_user2 = {"configurable": {"thread_id": "user-2"}}
result2 = agent.invoke({"messages": [HumanMessage(content="Hello")]}, config_user2)
```

## When to Use What

### Use Model Directly When:
- Simple question-answer
- No tools needed
- No conversation history required
- Quick, one-off queries

### Use Agent When:
- Need to use tools
- Multi-step reasoning required
- Conversation history needed
- Complex workflows

## ReAct Pattern Flow

```
User Question
     ↓
Agent Receives Question
     ↓
Agent Reasons: "Do I need a tool?"
     ↓
   Yes → Call Tool → Observe Result → Reason Again
     ↓
   No → Formulate Answer
     ↓
Return Response
```

## Common Issues

### Issue: Agent not using tools
**Solution:**
- Check tool docstrings are clear
- Mention tools in system prompt
- Verify tools are passed to `create_agent()`

### Issue: Memory not working
**Solution:**
- Use same `thread_id` across invocations
- Pass `config` to `agent.invoke()`
- Provide checkpointer to `create_agent()`

### Issue: Import errors
**Solution:**
```python
# Correct imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
```

## Parameter Reference

### Temperature
- `0.0` - Deterministic, consistent responses
- `0.3-0.5` - Focused, reliable
- `0.7-0.9` - Balanced creativity
- `1.0-2.0` - Very creative, unpredictable

### Max Tokens
- `500` - Short responses
- `1000-2000` - Standard responses
- `4000+` - Long, detailed responses

## Official Documentation Links

- **Models**: https://docs.langchain.com/oss/python/langchain/models
- **Messages**: https://docs.langchain.com/oss/python/langchain/messages
- **Tools**: https://docs.langchain.com/oss/python/langchain/tools
- **Memory**: https://docs.langchain.com/oss/python/langchain/short-term-memory
- **Agents**: https://docs.langchain.com/oss/python/langchain/agents
- **Structured Output**: https://docs.langchain.com/oss/python/langchain/structured-output

## Example Files

- `1.model_access_methods_guide.py` - 5 methods to access AI models
- `2.simple_agent_basics.py` - Complete beginner's guide
- `3.advanced_agent_patterns.py` - Advanced patterns
- `README_simple_agent_basics.md` - Detailed documentation

---

**Quick Tip:** Start with `2.simple_agent_basics.py` to learn the fundamentals, then explore advanced patterns in `3.advanced_agent_patterns.py`.
