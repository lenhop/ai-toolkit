# Simple AI Agent Basics - Official LangChain Patterns

## Overview

This example (`2.simple_agent_basics.py`) demonstrates fundamental AI agent concepts following **official LangChain documentation patterns**. It's designed as a beginner-friendly introduction to building AI agents with LangChain.

## What You'll Learn

This example covers 7 essential components:

1. **Models** - Creating and configuring chat models
2. **Messages** - Understanding SystemMessage, HumanMessage, AIMessage
3. **Tools** - Defining tools that agents can use
4. **Memory** - Implementing conversation history with checkpointers
5. **Agents** - Creating ReAct agents that combine reasoning and actions
6. **Structured Output** - Getting formatted, parseable responses
7. **Running Examples** - Practical demonstrations of agent capabilities

## Prerequisites

- Python 3.11+
- AI Toolkit installed (`pip install -e .`)
- DeepSeek API key configured in `.env`
- Required packages: `langchain`, `langchain-core`, `langgraph`, `pydantic`

## Quick Start

```bash
# Run the example
python examples/2.simple_agent_basics.py
```

## Component Breakdown

### 1. Models - The Reasoning Engine

Models are the brain of your agent. They drive decision-making, tool selection, and response generation.

**Key Concepts:**
- Models provide the reasoning capability
- They decide which tools to call and how to interpret results
- LangChain provides standard interfaces for multiple providers

**Code Example:**
```python
from ai_toolkit.models.model_manager import ModelManager

model_manager = ModelManager()
model = model_manager.create_model(
    provider_name="deepseek",
    model_name="deepseek-chat",
    temperature=0.7,      # Controls randomness (0.0-2.0)
    max_tokens=2000       # Maximum response length
)
```

**Key Parameters:**
- `temperature`: Controls randomness (0.0 = deterministic, 2.0 = very random)
- `max_tokens`: Maximum response length
- `model_name`: Specific model variant to use

**Official Docs:** https://docs.langchain.com/oss/python/langchain/models

---

### 2. Messages - Conversation Building Blocks

Messages are the fundamental unit of context for models. They represent input/output and carry conversation state.

**Three Message Types:**

1. **SystemMessage** - Initial instructions that prime the model's behavior
   ```python
   SystemMessage(content="You are a helpful AI assistant with access to tools.")
   ```

2. **HumanMessage** - User input and interactions
   ```python
   HumanMessage(content="What is 2 + 2?")
   ```

3. **AIMessage** - Model output (automatically created by model)
   ```python
   AIMessage(content="2 + 2 equals 4.")
   ```

**Conversation Structure:**
```python
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi! How can I help?"),
    HumanMessage(content="What is 5 + 5?")
]
```

**Official Docs:** https://docs.langchain.com/oss/python/langchain/messages

---

### 3. Tools - Extending Agent Capabilities

Tools extend what agents can do beyond text generation. They allow agents to fetch data, execute code, call APIs, and more.

**Key Concepts:**
- Tools are callable functions with well-defined inputs/outputs
- The model decides when to invoke a tool based on context
- Docstrings are CRITICAL - the model reads them to understand when to use the tool

**Tool Definition:**
```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.
    
    Use this tool when you need to perform mathematical calculations.
    Supports basic arithmetic operations: +, -, *, /, **, %.
    
    Args:
        expression: A mathematical expression to evaluate
    
    Returns:
        The result of the calculation as a string
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

**Best Practices:**
- Write clear, concise docstrings
- Use descriptive function names
- Define explicit parameter types (type hints are REQUIRED)
- Handle errors gracefully
- Return strings that the model can interpret

**Official Docs:** https://docs.langchain.com/oss/python/langchain/tools

---

### 4. Memory - Conversation History

Memory allows agents to remember previous conversations and maintain context across multiple interactions.

**Key Concepts:**
- Checkpointers save conversation state
- `thread_id` isolates different conversations
- Same thread_id = shared memory (agent remembers)
- Different thread_id = separate conversations

**Memory Types:**
- **MemorySaver**: In-memory storage (lost on restart)
- **SqliteSaver**: Persistent storage to disk
- **RedisSaver**: Distributed memory for scaling

**Code Example:**
```python
from langgraph.checkpoint.memory import MemorySaver

# Create memory checkpointer
checkpointer = MemorySaver()

# Configuration with thread_id
thread_id = "conversation-1"
config = {"configurable": {"thread_id": thread_id}}
```

**Use Cases:**
- Multi-turn conversations
- Context retention across interactions
- Multi-user applications (different thread_id per user)
- Conversation isolation

**Official Docs:** https://docs.langchain.com/oss/python/langchain/short-term-memory

---

### 5. Agents - Combining Everything

Agents combine models with tools to solve complex tasks. They use the **ReAct pattern** (Reason + Act).

**ReAct Pattern Flow:**
1. Agent receives user question
2. Agent reasons: "Do I need a tool?"
3. If yes: Agent calls tool, observes result
4. Agent formulates answer based on results
5. Repeat until answer is complete

**Creating an Agent:**
```python
from langchain.agents import create_agent

system_prompt = """You are a helpful AI assistant with access to tools.

Your capabilities:
- Perform mathematical calculations using the calculator tool
- Provide weather information using the get_weather tool

Guidelines:
- Be friendly and helpful
- Use tools when needed for accurate information
- Explain your reasoning clearly"""

agent = create_agent(
    model=model,                    # The LLM for reasoning
    tools=tools,                    # List of available tools
    checkpointer=checkpointer,      # Memory for conversation history
    system_prompt=system_prompt     # Instructions for agent behavior
)
```

**Agent Components:**
- `model`: The LLM for reasoning
- `tools`: Available tools the agent can use
- `checkpointer`: Memory for conversation history
- `system_prompt`: Instructions for agent behavior

**Official Docs:** https://docs.langchain.com/oss/python/langchain/agents

---

### 6. Running the Agent

**Basic Invocation:**
```python
result = agent.invoke(
    {"messages": [HumanMessage(content="What is 123 * 456?")]},
    config=config  # Include thread_id for memory
)

# Extract final response
final_response = result["messages"][-1].content
print(final_response)
```

**Example Interactions:**

1. **Mathematical Calculation:**
   ```
   User: "What is 123 multiplied by 456?"
   Agent: Uses calculator tool → Returns "56,088"
   ```

2. **Weather Query:**
   ```
   User: "What's the weather like in Tokyo?"
   Agent: Uses get_weather tool → Returns "Clear, 15°C, Pleasant"
   ```

3. **Memory Test:**
   ```
   User: "If the temperature in Tokyo increases by 5 degrees, what would it be?"
   Agent: Remembers Tokyo's temperature (15°C) → Uses calculator → Returns "20°C"
   ```

---

### 7. Structured Output - Formatted Responses

Structured output ensures predictable, parseable responses using Pydantic schemas.

**Key Concepts:**
- Define schemas using Pydantic models
- Ensures type-safe, validated responses
- Useful for API responses, database inserts, data validation

**Defining a Schema:**
```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInfo(BaseModel):
    """Structured weather information."""
    city: str = Field(description="The city name")
    temperature: int = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the information"
    )
```

**Requesting Structured Output:**
```python
query = """What's the weather in Beijing? 
Respond in this exact JSON format:
{
    "city": "city name",
    "temperature": temperature_in_celsius,
    "condition": "weather condition",
    "confidence": "high/medium/low"
}"""

result = model.invoke([
    SystemMessage(content="You provide structured JSON responses."),
    HumanMessage(content=query)
])

# Parse and validate
import json
parsed_data = json.loads(result.content)
weather_info = WeatherInfo(**parsed_data)
```

**Use Cases:**
- API integrations
- Database operations
- Data extraction
- Consistent formatting

**Official Docs:** https://docs.langchain.com/oss/python/langchain/structured-output

---

## Complete Example Output

When you run the example, you'll see:

```
================================================================================
SIMPLE AI AGENT BASICS - OFFICIAL LANGCHAIN PATTERNS
================================================================================

1. MODELS - Creating Chat Model
✅ Model created: deepseek-chat

2. MESSAGES - Message Types
✅ SystemMessage, HumanMessage, AIMessage created

3. TOOLS - Defining Tools
✅ Tools defined: calculator, get_weather

4. MEMORY - Conversation History
✅ Memory checkpointer created: MemorySaver

5. AGENTS - Creating ReAct Agent
✅ Agent created successfully!

6. RUNNING AGENT - Examples
Example 1: Mathematical Calculation
Example 2: Weather Information
Example 3: Memory Test

7. STRUCTURED OUTPUT - Formatted Responses
✅ Structured Output (Validated)
```

## Key Takeaways

1. **Models** provide the reasoning capability
2. **Messages** structure conversations (System, Human, AI)
3. **Tools** extend agent capabilities beyond text generation
4. **Memory** enables context retention across interactions
5. **Agents** combine models + tools + memory using ReAct pattern
6. **Structured Output** ensures predictable, parseable responses
7. **thread_id** isolates different conversations

## Next Steps

After mastering this example, try:

1. **Add Custom Tools** - Create your own tools for specific tasks
2. **Experiment with Prompts** - Try different system prompts
3. **Test Memory** - Use different thread_ids for conversation isolation
4. **Define Schemas** - Create your own Pydantic schemas
5. **Advanced Patterns** - Explore `3.advanced_agent_patterns.py`

## Official Documentation

- **Models**: https://docs.langchain.com/oss/python/langchain/models
- **Messages**: https://docs.langchain.com/oss/python/langchain/messages
- **Tools**: https://docs.langchain.com/oss/python/langchain/tools
- **Memory**: https://docs.langchain.com/oss/python/langchain/short-term-memory
- **Agents**: https://docs.langchain.com/oss/python/langchain/agents
- **Structured Output**: https://docs.langchain.com/oss/python/langchain/structured-output

## Troubleshooting

**Issue: Agent not using tools**
- Check tool docstrings - they must be clear and descriptive
- Verify system prompt mentions tool capabilities
- Ensure tools are passed to `create_agent()`

**Issue: Memory not working**
- Verify same `thread_id` is used across invocations
- Check `config` is passed to `agent.invoke()`
- Ensure checkpointer is provided to `create_agent()`

**Issue: Structured output parsing fails**
- Check JSON format in model response
- Handle markdown code blocks (```json)
- Validate against Pydantic schema

## Related Examples

- `1.model_access_methods_guide.py` - 5 methods to access AI models
- `3.advanced_agent_patterns.py` - Advanced patterns (dynamic models, prompts)
- `MESSAGE_TYPES_GUIDE.md` - Deep dive into message types
- `AGENT_VS_MODEL.md` - Understanding when to use agents vs models

---

**Happy Learning! 🎓**
