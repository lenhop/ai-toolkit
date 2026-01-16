#!/usr/bin/env python3
"""
Simple AI Agent Basics - Following Official LangChain Patterns

This example demonstrates fundamental AI agent concepts following official LangChain documentation:
https://docs.langchain.com/oss/python/langchain/

Components Demonstrated:
1. Models - Creating and configuring chat models
2. Messages - SystemMessage, HumanMessage, AIMessage
3. Tools - Defining and using tools with agents
4. Memory - Short-term conversation history with checkpointers
5. Agents - ReAct agent with tools and memory
6. Structured Output - Getting formatted responses
7. Middleware - Custom tool behavior (logging, error handling)

Official Documentation References:
- Models: https://docs.langchain.com/oss/python/langchain/models
- Messages: https://docs.langchain.com/oss/python/langchain/messages
- Tools: https://docs.langchain.com/oss/python/langchain/tools
- Memory: https://docs.langchain.com/oss/python/langchain/short-term-memory
- Agents: https://docs.langchain.com/oss/python/langchain/agents
- Structured Output: https://docs.langchain.com/oss/python/langchain/structured-output
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# Import ai_toolkit
from ai_toolkit.models.model_manager import ModelManager


print("\n" + "=" * 80)
print("SIMPLE AI AGENT BASICS - OFFICIAL LANGCHAIN PATTERNS")
print("=" * 80)
print("\nFollowing official LangChain documentation patterns")
print("This example demonstrates: Models, Messages, Tools, Memory, Agents, Structured Output")
print()


# =============================================================================
# 1. MODELS - Creating and Configuring Chat Models
# =============================================================================
# Official Docs: https://docs.langchain.com/oss/python/langchain/models
#
# Key Concepts:
# - Models are the reasoning engine of agents
# - They drive decision-making: which tools to call, how to interpret results
# - LangChain provides standard interfaces for multiple providers
# - Models can be used standalone or with agents
#
# Key Parameters:
# - temperature: Controls randomness (0.0 = deterministic, 2.0 = very random)
# - max_tokens: Maximum response length
# - model_name: Specific model variant to use
# =============================================================================

print("\n" + "=" * 80)
print("1. MODELS - Creating Chat Model")
print("=" * 80)

# Initialize model using ai_toolkit's ModelManager
model_manager = ModelManager()

# Create a chat model with specific parameters
# temperature=0.7 provides balanced creativity and consistency
# max_tokens=2000 limits response length
model = model_manager.create_model(
    provider_name="deepseek",      # Provider: deepseek, qwen, or glm
    model_name="deepseek-chat",    # Model variant
    temperature=0.7,               # Randomness: 0.0-2.0
    max_tokens=2000                # Max response length
)

print("✅ Model created: deepseek-chat")
print(f"   Temperature: 0.7 (balanced creativity)")
print(f"   Max tokens: 2000")
print()


# =============================================================================
# 2. MESSAGES - Understanding Message Types
# =============================================================================
# Official Docs: https://docs.langchain.com/oss/python/langchain/messages
#
# Key Concepts:
# - Messages are the fundamental unit of context for models
# - They represent input/output and carry conversation state
# - Three main types: SystemMessage, HumanMessage, AIMessage
#
# Message Types:
# - SystemMessage: Initial instructions that prime the model's behavior
# - HumanMessage: User input and interactions
# - AIMessage: Model output (automatically created by model)
#
# Message Attributes:
# - content: The message text or multimodal content
# - role: Who sent the message (system, user, assistant)
# - metadata: Additional information about the message
# =============================================================================

print("=" * 80)
print("2. MESSAGES - Message Types")
print("=" * 80)

# SystemMessage: Sets agent behavior and context
# Use at the start of conversation to define personality and capabilities
system_message = SystemMessage(
    content="You are a helpful AI assistant with access to tools."
)
print("✅ SystemMessage created")
print(f"   Purpose: Define agent behavior and personality")
print(f"   Content: '{system_message.content[:50]}...'")
print()

# HumanMessage: Represents user input
# Use for every user question or request
human_message = HumanMessage(
    content="What is 2 + 2?"
)
print("✅ HumanMessage created")
print(f"   Purpose: Represent user input")
print(f"   Content: '{human_message.content}'")
print()

# AIMessage: Represents model output
# Automatically created by the model, but can be manually created for history
ai_message = AIMessage(
    content="2 + 2 equals 4."
)
print("✅ AIMessage created")
print(f"   Purpose: Represent model output")
print(f"   Content: '{ai_message.content}'")
print(f"   Note: Usually auto-generated by model")
print()

# Message list example - how conversations are structured
messages_example = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi! How can I help?"),
    HumanMessage(content="What is 5 + 5?")
]
print("📝 Message List Example (Conversation Structure):")
for i, msg in enumerate(messages_example, 1):
    msg_type = type(msg).__name__
    print(f"   {i}. {msg_type}: '{msg.content}'")
print()


# =============================================================================
# 3. TOOLS - Defining Tools for Agents
# =============================================================================
# Official Docs: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Tools extend what agents can do (fetch data, execute code, etc.)
# - Tools are callable functions with well-defined inputs/outputs
# - The model decides when to invoke a tool based on context
#
# Tool Definition:
# - Use @tool decorator to create tools
# - Type hints are REQUIRED (define input schema)
# - Docstring is CRITICAL (model reads it to understand when to use tool)
# - Return strings that the model can interpret
#
# Best Practices:
# - Write clear, concise docstrings
# - Use descriptive function names
# - Define explicit parameter types
# - Handle errors gracefully
# =============================================================================

print("=" * 80)
print("3. TOOLS - Defining Tools")
print("=" * 80)

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.
    
    Use this tool when you need to perform mathematical calculations.
    Supports basic arithmetic operations: +, -, *, /, **, %.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result of the calculation as a string
    
    Examples:
        calculator("15 + 27") -> "Result: 42"
        calculator("10 * 5") -> "Result: 50"
    """
    try:
        # Safe evaluation - empty __builtins__ prevents dangerous operations
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a city.
    
    Use this tool when the user asks about weather conditions.
    Provides temperature in Celsius and general conditions.
    
    Args:
        city: Name of the city to get weather for
    
    Returns:
        Weather information including temperature and conditions
    
    Examples:
        get_weather("Beijing") -> "Weather in Beijing: Sunny, 22°C, Light breeze"
        get_weather("Tokyo") -> "Weather in Tokyo: Clear, 15°C, Pleasant"
    """
    # Simulated weather data (in production, would call real weather API)
    weather_data = {
        "beijing": "Sunny, 22°C, Light breeze",
        "shanghai": "Cloudy, 18°C, Moderate wind",
        "tokyo": "Clear, 15°C, Pleasant",
        "new york": "Cold, 5°C, Snow expected",
        "london": "Foggy, 10°C, Drizzle",
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    else:
        return f"Weather data not available for {city}"


# List of tools to provide to the agent
tools = [calculator, get_weather]

print("✅ Tools defined:")
for tool_func in tools:
    print(f"   - {tool_func.name}")
    print(f"     Description: {tool_func.description[:60]}...")
    print(f"     Args: {list(tool_func.args.keys())}")
print()


# =============================================================================
# 4. MEMORY - Short-term Conversation History
# =============================================================================
# Official Docs: https://docs.langchain.com/oss/python/langchain/short-term-memory
#
# Key Concepts:
# - Memory allows agents to remember previous conversations
# - Checkpointers save conversation state
# - thread_id isolates different conversations
#
# Memory Types:
# - MemorySaver: In-memory storage (lost on restart)
# - SqliteSaver: Persistent storage to disk
# - RedisSaver: Distributed memory for scaling
#
# Thread ID:
# - Same thread_id = shared memory (agent remembers)
# - Different thread_id = separate conversations
# - Use for multi-user applications or conversation isolation
# =============================================================================

print("=" * 80)
print("4. MEMORY - Conversation History")
print("=" * 80)

# Create memory checkpointer
# MemorySaver stores conversation history in RAM
checkpointer = MemorySaver()

print("✅ Memory checkpointer created: MemorySaver")
print(f"   Type: In-memory storage")
print(f"   Persistence: Session-based (lost on restart)")
print(f"   Alternative: SqliteSaver (disk), RedisSaver (distributed)")
print()

# Configuration with thread_id
# thread_id is used to isolate different conversations
thread_id = "conversation-1"
config = {"configurable": {"thread_id": thread_id}}

print(f"✅ Configuration created:")
print(f"   thread_id: '{thread_id}'")
print(f"   Purpose: Isolate this conversation from others")
print()


# =============================================================================
# 5. AGENTS - Creating ReAct Agent
# =============================================================================
# Official Docs: https://docs.langchain.com/oss/python/langchain/agents
#
# Key Concepts:
# - Agents combine models with tools to solve tasks
# - ReAct pattern: Reason (think) + Act (use tools)
# - Agents iterate until they reach a solution
#
# ReAct Pattern:
# 1. Agent receives user question
# 2. Agent reasons: "Do I need a tool?"
# 3. If yes: Agent calls tool, observes result
# 4. Agent formulates answer based on results
# 5. Repeat until answer is complete
#
# Agent Components:
# - model: The LLM for reasoning
# - tools: Available tools the agent can use
# - checkpointer: Memory for conversation history
# - system_prompt: Instructions for agent behavior
# =============================================================================

print("=" * 80)
print("5. AGENTS - Creating ReAct Agent")
print("=" * 80)

# System prompt defines agent behavior
system_prompt = """You are a helpful AI assistant with access to tools.

Your capabilities:
- Perform mathematical calculations using the calculator tool
- Provide weather information using the get_weather tool

Guidelines:
- Be friendly and helpful
- Use tools when needed for accurate information
- Explain your reasoning clearly
- If unsure, say so honestly"""

print("✅ System prompt defined")
print(f"   Length: {len(system_prompt)} characters")
print()

# Create agent using create_agent
# This is the official LangChain pattern for creating agents
agent = create_agent(
    model=model,                    # The LLM to use for reasoning
    tools=tools,                    # List of available tools
    checkpointer=checkpointer,      # Memory for conversation history
    system_prompt=system_prompt     # Instructions for agent behavior
)

print("✅ Agent created successfully!")
print(f"   Type: ReAct Agent")
print(f"   Model: deepseek-chat")
print(f"   Tools: {len(tools)} tools available")
print(f"   Memory: Enabled with checkpointer")
print()


# =============================================================================
# 6. RUNNING THE AGENT - Examples
# =============================================================================

print("=" * 80)
print("6. RUNNING AGENT - Examples")
print("=" * 80)
print()

# Example 1: Simple calculation
print("-" * 80)
print("Example 1: Mathematical Calculation")
print("-" * 80)

query1 = "What is 123 multiplied by 456?"
print(f"👤 User: {query1}")

# Invoke agent with message
# The agent will:
# 1. Receive the question
# 2. Reason: "This needs calculator tool"
# 3. Call calculator("123 * 456")
# 4. Get result: 56088
# 5. Respond with answer
result1 = agent.invoke(
    {"messages": [HumanMessage(content=query1)]},
    config=config
)

# Extract final response
final_response1 = result1["messages"][-1].content
print(f"🤖 Agent: {final_response1}")
print()

# Example 2: Weather query
print("-" * 80)
print("Example 2: Weather Information")
print("-" * 80)

query2 = "What's the weather like in Tokyo?"
print(f"👤 User: {query2}")

# Agent will use get_weather tool
result2 = agent.invoke(
    {"messages": [HumanMessage(content=query2)]},
    config=config  # Same thread_id = shared memory
)

final_response2 = result2["messages"][-1].content
print(f"🤖 Agent: {final_response2}")
print()

# Example 3: Memory test
print("-" * 80)
print("Example 3: Memory Test (Multi-step Reasoning)")
print("-" * 80)

query3 = "If the temperature in Tokyo increases by 5 degrees, what would it be?"
print(f"👤 User: {query3}")
print(f"   (Tests if agent remembers Tokyo's temperature from Example 2)")

# Agent will:
# 1. Remember Tokyo's temperature from previous conversation (15°C)
# 2. Use calculator to add 5
# 3. Respond with result (20°C)
result3 = agent.invoke(
    {"messages": [HumanMessage(content=query3)]},
    config=config  # Same thread_id = agent remembers
)

final_response3 = result3["messages"][-1].content
print(f"🤖 Agent: {final_response3}")
print()

# Show conversation summary
print("-" * 80)
print("Conversation Summary")
print("-" * 80)
print(f"   Total messages: {len(result3['messages'])}")
print(f"   Thread ID: {thread_id}")
print(f"   Tools used: calculator, get_weather")
print()


# =============================================================================
# 7. STRUCTURED OUTPUT - Getting Formatted Responses
# =============================================================================
# Official Docs: https://docs.langchain.com/oss/python/langchain/structured-output
#
# Key Concepts:
# - Structured output ensures predictable, parseable responses
# - Define schemas using Pydantic models
# - Use ToolStrategy for automatic validation and error handling
# - Useful for API responses, database inserts, data extraction
#
# Methods:
# - ToolStrategy: Official pattern for structured output with agents
# - Pydantic models: Type-safe schemas with validation
# - handle_errors: Automatic error handling (default: True)
#
# Use Cases:
# - API integrations
# - Database operations
# - Data extraction
# - Consistent formatting
# =============================================================================

print("=" * 80)
print("7. STRUCTURED OUTPUT - Formatted Responses")
print("=" * 80)

# Import structured output components
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.structured_output import StructuredOutputValidationError

# Define Pydantic schema for structured output
class WeatherInfo(BaseModel):
    """Structured weather information."""
    city: str = Field(description="The city name")
    temperature: int = Field(description="Temperature in Celsius", ge=-50, le=60)
    condition: str = Field(description="Weather condition")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the information"
    )

print("✅ Pydantic schema defined: WeatherInfo")
print(f"   Fields: city, temperature, condition, confidence")
print(f"   Validation: temperature must be between -50 and 60")
print()

# Example 1: Basic Structured Output with ToolStrategy
print("-" * 80)
print("Example 1: Basic Structured Output")
print("-" * 80)

# Create agent with structured output
# ToolStrategy automatically handles validation and error recovery
structured_agent = create_agent(
    model=model,
    tools=[],  # No tools needed for structured output
    response_format=ToolStrategy(WeatherInfo),  # Default: handle_errors=True
    system_prompt="You are a helpful assistant that provides weather information in structured format. Do not make up any field or value."
)

query_structured = "What's the weather in Beijing?"
print(f"👤 User: {query_structured}")

# Invoke agent with structured output
result_structured = structured_agent.invoke(
    {"messages": [HumanMessage(content=query_structured)]},
    config=config
)

# Extract structured output from ToolMessage
print(f"\n🤖 Agent Response:")
structured_found = False
for msg in result_structured['messages']:
    # Check if message is a ToolMessage (contains structured output)
    if type(msg).__name__ == "ToolMessage":
        print(f"   Raw content: {msg.content}")
        structured_found = True
        
        # Parse the structured output
        try:
            import json
            # Try to parse as JSON first
            weather_data = json.loads(msg.content)
            weather_info = WeatherInfo(**weather_data)
            
            print(f"\n✅ Structured Output (Validated):")
            print(f"   City: {weather_info.city}")
            print(f"   Temperature: {weather_info.temperature}°C")
            print(f"   Condition: {weather_info.condition}")
            print(f"   Confidence: {weather_info.confidence}")
        except json.JSONDecodeError:
            # If not JSON, try to parse the string representation
            try:
                # Extract values from string like "city='Beijing' temperature=15..."
                import re
                content = msg.content
                city_match = re.search(r"city='([^']+)'", content)
                temp_match = re.search(r"temperature=(\d+)", content)
                cond_match = re.search(r"condition='([^']+)'", content)
                conf_match = re.search(r"confidence='([^']+)'", content)
                
                if all([city_match, temp_match, cond_match, conf_match]):
                    weather_info = WeatherInfo(
                        city=city_match.group(1),
                        temperature=int(temp_match.group(1)),
                        condition=cond_match.group(1),
                        confidence=conf_match.group(1)
                    )
                    
                    print(f"\n✅ Structured Output (Validated):")
                    print(f"   City: {weather_info.city}")
                    print(f"   Temperature: {weather_info.temperature}°C")
                    print(f"   Condition: {weather_info.condition}")
                    print(f"   Confidence: {weather_info.confidence}")
                else:
                    print(f"   ⚠️  Could not parse structured output")
            except Exception as e:
                print(f"   ⚠️  Parsing error: {e}")
        except Exception as e:
            print(f"   ⚠️  Validation error: {e}")

if not structured_found:
    print("   ℹ️  No ToolMessage found - structured output may be in final response")

print()

# Example 2: Custom Error Handling
print("-" * 80)
print("Example 2: Custom Error Handling")
print("-" * 80)

# Define custom error handler
def custom_error_handler(error: Exception) -> str:
    """
    Custom error handler for structured output validation.
    
    Args:
        error: The validation error
    
    Returns:
        Error message to send back to the model
    """
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Please ensure all fields are correct and try again."
    else:
        return f"Error: {str(error)}. Please provide valid weather information."

print("✅ Custom error handler defined")
print(f"   Handles: StructuredOutputValidationError and general errors")
print()

# Create agent with custom error handling
structured_agent_custom = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=WeatherInfo,
        handle_errors=custom_error_handler  # Custom error handler
    ),
    system_prompt="You are a helpful assistant that provides weather information. Ensure temperature is realistic (-50 to 60°C)."
)

query_custom = "What's the weather in Tokyo?"
print(f"👤 User: {query_custom}")

# Invoke agent with custom error handling
result_custom = structured_agent_custom.invoke(
    {"messages": [HumanMessage(content=query_custom)]},
    config=config
)

# Extract structured output
print(f"\n🤖 Agent Response:")
structured_found = False
for msg in result_custom['messages']:
    if type(msg).__name__ == "ToolMessage":
        structured_found = True
        try:
            import json
            # Try JSON first
            weather_data = json.loads(msg.content)
            weather_info = WeatherInfo(**weather_data)
            
            print(f"✅ Structured Output (Validated):")
            print(f"   City: {weather_info.city}")
            print(f"   Temperature: {weather_info.temperature}°C")
            print(f"   Condition: {weather_info.condition}")
            print(f"   Confidence: {weather_info.confidence}")
        except json.JSONDecodeError:
            # Parse string representation
            try:
                import re
                content = msg.content
                city_match = re.search(r"city='([^']+)'", content)
                temp_match = re.search(r"temperature=(\d+)", content)
                cond_match = re.search(r"condition='([^']+)'", content)
                conf_match = re.search(r"confidence='([^']+)'", content)
                
                if all([city_match, temp_match, cond_match, conf_match]):
                    weather_info = WeatherInfo(
                        city=city_match.group(1),
                        temperature=int(temp_match.group(1)),
                        condition=cond_match.group(1),
                        confidence=conf_match.group(1)
                    )
                    
                    print(f"✅ Structured Output (Validated):")
                    print(f"   City: {weather_info.city}")
                    print(f"   Temperature: {weather_info.temperature}°C")
                    print(f"   Condition: {weather_info.condition}")
                    print(f"   Confidence: {weather_info.confidence}")
                else:
                    print(f"⚠️  Could not parse: {msg.content}")
            except Exception as e:
                print(f"⚠️  Error: {e}")
        except Exception as e:
            print(f"⚠️  Error: {e}")

if not structured_found:
    print("ℹ️  No ToolMessage found")

print()

# Example 3: Multiple Schema Types (Union)
print("-" * 80)
print("Example 3: Multiple Schema Types")
print("-" * 80)

# Define additional schema
class CityInfo(BaseModel):
    """City information."""
    name: str = Field(description="City name")
    country: str = Field(description="Country name")
    population: int = Field(description="Population", ge=0)
    famous_for: str = Field(description="What the city is famous for")

print("✅ Additional schema defined: CityInfo")
print(f"   Fields: name, country, population, famous_for")
print()

# Note: Union types require Python 3.10+ or typing.Union
from typing import Union

# Create agent that can return either WeatherInfo or CityInfo
multi_schema_agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=Union[WeatherInfo, CityInfo],
        handle_errors=True  # Automatic error handling
    ),
    system_prompt="You are a helpful assistant. Provide weather information OR city information based on the user's question."
)

query_multi = "Tell me about Beijing as a city"
print(f"👤 User: {query_multi}")

# Invoke agent with multiple schema types
result_multi = multi_schema_agent.invoke(
    {"messages": [HumanMessage(content=query_multi)]},
    config=config
)

# Extract structured output (could be either type)
print(f"\n🤖 Agent Response:")
structured_found = False
for msg in result_multi['messages']:
    if type(msg).__name__ == "ToolMessage":
        structured_found = True
        try:
            import json
            # Try JSON first
            data = json.loads(msg.content)
            
            # Try to determine which schema was used
            if "temperature" in data:
                info = WeatherInfo(**data)
                print(f"✅ WeatherInfo returned:")
                print(f"   City: {info.city}")
                print(f"   Temperature: {info.temperature}°C")
                print(f"   Condition: {info.condition}")
            elif "population" in data:
                info = CityInfo(**data)
                print(f"✅ CityInfo returned:")
                print(f"   Name: {info.name}")
                print(f"   Country: {info.country}")
                print(f"   Population: {info.population:,}")
                print(f"   Famous for: {info.famous_for}")
        except json.JSONDecodeError:
            # Parse string representation
            try:
                import re
                content = msg.content
                
                # Check if it's WeatherInfo
                if "temperature=" in content:
                    city_match = re.search(r"city='([^']+)'", content)
                    temp_match = re.search(r"temperature=(\d+)", content)
                    cond_match = re.search(r"condition='([^']+)'", content)
                    conf_match = re.search(r"confidence='([^']+)'", content)
                    
                    if all([city_match, temp_match, cond_match, conf_match]):
                        info = WeatherInfo(
                            city=city_match.group(1),
                            temperature=int(temp_match.group(1)),
                            condition=cond_match.group(1),
                            confidence=conf_match.group(1)
                        )
                        print(f"✅ WeatherInfo returned:")
                        print(f"   City: {info.city}")
                        print(f"   Temperature: {info.temperature}°C")
                        print(f"   Condition: {info.condition}")
                
                # Check if it's CityInfo
                elif "population=" in content:
                    name_match = re.search(r"name='([^']+)'", content)
                    country_match = re.search(r"country='([^']+)'", content)
                    pop_match = re.search(r"population=(\d+)", content)
                    famous_match = re.search(r"famous_for='([^']+)'", content)
                    
                    if all([name_match, country_match, pop_match, famous_match]):
                        info = CityInfo(
                            name=name_match.group(1),
                            country=country_match.group(1),
                            population=int(pop_match.group(1)),
                            famous_for=famous_match.group(1)
                        )
                        print(f"✅ CityInfo returned:")
                        print(f"   Name: {info.name}")
                        print(f"   Country: {info.country}")
                        print(f"   Population: {info.population:,}")
                        print(f"   Famous for: {info.famous_for}")
                else:
                    print(f"⚠️  Could not parse: {msg.content}")
            except Exception as e:
                print(f"⚠️  Error: {e}")
        except Exception as e:
            print(f"⚠️  Error: {e}")

if not structured_found:
    print("ℹ️  No ToolMessage found")

print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("EXAMPLE COMPLETE!")
print("=" * 80)

print("\n🎓 What You Learned:")
print("   1. Models - Creating and configuring chat models")
print("   2. Messages - SystemMessage, HumanMessage, AIMessage")
print("   3. Tools - Defining tools with @tool decorator")
print("   4. Memory - Using checkpointers for conversation history")
print("   5. Agents - Creating ReAct agents with create_agent")
print("   6. Structured Output - Getting formatted responses with Pydantic")

print("\n📚 Official Documentation:")
print("   - Models: https://docs.langchain.com/oss/python/langchain/models")
print("   - Messages: https://docs.langchain.com/oss/python/langchain/messages")
print("   - Tools: https://docs.langchain.com/oss/python/langchain/tools")
print("   - Memory: https://docs.langchain.com/oss/python/langchain/short-term-memory")
print("   - Agents: https://docs.langchain.com/oss/python/langchain/agents")
print("   - Structured Output: https://docs.langchain.com/oss/python/langchain/structured-output")

print("\n💡 Next Steps:")
print("   - Try adding your own custom tools")
print("   - Experiment with different system prompts")
print("   - Test with different thread_ids for conversation isolation")
print("   - Define your own Pydantic schemas for structured output")
print("   - Explore advanced agent patterns in 3.advanced_agent_patterns.py")

print()
