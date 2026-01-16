#!/usr/bin/env python3
"""
Tool Creation Guide - Creating AI Tools for LangChain Agents

This example demonstrates various methods for creating tools for AI agents
following official LangChain documentation:
https://docs.langchain.com/oss/python/langchain/tools

Tool Creation Methods Covered:
1. @tool Decorator - Simple function-based tools
2. StructuredTool - Tools with complex input schemas
3. BaseTool Class - Custom tool classes with full control
4. Tool from Function - Dynamic tool creation
5. Tool with Error Handling - Robust tools with validation
6. Tool with Callbacks - Tools with lifecycle hooks
7. Async Tools - Asynchronous tool execution
8. Tool with Artifacts - Tools that return structured data

Official Documentation:
https://docs.langchain.com/oss/python/langchain/tools
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Type, Any, Dict, List
from pydantic import BaseModel, Field
import asyncio

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_core.tools import tool, StructuredTool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Import ai_toolkit
from ai_toolkit.models.model_manager import ModelManager


print("\n" + "=" * 80)
print("TOOL CREATION GUIDE - CREATING AI TOOLS")
print("=" * 80)
print("\nDemonstrating various methods for creating LangChain tools")
print("Official Docs: https://docs.langchain.com/oss/python/langchain/tools")
print()


# =============================================================================
# 1. @tool DECORATOR - Simple Function-Based Tools
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Simplest way to create tools
# - Use @tool decorator on any function
# - Type hints define input schema
# - Docstring is CRITICAL (model reads it)
# - Return strings for best results
#
# Best Practices:
# - Write clear, descriptive docstrings
# - Use type hints for all parameters
# - Handle errors gracefully
# - Return informative strings
# =============================================================================

print("=" * 80)
print("1. @tool DECORATOR - Simple Function-Based Tools")
print("=" * 80)
print()

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.
    
    Use this tool when you need to perform mathematical calculations.
    Supports basic arithmetic: +, -, *, /, **, %.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2")
    
    Returns:
        The calculation result as a string
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_database(query: str, limit: int = 5) -> str:
    """
    Search the database for information.
    
    Use this tool to find information in the database.
    
    Args:
        query: Search query string
        limit: Maximum number of results (default: 5)
    
    Returns:
        Search results as formatted string
    """
    # Simulated database search
    results = [
        f"Result {i+1}: Information about '{query}'"
        for i in range(min(limit, 3))
    ]
    return "\n".join(results)


print("✅ Created tools with @tool decorator:")
print(f"   - calculator: {calculator.name}")
print(f"     Description: {calculator.description[:50]}...")
print(f"     Args: {list(calculator.args.keys())}")
print()
print(f"   - search_database: {search_database.name}")
print(f"     Description: {search_database.description[:50]}...")
print(f"     Args: {list(search_database.args.keys())}")
print()


# =============================================================================
# 2. STRUCTUREDTOOL - Tools with Complex Input Schemas
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Use when you need complex input validation
# - Define Pydantic models for input schema
# - More control over input structure
# - Better validation and error messages
#
# Use Cases:
# - Multiple required parameters
# - Complex data types
# - Input validation rules
# - Nested data structures
# =============================================================================

print("=" * 80)
print("2. STRUCTUREDTOOL - Complex Input Schemas")
print("=" * 80)
print()

# Define input schema with Pydantic
class EmailInput(BaseModel):
    """Input schema for sending emails."""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body content")
    cc: Optional[List[str]] = Field(default=None, description="CC recipients")


def send_email_func(to: str, subject: str, body: str, cc: Optional[List[str]] = None) -> str:
    """Send an email."""
    cc_str = f", CC: {', '.join(cc)}" if cc else ""
    return f"Email sent to {to}{cc_str}\nSubject: {subject}\nBody: {body[:50]}..."


# Create StructuredTool
send_email = StructuredTool.from_function(
    func=send_email_func,
    name="send_email",
    description="Send an email to specified recipients",
    args_schema=EmailInput,
    return_direct=False
)

print("✅ Created StructuredTool:")
print(f"   Name: {send_email.name}")
print(f"   Description: {send_email.description}")
print(f"   Input schema: EmailInput (to, subject, body, cc)")
print()

# Example usage
result = send_email.invoke({
    "to": "user@example.com",
    "subject": "Test Email",
    "body": "This is a test email",
    "cc": ["manager@example.com"]
})
print(f"   Test result: {result}")
print()


# =============================================================================
# 3. BASETOOL CLASS - Custom Tool Classes
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Maximum control over tool behavior
# - Inherit from BaseTool
# - Implement _run() method
# - Define args_schema with Pydantic
# - Can maintain state
#
# Use Cases:
# - Complex tool logic
# - Stateful tools
# - Custom initialization
# - Advanced error handling
# =============================================================================

print("=" * 80)
print("3. BASETOOL CLASS - Custom Tool Classes")
print("=" * 80)
print()

class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    city: str = Field(description="City name to get weather for")
    units: str = Field(
        default="celsius",
        description="Temperature units: celsius or fahrenheit"
    )


class WeatherTool(BaseTool):
    """Custom weather tool with state."""
    
    name: str = "get_weather"
    description: str = "Get current weather information for a city"
    args_schema: Type[BaseModel] = WeatherInput
    
    # Tool can have state
    api_key: str = "demo_key"
    cache: Dict[str, str] = {}
    
    def _run(
        self,
        city: str,
        units: str = "celsius",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the tool."""
        # Check cache
        cache_key = f"{city}_{units}"
        if cache_key in self.cache:
            return f"[Cached] {self.cache[cache_key]}"
        
        # Simulated weather data
        weather_data = {
            "beijing": {"celsius": "22°C, Sunny", "fahrenheit": "72°F, Sunny"},
            "tokyo": {"celsius": "15°C, Clear", "fahrenheit": "59°F, Clear"},
            "new york": {"celsius": "5°C, Cold", "fahrenheit": "41°F, Cold"},
        }
        
        city_lower = city.lower()
        if city_lower in weather_data:
            result = f"Weather in {city}: {weather_data[city_lower][units]}"
            self.cache[cache_key] = result
            return result
        else:
            return f"Weather data not available for {city}"


# Create instance
weather_tool = WeatherTool()

print("✅ Created custom BaseTool:")
print(f"   Name: {weather_tool.name}")
print(f"   Description: {weather_tool.description}")
print(f"   Has state: api_key, cache")
print()

# Test the tool
result1 = weather_tool.invoke({"city": "Tokyo", "units": "celsius"})
print(f"   First call: {result1}")

result2 = weather_tool.invoke({"city": "Tokyo", "units": "celsius"})
print(f"   Second call: {result2}")
print()


# =============================================================================
# 4. TOOL FROM FUNCTION - Dynamic Tool Creation
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Create tools dynamically at runtime
# - Useful for generating tools from config
# - Can create multiple similar tools
# - Flexible tool generation
# =============================================================================

print("=" * 80)
print("4. TOOL FROM FUNCTION - Dynamic Creation")
print("=" * 80)
print()

def create_converter_tool(from_unit: str, to_unit: str, conversion_factor: float):
    """Factory function to create unit converter tools."""
    
    # Define input schema properly
    class ConvertInput(BaseModel):
        value: float = Field(description=f"Value in {from_unit}")
    
    def convert(value: float) -> str:
        """Convert between units."""
        result = value * conversion_factor
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    
    # Create tool dynamically
    tool_instance = StructuredTool.from_function(
        func=convert,
        name=f"convert_{from_unit}_to_{to_unit}",
        description=f"Convert {from_unit} to {to_unit}",
        args_schema=ConvertInput
    )
    
    return tool_instance


# Create multiple converter tools dynamically
km_to_miles = create_converter_tool("kilometers", "miles", 0.621371)
celsius_to_fahrenheit = create_converter_tool("celsius", "fahrenheit", 1.8)

print("✅ Created tools dynamically:")
print(f"   - {km_to_miles.name}")
print(f"   - {celsius_to_fahrenheit.name}")
print()

# Test dynamic tools
result1 = km_to_miles.invoke({"value": 10})
print(f"   Test 1: {result1}")

result2 = celsius_to_fahrenheit.invoke({"value": 20})
print(f"   Test 2: {result2}")
print()


# =============================================================================
# 5. TOOL WITH ERROR HANDLING - Robust Tools
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Handle errors gracefully
# - Provide informative error messages
# - Validate inputs
# - Return helpful feedback
# =============================================================================

print("=" * 80)
print("5. TOOL WITH ERROR HANDLING - Robust Tools")
print("=" * 80)
print()

@tool
def divide_numbers(dividend: float, divisor: float) -> str:
    """
    Divide two numbers with error handling.
    
    Use this tool to divide numbers safely.
    
    Args:
        dividend: The number to be divided
        divisor: The number to divide by
    
    Returns:
        Division result or error message
    """
    try:
        # Validate inputs
        if not isinstance(dividend, (int, float)):
            return f"Error: dividend must be a number, got {type(dividend).__name__}"
        
        if not isinstance(divisor, (int, float)):
            return f"Error: divisor must be a number, got {type(divisor).__name__}"
        
        # Check for division by zero
        if divisor == 0:
            return "Error: Cannot divide by zero"
        
        # Perform division
        result = dividend / divisor
        
        # Check for special cases
        if result == float('inf'):
            return "Error: Result is infinity"
        
        if result != result:  # NaN check
            return "Error: Result is not a number"
        
        return f"Result: {dividend} ÷ {divisor} = {result}"
        
    except Exception as e:
        return f"Error: {str(e)}"


print("✅ Created tool with error handling:")
print(f"   Name: {divide_numbers.name}")
print()

# Test error handling
print("   Test cases:")
print(f"   - Normal: {divide_numbers.invoke({'dividend': 10, 'divisor': 2})}")
print(f"   - Divide by zero: {divide_numbers.invoke({'dividend': 10, 'divisor': 0})}")
print()


# =============================================================================
# 6. TOOL WITH CALLBACKS - Lifecycle Hooks
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Monitor tool execution
# - Log tool usage
# - Track performance
# - Debug tool behavior
# =============================================================================

print("=" * 80)
print("6. TOOL WITH CALLBACKS - Lifecycle Hooks")
print("=" * 80)
print()

class LoggingTool(BaseTool):
    """Tool with logging callbacks."""
    
    name: str = "logging_calculator"
    description: str = "Calculator with execution logging"
    
    def _run(
        self,
        expression: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute with logging."""
        # Log start
        if run_manager:
            run_manager.on_text(f"Starting calculation: {expression}\n", color="blue")
        
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            
            # Log success
            if run_manager:
                run_manager.on_text(f"Calculation successful: {result}\n", color="green")
            
            return f"Result: {result}"
            
        except Exception as e:
            # Log error
            if run_manager:
                run_manager.on_text(f"Calculation failed: {e}\n", color="red")
            
            return f"Error: {str(e)}"


logging_calc = LoggingTool()

print("✅ Created tool with callbacks:")
print(f"   Name: {logging_calc.name}")
print(f"   Features: Execution logging, performance tracking")
print()


# =============================================================================
# 7. ASYNC TOOLS - Asynchronous Execution
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Use for I/O-bound operations
# - Implement _arun() method
# - Better performance for API calls
# - Non-blocking execution
# =============================================================================

print("=" * 80)
print("7. ASYNC TOOLS - Asynchronous Execution")
print("=" * 80)
print()

class AsyncWeatherTool(BaseTool):
    """Async weather tool for non-blocking API calls."""
    
    name: str = "async_weather"
    description: str = "Get weather asynchronously"
    
    def _run(self, city: str) -> str:
        """Sync fallback."""
        return f"Weather in {city}: Sunny, 22°C"
    
    async def _arun(self, city: str) -> str:
        """Async implementation."""
        # Simulate async API call
        await asyncio.sleep(0.1)
        return f"[Async] Weather in {city}: Sunny, 22°C"


async_weather = AsyncWeatherTool()

print("✅ Created async tool:")
print(f"   Name: {async_weather.name}")
print(f"   Features: Non-blocking execution, async API calls")
print()

# Test async tool
async def test_async():
    result = await async_weather._arun("Tokyo")
    return result

result = asyncio.run(test_async())
print(f"   Async test: {result}")
print()


# =============================================================================
# 8. TOOL WITH ARTIFACTS - Structured Data Return
# =============================================================================
# Official Pattern: https://docs.langchain.com/oss/python/langchain/tools
#
# Key Concepts:
# - Return structured data
# - Use for complex outputs
# - Better for downstream processing
# - Type-safe returns
# =============================================================================

print("=" * 80)
print("8. TOOL WITH ARTIFACTS - Structured Data")
print("=" * 80)
print()

class DataAnalysisResult(BaseModel):
    """Structured result from data analysis."""
    mean: float = Field(description="Mean value")
    median: float = Field(description="Median value")
    std_dev: float = Field(description="Standard deviation")
    count: int = Field(description="Number of data points")


@tool
def analyze_data(numbers: str) -> str:
    """
    Analyze a list of numbers and return statistics.
    
    Use this tool to get statistical analysis of numeric data.
    
    Args:
        numbers: Comma-separated list of numbers (e.g., "1,2,3,4,5")
    
    Returns:
        Statistical analysis as formatted string
    """
    try:
        # Parse numbers
        data = [float(x.strip()) for x in numbers.split(",")]
        
        if not data:
            return "Error: No numbers provided"
        
        # Calculate statistics
        mean = sum(data) / len(data)
        sorted_data = sorted(data)
        n = len(sorted_data)
        median = sorted_data[n//2] if n % 2 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5
        
        # Create structured result
        result = DataAnalysisResult(
            mean=mean,
            median=median,
            std_dev=std_dev,
            count=len(data)
        )
        
        # Return formatted string
        return f"""Statistical Analysis:
- Count: {result.count}
- Mean: {result.mean:.2f}
- Median: {result.median:.2f}
- Std Dev: {result.std_dev:.2f}"""
        
    except Exception as e:
        return f"Error: {str(e)}"


print("✅ Created tool with structured output:")
print(f"   Name: {analyze_data.name}")
print(f"   Output: DataAnalysisResult (mean, median, std_dev, count)")
print()

# Test structured output
result = analyze_data.invoke({"numbers": "10, 20, 30, 40, 50"})
print(f"   Test result:\n{result}")
print()


# =============================================================================
# 9. USING TOOLS WITH AGENTS - Practical Example
# =============================================================================

print("=" * 80)
print("9. USING TOOLS WITH AGENTS - Practical Example")
print("=" * 80)
print()

# Create model
model_manager = ModelManager()
model = model_manager.create_model(
    provider_name="deepseek",
    model_name="deepseek-chat",
    temperature=0.7,
    max_tokens=2000
)

# Collect all tools
all_tools = [
    calculator,
    search_database,
    send_email,
    weather_tool,
    divide_numbers,
    analyze_data
]

# Create agent with tools
checkpointer = MemorySaver()
config = {"configurable": {"thread_id": "tool-demo"}}

system_prompt = """You are a helpful AI assistant with access to various tools.

Available tools:
- calculator: Perform mathematical calculations
- search_database: Search for information
- send_email: Send emails
- get_weather: Get weather information
- divide_numbers: Divide numbers safely
- analyze_data: Analyze numeric data

Use tools when needed to provide accurate information."""

agent = create_agent(
    model=model,
    tools=all_tools,
    checkpointer=checkpointer,
    system_prompt=system_prompt
)

print("✅ Created agent with 6 tools")
print()

# Test agent with tools
print("-" * 80)
print("Example: Agent Using Tools")
print("-" * 80)

query = "What is 15 multiplied by 23?"
print(f"👤 User: {query}")

result = agent.invoke(
    {"messages": [HumanMessage(content=query)]},
    config=config
)

response = result["messages"][-1].content
print(f"🤖 Agent: {response}")
print()


# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("EXAMPLE COMPLETE!")
print("=" * 80)

print("\n🎓 Tool Creation Methods Covered:")
print("   1. @tool Decorator - Simple function-based tools")
print("   2. StructuredTool - Complex input schemas with Pydantic")
print("   3. BaseTool Class - Custom tool classes with full control")
print("   4. Tool from Function - Dynamic tool creation")
print("   5. Error Handling - Robust tools with validation")
print("   6. Callbacks - Tools with lifecycle hooks")
print("   7. Async Tools - Non-blocking execution")
print("   8. Structured Output - Tools returning complex data")

print("\n📚 Key Takeaways:")
print("   - Use @tool for simple tools")
print("   - Use StructuredTool for complex inputs")
print("   - Use BaseTool for maximum control")
print("   - Always write clear docstrings (model reads them!)")
print("   - Handle errors gracefully")
print("   - Return strings for best results")
print("   - Use type hints for input validation")

print("\n💡 Best Practices:")
print("   - Clear, descriptive docstrings")
print("   - Type hints for all parameters")
print("   - Error handling with informative messages")
print("   - Return strings the model can interpret")
print("   - Test tools before using with agents")
print("   - Use async for I/O-bound operations")

print("\n📖 Official Documentation:")
print("   https://docs.langchain.com/oss/python/langchain/tools")

print()
