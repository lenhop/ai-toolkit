#!/usr/bin/env python3
"""
Advanced AI Agent Patterns - Following LangChain Official Docs

This example demonstrates advanced agent patterns inspired by official LangChain documentation:
1. Dynamic Model Selection - Choose model based on task complexity  
2. Dynamic Prompt - Modify system prompt based on context
3. Structured Output - Using proper response formatting
4. Best Practices - Production-ready patterns

Official References:
- Agents: https://docs.langchain.com/oss/python/langchain/agents
- Structured Output: https://docs.langchain.com/oss/python/langchain/structured-output

Note: This example uses patterns compatible with current LangChain version.
Some advanced features (like @wrap_model_call) may require newer versions.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal, Any
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, dynamic_prompt
from langgraph.checkpoint.memory import MemorySaver

# Import ai_toolkit
from ai_toolkit.models.model_manager import ModelManager


# =============================================================================
# Part 1: Define Structured Output Schemas
# =============================================================================

class WeatherResponse(BaseModel):
    """Structured response for weather queries."""
    city: str = Field(description="The city name")
    temperature: int = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition (sunny, cloudy, rainy, etc.)")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level of the information"
    )


class CalculationResponse(BaseModel):
    """Structured response for calculation queries."""
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The calculation result")
    explanation: str = Field(description="Brief explanation of the calculation")


class TaskAnalysis(BaseModel):
    """Analysis of task complexity for dynamic model selection."""
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="Task complexity level"
    )
    requires_tools: bool = Field(description="Whether tools are needed")
    reasoning: str = Field(description="Reasoning for the complexity assessment")


# =============================================================================
# Part 2: Define Tools
# =============================================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    
    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a city.
    
    Args:
        city: Name of the city to get weather for
    
    Returns:
        Weather information for the specified city
    """
    # Simulated weather data
    weather_data = {
        "beijing": {"temp": 22, "condition": "Sunny"},
        "shanghai": {"temp": 18, "condition": "Cloudy"},
        "new york": {"temp": 5, "condition": "Cold"},
        "london": {"temp": 10, "condition": "Foggy"},
        "tokyo": {"temp": 15, "condition": "Clear"},
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return f"Weather in {city}: {data['condition']}, {data['temp']}°C"
    else:
        return f"Weather data not available for {city}"


# =============================================================================
# Step 1: Dynamic Model Selection (Concept Demonstration)
# =============================================================================

def demonstrate_dynamic_model_selection():
    """
    Demonstrate dynamic model selection concept.
    
    Official Pattern: Select models based on task complexity
    Reference: https://docs.langchain.com/oss/python/langchain/agents#dynamic-model
    
    In production, you would use @wrap_model_call decorator.
    This example shows the concept with manual selection.
    """
    print("\n" + "=" * 80)
    print("STEP 1: Dynamic Model Selection (Concept)")
    print("=" * 80)
    print("\nOfficial Pattern: Dynamic model selection based on task")
    print("Purpose: Choose optimal model for each task")
    print()
    
    # Create model manager
    model_manager = ModelManager()
    
    # Function to analyze task complexity
    def analyze_task_complexity(query: str) -> str:
        """Analyze query to determine complexity."""
        complex_keywords = ["analyze", "complex", "detailed", "comprehensive", "explain in depth"]
        
        if any(keyword in query.lower() for keyword in complex_keywords):
            return "complex"
        elif any(word in query.lower() for word in ["calculate", "compute", "math"]):
            return "medium"
        else:
            return "simple"
    
    # Function to select model based on complexity
    def select_model(complexity: str):
        """Select appropriate model based on task complexity."""
        print(f"🔄 Task Complexity: {complexity}")
        
        if complexity == "complex":
            print(f"   Selected: deepseek-chat (for complex reasoning)")
            # In production, might use a more powerful model
            return model_manager.create_model(
                provider_name="deepseek",
                model_name="deepseek-chat",
                temperature=0.7
            )
        elif complexity == "medium":
            print(f"   Selected: deepseek-chat (for calculations)")
            return model_manager.create_model(
                provider_name="deepseek",
                model_name="deepseek-chat",
                temperature=0.5
            )
        else:
            print(f"   Selected: deepseek-chat (for simple queries)")
            return model_manager.create_model(
                provider_name="deepseek",
                model_name="deepseek-chat",
                temperature=0.3
            )
    
    print("📝 Dynamic Model Selection Logic:")
    print("   - Simple tasks → Lower temperature (more focused)")
    print("   - Medium tasks → Medium temperature")
    print("   - Complex tasks → Higher temperature (more creative)")
    print()
    
    # Test with different queries
    test_queries = [
        "What is 2 + 2?",
        "Calculate 123 * 456",
        "Analyze the implications of AI in healthcare"
    ]
    
    for query in test_queries:
        print(f"\n📤 Query: '{query}'")
        complexity = analyze_task_complexity(query)
        model = select_model(complexity)
        print(f"   ✅ Model ready with temperature={model.temperature}")
    
    print()
    return model_manager


# =============================================================================
# Step 2: Dynamic Prompt (Concept Demonstration)
# =============================================================================

def demonstrate_dynamic_prompt():
    """
    Demonstrate dynamic prompt generation based on context.
    
    Official Pattern: Generate prompts based on runtime context
    Reference: https://docs.langchain.com/oss/python/langchain/agents#dynamic-system-prompt
    
    In production, you would use @dynamic_prompt decorator.
    This example shows the concept with manual prompt selection.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Dynamic Prompt (Concept)")
    print("=" * 80)
    print("\nOfficial Pattern: Context-aware prompt generation")
    print("Purpose: Adapt system prompt based on query type")
    print()
    
    # Define different prompts for different contexts
    prompts = {
        "weather": """You are a weather information specialist.

Your role:
- Provide accurate weather information
- Use the weather tool when needed
- Be concise and friendly
- Always include temperature in Celsius""",
        
        "math": """You are a mathematical calculation expert.

Your role:
- Perform accurate calculations
- Use the calculator tool for complex math
- Explain your calculations step-by-step
- Show your reasoning clearly""",
        
        "general": """You are a helpful AI assistant with access to tools.

Your capabilities:
- Answer questions accurately
- Use tools when appropriate
- Be friendly and professional
- Provide clear explanations"""
    }
    
    # Function to select prompt based on query
    def select_prompt(query: str) -> tuple[str, str]:
        """Select appropriate prompt based on query content."""
        query_lower = query.lower()
        
        if "weather" in query_lower or "temperature" in query_lower:
            return "weather", prompts["weather"]
        elif any(word in query_lower for word in ["calculate", "math", "compute", "multiply", "add"]):
            return "math", prompts["math"]
        else:
            return "general", prompts["general"]
    
    print("📝 Dynamic Prompt Selection Logic:")
    print("   - Weather queries → Weather specialist prompt")
    print("   - Math queries → Math expert prompt")
    print("   - Other queries → General assistant prompt")
    print()
    
    # Test with different queries
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate 15 * 23",
        "Tell me about Python programming"
    ]
    
    for query in test_queries:
        print(f"\n� Query: '{query}'")
        context, prompt = select_prompt(query)
        print(f"🔄 Selected Context: {context}")
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   First line: {prompt.split(chr(10))[0]}")
    
    print()
    
    # Create agent with dynamic prompt example
    model_manager = ModelManager()
    model = model_manager.create_model(
        provider_name="deepseek",
        model_name="deepseek-chat",
        temperature=0.7
    )
    
    # Example: Using selected prompt with agent
    query = "What's the weather in Beijing?"
    context, system_prompt = select_prompt(query)
    
    print(f"\n📤 Full Example:")
    print(f"   Query: '{query}'")
    print(f"   Context: {context}")
    
    agent = create_agent(
        model=model,
        tools=[calculator, get_weather],
        system_prompt=system_prompt,
        checkpointer=MemorySaver()
    )
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "2"}}
    )
    
    print(f"   Response: {result['messages'][-1].content[:100]}...")
    print()
    
    return agent


# =============================================================================
# Step 3: Structured Output (Concept Demonstration)
# =============================================================================

def demonstrate_structured_output():
    """
    Demonstrate structured output concept.
    
    Official Pattern: Get structured data from agent responses
    Reference: https://docs.langchain.com/oss/python/langchain/structured-output
    
    This example shows how to request and parse structured output.
    """
    print("\n" + "=" * 80)
    print("STEP 3: Structured Output (Concept)")
    print("=" * 80)
    print("\nOfficial Pattern: Structured response formatting")
    print("Purpose: Get predictable, parseable output from agents")
    print()
    
    # Create model
    model_manager = ModelManager()
    model = model_manager.create_model(
        provider_name="deepseek",
        model_name="deepseek-chat",
        temperature=0.7
    )
    
    print("✅ Model created: deepseek-chat")
    print()
    
    # Define structured output schema
    print("📋 Structured Output Schema: WeatherResponse")
    print("   Fields:")
    print("   - city: str")
    print("   - temperature: int")
    print("   - condition: str")
    print("   - confidence: Literal['high', 'medium', 'low']")
    print()
    
    # Create prompt that requests structured output
    structured_prompt = """Please provide weather information in this exact JSON format:
{
    "city": "city name",
    "temperature": temperature_in_celsius,
    "condition": "weather condition",
    "confidence": "high/medium/low"
}

Question: What's the weather in Beijing?

Respond ONLY with the JSON, no other text."""
    
    print("📤 Requesting structured output...")
    print("   Using prompt engineering to get JSON format")
    
    result = model.invoke([
        SystemMessage(content="You provide structured JSON responses."),
        HumanMessage(content=structured_prompt)
    ])
    
    print(f"\n🤖 Raw Response:")
    print(f"{result.content}")
    
    # Parse the JSON response
    try:
        import json
        content = result.content.strip()
        
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(content)
        
        # Validate against our schema
        weather_response = WeatherResponse(**parsed)
        
        print(f"\n✅ Parsed Structured Response:")
        print(f"   City: {weather_response.city}")
        print(f"   Temperature: {weather_response.temperature}°C")
        print(f"   Condition: {weather_response.condition}")
        print(f"   Confidence: {weather_response.confidence}")
        print()
        print("✅ Successfully validated against Pydantic schema")
        
    except Exception as e:
        print(f"\n⚠️  Parsing error: {e}")
    
    print()
    
    # Show the official pattern (requires compatible LangChain version)
    print("📚 Official Pattern (for reference):")
    print("   In newer LangChain versions, you can use:")
    print()
    print("   from langchain.agents import ToolStrategy")
    print("   ")
    print("   agent = create_agent(")
    print("       model=model,")
    print("       tools=tools,")
    print("       response_format=ToolStrategy(WeatherResponse)")
    print("   )")
    print()
    print("   result = agent.invoke({...})")
    print("   structured_data = result['structured_response']")
    print()


# =============================================================================
# Step 4: Best Practices Summary
# =============================================================================

def demonstrate_best_practices():
    """
    Demonstrate best practices for production agents.
    
    Based on official LangChain patterns and recommendations.
    """
    print("\n" + "=" * 80)
    print("STEP 4: Best Practices for Production Agents")
    print("=" * 80)
    print()
    
    print("📚 Official LangChain Best Practices:")
    print()
    
    print("1️⃣  Model Selection:")
    print("   ✓ Choose models based on task complexity")
    print("   ✓ Use faster models for simple tasks")
    print("   ✓ Use powerful models for complex reasoning")
    print("   ✓ Consider cost vs performance tradeoffs")
    print()
    
    print("2️⃣  Prompt Engineering:")
    print("   ✓ Provide clear system prompts")
    print("   ✓ Adapt prompts to task context")
    print("   ✓ Include tool usage guidelines")
    print("   ✓ Set clear expectations for output format")
    print()
    
    print("3️⃣  Structured Output:")
    print("   ✓ Define clear Pydantic schemas")
    print("   ✓ Use ToolStrategy for models without native support")
    print("   ✓ Use ProviderStrategy when available")
    print("   ✓ Validate output against schemas")
    print()
    
    print("4️⃣  Error Handling:")
    print("   ✓ Implement retry logic for tool failures")
    print("   ✓ Provide clear error messages")
    print("   ✓ Handle schema validation errors")
    print("   ✓ Set appropriate timeouts")
    print()
    
    print("5️⃣  Memory Management:")
    print("   ✓ Use checkpointers for conversation history")
    print("   ✓ Implement proper thread_id management")
    print("   ✓ Consider memory limits for long conversations")
    print("   ✓ Use SqliteSaver for persistence")
    print()
    
    print("6️⃣  Tool Design:")
    print("   ✓ Write clear tool descriptions")
    print("   ✓ Define explicit parameters")
    print("   ✓ Return structured, parseable results")
    print("   ✓ Handle errors gracefully")
    print()
    
    print("7️⃣  Performance:")
    print("   ✓ Use streaming for long responses")
    print("   ✓ Implement caching where appropriate")
    print("   ✓ Monitor token usage")
    print("   ✓ Optimize tool execution")
    print()
    
    # Create a production-ready agent example
    print("📦 Production-Ready Agent Example:")
    print()
    
    model_manager = ModelManager()
    model = model_manager.create_model(
        provider_name="deepseek",
        model_name="deepseek-chat",
        temperature=0.7,
        max_tokens=2000
    )
    
    system_prompt = """You are a professional AI assistant.

Guidelines:
- Be accurate and helpful
- Use tools when appropriate
- Provide clear explanations
- Handle errors gracefully
- Format responses clearly"""
    
    agent = create_agent(
        model=model,
        tools=[calculator, get_weather],
        system_prompt=system_prompt,
        checkpointer=MemorySaver()
    )
    
    print("✅ Production agent created with:")
    print("   - Clear system prompt")
    print("   - Well-defined tools")
    print("   - Memory management")
    print("   - Error handling")
    print()
    
    return agent


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Run all advanced pattern demonstrations."""
    print("\n" + "=" * 80)
    print("ADVANCED AI AGENT PATTERNS - OFFICIAL LANGCHAIN CONCEPTS")
    print("=" * 80)
    print("\nBased on official LangChain documentation:")
    print("  📚 https://docs.langchain.com/oss/python/langchain/agents")
    print("  📚 https://docs.langchain.com/oss/python/langchain/structured-output")
    print()
    print("This example demonstrates key concepts:")
    print("  ✓ Dynamic model selection")
    print("  ✓ Dynamic prompt generation")
    print("  ✓ Structured output")
    print("  ✓ Production best practices")
    print()
    print("Note: Some advanced features (middleware decorators) require")
    print("      newer LangChain versions. This example shows the concepts.")
    
    try:
        # Step 1: Dynamic model selection
        demonstrate_dynamic_model_selection()
        
        # Step 2: Dynamic prompt
        demonstrate_dynamic_prompt()
        
        # Step 3: Structured output
        demonstrate_structured_output()
        
        # Step 4: Best practices
        demonstrate_best_practices()
        
        print("\n" + "=" * 80)
        print("ADVANCED PATTERNS COMPLETE!")
        print("=" * 80)
        print("\n🎓 What you learned:")
        print("   1. Dynamic model selection concepts")
        print("   2. Context-aware prompt generation")
        print("   3. Structured output with Pydantic")
        print("   4. Production-ready agent patterns")
        print("   5. Official LangChain best practices")
        
        print("\n💡 Next Steps:")
        print("   - Implement custom model selection logic")
        print("   - Create domain-specific dynamic prompts")
        print("   - Define your own structured output schemas")
        print("   - Build production agents with error handling")
        print("   - Explore LangChain middleware in newer versions")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
