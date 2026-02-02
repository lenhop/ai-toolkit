"""
AI Toolkit Usage Guide - Comprehensive examples

This file demonstrates how to use the new ai_toolkit utilities
extracted from the practice examples.

Sections:
    1. Agent Creation Helpers
    2. Tool Utilities
    3. Middleware Utilities
    4. RAG Utilities
    5. Memory Management
    6. Complete RAG Workflow

Based on: examples/practice/*.py
LangChain Version: 1.0
Python Version: 3.11
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 1. AGENT CREATION HELPERS
# =============================================================================

def example_agent_helpers():
    """Demonstrate agent creation helpers."""
    print("=" * 80)
    print("1. Agent Creation Helpers")
    print("=" * 80)
    
    from ai_toolkit.models import ModelManager
    from ai_toolkit.agents import (
        create_agent_with_tools,
        create_agent_with_memory,
        create_streaming_agent,
        create_structured_output_agent,
    )
    from ai_toolkit.tools import create_search_tool, create_weather_tool
    from pydantic import BaseModel, Field
    
    # Create model
    manager = ModelManager()
    model = manager.create_model(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        provider="deepseek"
    )
    
    # Create tools
    search = create_search_tool()
    weather = create_weather_tool()
    
    # Example 1a: Simple agent with tools
    print("\n1a. Simple agent with tools:")
    agent = create_agent_with_tools(
        model=model,
        tools=[search, weather],
        system_prompt="You are a helpful research assistant."
    )
    print("✓ Agent created with tools")
    
    # Example 1b: Agent with memory
    print("\n1b. Agent with memory:")
    agent_with_memory = create_agent_with_memory(
        model=model,
        tools=[search],
        checkpointer_type="inmemory"
    )
    print("✓ Agent created with in-memory checkpointer")
    
    # Example 1c: Streaming agent
    print("\n1c. Streaming agent:")
    streaming_agent = create_streaming_agent(
        model=model,
        tools=[weather]
    )
    print("✓ Streaming agent created")
    
    # Example 1d: Structured output agent
    print("\n1d. Structured output agent:")
    
    class ContactInfo(BaseModel):
        name: str = Field(description="Person's name")
        email: str = Field(description="Email address")
        phone: str = Field(description="Phone number")
    
    structured_agent = create_structured_output_agent(
        model=model,
        tools=[search],
        schema=ContactInfo,
        strategy="tool"
    )
    print("✓ Structured output agent created")
    print()


# =============================================================================
# 2. TOOL UTILITIES
# =============================================================================

def example_tool_utilities():
    """Demonstrate tool utilities."""
    print("=" * 80)
    print("2. Tool Utilities")
    print("=" * 80)
    
    from ai_toolkit.tools import (
        create_search_tool,
        create_weather_tool,
        create_calculator_tool,
        wrap_tool_with_error_handler,
    )
    from langchain.tools import tool
    
    # Example 2a: Create common tools
    print("\n2a. Create common tools:")
    search = create_search_tool(description="Search the knowledge base.")
    weather = create_weather_tool(description="Get current weather.")
    calculator = create_calculator_tool(safe_mode=True)
    print("✓ Created search, weather, and calculator tools")
    
    # Example 2b: Custom search function
    print("\n2b. Custom search function:")
    
    def my_search(query: str) -> str:
        # Your custom search implementation
        return f"Custom search results for: {query}"
    
    custom_search = create_search_tool(
        search_function=my_search,
        description="Search using custom implementation."
    )
    print("✓ Created custom search tool")
    
    # Example 2c: Wrap tool with error handler
    print("\n2c. Wrap tool with error handler:")
    
    @tool
    def risky_tool(input: str) -> str:
        """A tool that might fail."""
        if not input:
            raise ValueError("Input required")
        return f"Processed: {input}"
    
    safe_tool = wrap_tool_with_error_handler(
        risky_tool,
        error_message="Tool failed. Please provide valid input."
    )
    print("✓ Wrapped tool with error handler")
    print()


# =============================================================================
# 3. MIDDLEWARE UTILITIES
# =============================================================================

def example_middleware_utilities():
    """Demonstrate middleware utilities."""
    print("=" * 80)
    print("3. Middleware Utilities")
    print("=" * 80)
    
    from ai_toolkit.models import ModelManager
    from ai_toolkit.agents import create_agent_with_tools
    from ai_toolkit.agents.middleware_utils import (
        create_dynamic_model_selector,
        create_tool_error_handler,
        create_context_based_prompt,
    )
    from ai_toolkit.tools import create_search_tool
    from typing import TypedDict
    
    # Create models
    manager = ModelManager()
    basic_model = manager.create_model(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        provider="deepseek"
    )
    advanced_model = manager.create_model(
        api_key=os.environ.get('QWEN_API_KEY'),
        provider="qwen"
    )
    
    # Example 3a: Dynamic model selector
    print("\n3a. Dynamic model selector:")
    model_selector = create_dynamic_model_selector(
        basic_model=basic_model,
        advanced_model=advanced_model,
        threshold=10
    )
    print("✓ Created dynamic model selector (switches at 10 messages)")
    
    # Example 3b: Tool error handler
    print("\n3b. Tool error handler:")
    error_handler = create_tool_error_handler(
        error_message_template="Tool failed: {error}. Please try again.",
        log_errors=True
    )
    print("✓ Created tool error handler")
    
    # Example 3c: Context-based prompt
    print("\n3c. Context-based prompt:")
    
    class UserContext(TypedDict):
        user_role: str
        expertise_level: str
    
    def generate_prompt(context):
        role = context.get("user_role", "user")
        if role == "expert":
            return "Provide detailed technical responses."
        else:
            return "Explain concepts simply."
    
    prompt_middleware = create_context_based_prompt(generate_prompt)
    print("✓ Created context-based prompt middleware")
    
    # Example 3d: Agent with middleware
    print("\n3d. Agent with middleware:")
    search = create_search_tool()
    agent = create_agent_with_tools(
        model=basic_model,
        tools=[search],
        middleware=[model_selector, error_handler]
    )
    print("✓ Created agent with multiple middleware")
    print()


# =============================================================================
# 4. RAG UTILITIES
# =============================================================================

def example_rag_utilities():
    """Demonstrate RAG utilities."""
    print("=" * 80)
    print("4. RAG Utilities")
    print("=" * 80)
    
    from ai_toolkit.rag import (
        load_web_document,
        split_document_recursive,
        split_for_chinese,
        create_vector_store,
        create_retrieval_tool,
        create_rag_agent,
    )
    from ai_toolkit.models import ModelManager
    
    # Example 4a: Load and split documents
    print("\n4a. Load and split documents:")
    docs = load_web_document(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        selector="p"
    )
    print(f"✓ Loaded {len(docs)} document(s)")
    
    chunks = split_document_recursive(
        docs,
        chunk_size=1000,
        chunk_overlap=200
    )
    print(f"✓ Split into {len(chunks)} chunks")
    
    # Example 4b: Create vector store (requires embeddings)
    print("\n4b. Create vector store:")
    print("  (Skipped - requires API key for embeddings)")
    # embeddings = DashScopeEmbeddings(...)
    # vector_store = create_vector_store(chunks, embeddings)
    
    # Example 4c: Chinese document splitting
    print("\n4c. Chinese document splitting:")
    chinese_chunks = split_for_chinese(
        docs,
        chunk_size=1000,
        chunk_overlap=200
    )
    print(f"✓ Split with Chinese-friendly separators: {len(chinese_chunks)} chunks")
    print()


# =============================================================================
# 5. MEMORY MANAGEMENT
# =============================================================================

def example_memory_management():
    """Demonstrate memory management utilities."""
    print("=" * 80)
    print("5. Memory Management")
    print("=" * 80)
    
    from ai_toolkit.memory import MemoryManager
    from ai_toolkit.memory.memory_manager import (
        create_trimming_middleware,
        create_deletion_middleware,
    )
    
    # Example 5a: Create checkpointers
    print("\n5a. Create checkpointers:")
    manager = MemoryManager()
    
    inmemory_checkpointer = manager.create_checkpointer("inmemory")
    print("✓ Created in-memory checkpointer")
    
    # postgres_checkpointer = manager.create_checkpointer(
    #     "postgres",
    #     db_uri="postgresql://user:pass@localhost:5432/db"
    # )
    # print("✓ Created PostgreSQL checkpointer")
    
    # Example 5b: Create config
    print("\n5b. Create config:")
    config = MemoryManager.create_config("user-123")
    print(f"✓ Created config: {config}")
    
    # Example 5c: Create trimmer
    print("\n5c. Create message trimmer:")
    trimmer = MemoryManager.create_trimmer(
        strategy="keep_first_and_recent",
        max_messages=10
    )
    print("✓ Created message trimmer (keeps first + 10 recent)")
    
    # Example 5d: Create middleware
    print("\n5d. Create trimming middleware:")
    trim_middleware = create_trimming_middleware(max_messages=5)
    print("✓ Created trimming middleware")
    
    delete_middleware = create_deletion_middleware(delete_count=2)
    print("✓ Created deletion middleware")
    print()


# =============================================================================
# 6. COMPLETE RAG WORKFLOW
# =============================================================================

def example_complete_rag_workflow():
    """Demonstrate complete RAG workflow."""
    print("=" * 80)
    print("6. Complete RAG Workflow (Conceptual)")
    print("=" * 80)
    
    print("""
Complete RAG workflow steps:

1. Load documents:
   >>> from ai_toolkit.rag import load_web_document
   >>> docs = load_web_document("https://example.com/article")

2. Split into chunks:
   >>> from ai_toolkit.rag import split_document_recursive
   >>> chunks = split_document_recursive(docs, chunk_size=1000)

3. Create embeddings:
   >>> from langchain_community.embeddings import DashScopeEmbeddings
   >>> embeddings = DashScopeEmbeddings(
   ...     model="text-embedding-v3",
   ...     dashscope_api_key="your-api-key"
   ... )

4. Create vector store:
   >>> from ai_toolkit.rag import create_vector_store
   >>> vector_store = create_vector_store(chunks, embeddings)

5. Create model:
   >>> from ai_toolkit.models import ModelManager
   >>> manager = ModelManager()
   >>> model = manager.create_model("qwen")

6. Create RAG agent:
   >>> from ai_toolkit.rag import create_rag_agent
   >>> agent = create_rag_agent(model, vector_store, k=3)

7. Ask questions:
   >>> result = agent.invoke({
   ...     "messages": [{"role": "user", "content": "What is the main topic?"}]
   ... })
   >>> print(result["messages"][-1].content)

Alternative patterns:

- Document-only retrieval (no AI training data):
  >>> from ai_toolkit.rag import retrieve_document_only
  >>> result = retrieve_document_only(vector_store, model, "query", k=3)

- Priority pattern (document first, AI fallback):
  >>> from ai_toolkit.rag import retrieve_with_priority
  >>> result = retrieve_with_priority(
  ...     vector_store, model, "query", use_ai_fallback=True
  ... )
  >>> print(f"Source: {result['source']}")  # "document" or "ai_model"
    """)
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AI TOOLKIT USAGE GUIDE")
    print("=" * 80 + "\n")
    
    # Run examples
    example_agent_helpers()
    example_tool_utilities()
    example_middleware_utilities()
    example_rag_utilities()
    example_memory_management()
    example_complete_rag_workflow()
    
    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nFor more details, see:")
    print("  - ai_toolkit/agents/agent_helpers.py")
    print("  - ai_toolkit/tools/tool_utils.py")
    print("  - ai_toolkit/agents/middleware_utils.py")
    print("  - ai_toolkit/rag/")
    print("  - ai_toolkit/memory/memory_manager.py")
    print()
