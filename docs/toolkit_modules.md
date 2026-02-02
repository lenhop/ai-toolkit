# AI Toolkit Modules Documentation

This document provides an overview of the new reusable modules extracted from practice examples.

## Overview

The AI Toolkit now includes comprehensive utilities for building LangChain-based AI applications, extracted from the practice examples (11-19, RAG files).

## Module Structure

```
ai_toolkit/
├── agents/              # NEW - Agent creation helpers
│   ├── agent_helpers.py
│   └── middleware_utils.py
├── tools/               # NEW - Tool utilities
│   └── tool_utils.py
├── rag/                 # NEW - RAG utilities
│   ├── loaders.py
│   ├── splitters.py
│   └── retrievers.py
├── memory/              # UPDATED - Memory management
│   └── memory_manager.py
├── messages/            # EXISTING - Message building
│   └── message_builder.py
├── streaming/           # EXISTING - Streaming utilities
│   └── stream_handler.py
└── parsers/             # EXISTING - Output parsing
    └── output_parser.py
```

## 1. Agents Module (`ai_toolkit/agents/`)

### Agent Helpers (`agent_helpers.py`)

Simplified agent creation functions based on practice examples.

**Functions:**
- `create_agent_with_tools()` - Create agent with tools and defaults
- `create_agent_with_memory()` - Create agent with checkpointer
- `create_streaming_agent()` - Create streaming-optimized agent
- `create_structured_output_agent()` - Create agent with structured output

**Example:**
```python
from ai_toolkit.models import ModelManager
from ai_toolkit.agents import create_agent_with_tools
from ai_toolkit.tools import create_search_tool

manager = ModelManager()
model = manager.create_model("deepseek")
search = create_search_tool()

agent = create_agent_with_tools(
    model=model,
    tools=[search],
    system_prompt="You are a helpful assistant."
)
```

**Based on:** `examples/practice/13_agent_base.py`, `14_agent_advanced.py`

### Middleware Utilities (`middleware_utils.py`)

Common middleware patterns for agents.

**Functions:**
- `create_dynamic_model_selector()` - Switch models based on complexity
- `create_tool_error_handler()` - Handle tool execution errors
- `create_context_based_prompt()` - Generate prompts from context

**Example:**
```python
from ai_toolkit.agents.middleware_utils import create_dynamic_model_selector

model_selector = create_dynamic_model_selector(
    basic_model=basic_model,
    advanced_model=advanced_model,
    threshold=10  # Switch after 10 messages
)

agent = create_agent_with_tools(
    model=basic_model,
    tools=[search],
    middleware=[model_selector]
)
```

**Based on:** `examples/practice/13_agent_base.py`

## 2. Tools Module (`ai_toolkit/tools/`)

### Tool Utilities (`tool_utils.py`)

Factory functions for common tools.

**Functions:**
- `create_search_tool()` - Generic search tool
- `create_weather_tool()` - Weather lookup tool
- `create_calculator_tool()` - Math calculation tool
- `create_memory_access_tool()` - Read from agent memory
- `create_memory_update_tool()` - Write to agent memory
- `wrap_tool_with_error_handler()` - Add error handling to any tool

**Example:**
```python
from ai_toolkit.tools import (
    create_search_tool,
    create_weather_tool,
    create_calculator_tool
)

# Create tools with defaults
search = create_search_tool()
weather = create_weather_tool()
calculator = create_calculator_tool(safe_mode=True)

# Create tools with custom functions
def my_search(query: str) -> str:
    return f"Results for: {query}"

custom_search = create_search_tool(
    search_function=my_search,
    description="Search the knowledge base."
)
```

**Based on:** `examples/practice/13_agent_base.py`, `16_tool_base.py`

## 3. RAG Module (`ai_toolkit/rag/`)

### Document Loaders (`loaders.py`)

Load documents from various sources.

**Functions:**
- `load_web_document()` - Load from URL
- `load_pdf_document()` - Load PDF files
- `load_json_document()` - Load JSON files
- `load_csv_document()` - Load CSV files

**Example:**
```python
from ai_toolkit.rag import load_web_document

# Load entire page
docs = load_web_document("https://example.com")

# Load only paragraphs
docs = load_web_document(
    "https://example.com",
    selector="p"
)
```

**Based on:** `examples/practice/rag_01_loader_base.py`

### Document Splitters (`splitters.py`)

Split documents into chunks for RAG.

**Functions:**
- `split_document_recursive()` - RecursiveCharacterTextSplitter wrapper
- `split_with_overlap()` - Split with custom separators
- `split_for_chinese()` - Chinese-optimized splitting

**Example:**
```python
from ai_toolkit.rag import split_document_recursive, split_for_chinese

# Standard splitting
chunks = split_document_recursive(
    docs,
    chunk_size=1000,
    chunk_overlap=200
)

# Chinese-optimized splitting
chinese_chunks = split_for_chinese(
    docs,
    chunk_size=1000,
    chunk_overlap=200
)
```

**Based on:** `examples/practice/rag_03_split_document.py`, `rag_04_agent_workflow.py`

### Retrievers (`retrievers.py`)

RAG retrieval and agent creation.

**Functions:**
- `create_vector_store()` - Create vector store with embeddings
- `create_retrieval_tool()` - Create retrieval tool for agent
- `create_rag_agent()` - Complete RAG agent factory
- `retrieve_document_only()` - Document-only retrieval (no AI)
- `retrieve_with_priority()` - Document first, AI fallback

**Example:**
```python
from ai_toolkit.rag import (
    load_web_document,
    split_document_recursive,
    create_vector_store,
    create_rag_agent
)
from langchain_community.embeddings import DashScopeEmbeddings

# 1. Load and split
docs = load_web_document("https://example.com/article")
chunks = split_document_recursive(docs, chunk_size=1000)

# 2. Create embeddings and vector store
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key="your-api-key"
)
vector_store = create_vector_store(chunks, embeddings)

# 3. Create RAG agent
agent = create_rag_agent(model, vector_store, k=3)

# 4. Ask questions
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the main topic?"}]
})
```

**Based on:** `examples/practice/rag_04_agent_workflow.py`, `rag_05_retrieval_document_only.py`, `rag_06_priority_fallback.py`

## 4. Memory Module (`ai_toolkit/memory/`)

### Memory Manager (`memory_manager.py`)

Comprehensive memory management utilities (already existed, well-implemented).

**Key Features:**
- Multiple checkpointer types (InMemory, PostgreSQL)
- Message trimming strategies
- Conversation summarization
- Middleware creation utilities

**Example:**
```python
from ai_toolkit.memory import MemoryManager
from ai_toolkit.memory.memory_manager import create_trimming_middleware

# Create checkpointer
manager = MemoryManager()
checkpointer = manager.create_checkpointer("inmemory")

# Create trimming middleware
trim_middleware = create_trimming_middleware(max_messages=10)

# Use in agent
agent = create_agent_with_memory(
    model=model,
    tools=[search],
    checkpointer_type="inmemory"
)
```

**Based on:** `examples/practice/17_memory_base.py`

## 5. Existing Modules (Already Well-Implemented)

### Messages Module (`ai_toolkit/messages/`)
- `MessageBuilder` - Fluent interface for building messages
- Based on: `examples/practice/15_message_base.py`

### Streaming Module (`ai_toolkit/streaming/`)
- `StreamHandler` - Streaming utilities
- Based on: `examples/practice/18_streaming_base.py`

### Parsers Module (`ai_toolkit/parsers/`)
- `JsonOutputParser`, `PydanticOutputParser`, etc.
- Based on: `examples/practice/19_output_structure.py`

## Quick Start

```python
# 1. Install dependencies
# pip install ai-toolkit langchain langchain-community

# 2. Import and use
from ai_toolkit.models import ModelManager
from ai_toolkit.agents import create_agent_with_tools
from ai_toolkit.tools import create_search_tool, create_weather_tool

# 3. Create model
manager = ModelManager()
model = manager.create_model("deepseek")

# 4. Create tools
search = create_search_tool()
weather = create_weather_tool()

# 5. Create agent
agent = create_agent_with_tools(
    model=model,
    tools=[search, weather],
    system_prompt="You are a helpful assistant."
)

# 6. Use agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in SF?"}]
})
print(result["messages"][-1].content)
```

## Complete RAG Example

```python
from ai_toolkit.models import ModelManager
from ai_toolkit.rag import (
    load_web_document,
    split_document_recursive,
    create_vector_store,
    create_rag_agent
)
from langchain_community.embeddings import DashScopeEmbeddings

# Load and split documents
docs = load_web_document("https://example.com/article")
chunks = split_document_recursive(docs, chunk_size=1000)

# Create embeddings and vector store
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key="your-api-key"
)
vector_store = create_vector_store(chunks, embeddings)

# Create model and RAG agent
manager = ModelManager()
model = manager.create_model("qwen")
agent = create_rag_agent(model, vector_store, k=3)

# Ask questions
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the main topic?"}]
})
print(result["messages"][-1].content)
```

## Summary

**New Modules Created:**
- `ai_toolkit/agents/` - Agent helpers and middleware (2 files)
- `ai_toolkit/tools/` - Tool utilities (1 file)
- `ai_toolkit/rag/` - RAG utilities (3 files)

**Total Functions:** ~50+ reusable helper functions

**Based on:** Practice examples 11-19, RAG examples 01-06

**LangChain Version:** 1.0  
**Python Version:** 3.11

For detailed usage examples, see `examples/toolkit_usage_guide.py`.
