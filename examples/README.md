# AI Toolkit Examples

This directory contains comprehensive examples demonstrating how to use the AI Toolkit and LangChain to build AI agents.

## 📚 Learning Path

Follow these examples in order for the best learning experience:

### 1. Model Access Methods Guide
**File:** `1.model_access_methods_guide.py`

Learn 5 different methods to access and use AI models:
- Method 1: Direct model invocation
- Method 2: Using ModelManager
- Method 3: Streaming responses
- Method 4: Batch processing
- Method 5: Async operations

**Run:**
```bash
python examples/1.model_access_methods_guide.py
```

---

### 2. Simple Agent Basics (RECOMMENDED START)
**File:** `2.simple_agent_basics.py`

**⭐ Start here if you're new to AI agents!**

Complete beginner's guide covering all fundamental concepts:
- ✅ Models - Creating and configuring chat models
- ✅ Messages - SystemMessage, HumanMessage, AIMessage
- ✅ Tools - Defining tools with @tool decorator
- ✅ Memory - Using checkpointers for conversation history
- ✅ Agents - Creating ReAct agents with create_agent
- ✅ Structured Output - Getting formatted responses with Pydantic

**Features:**
- Follows official LangChain documentation patterns
- Comprehensive annotations explaining every concept
- Working examples with calculator and weather tools
- Memory demonstration with multi-turn conversations
- Structured output with Pydantic validation

**Run:**
```bash
python examples/2.simple_agent_basics.py
```

**Documentation:** See `README_simple_agent_basics.md` for detailed explanations

---

### 3. Advanced Agent Patterns
**File:** `3.advanced_agent_patterns.py`

Advanced patterns for production-ready agents:
- Dynamic model selection based on task complexity
- Dynamic prompt generation
- Structured output with complex schemas
- Production best practices
- Error handling and retry logic

**Run:**
```bash
python examples/3.advanced_agent_patterns.py
```

**Documentation:** See `README_advanced_patterns.md` for detailed explanations

---

## 📖 Documentation Files

### Quick Reference
**File:** `QUICK_REFERENCE.md`

Fast lookup guide with:
- Quick start template
- Component cheat sheet
- Common patterns
- Parameter reference
- Troubleshooting tips

### Message Types Guide
**File:** `MESSAGE_TYPES_GUIDE.md`

Deep dive into LangChain message types:
- SystemMessage, HumanMessage, AIMessage
- Visual flow diagrams
- Practical examples
- Best practices

### Agent vs Model
**File:** `AGENT_VS_MODEL.md`

Understanding when to use agents vs models:
- Key differences
- Use case comparisons
- Decision flowchart
- Code examples

---

## 🚀 Quick Start

### Prerequisites

1. **Install AI Toolkit:**
   ```bash
   pip install -e .
   ```

2. **Configure API Keys:**
   Create `.env` file in project root:
   ```env
   DEEPSEEK_API_KEY=your_key_here
   QWEN_API_KEY=your_key_here
   GLM_API_KEY=your_key_here
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Run Your First Example

```bash
# Start with the simple agent basics
python examples/2.simple_agent_basics.py
```

You'll see output demonstrating:
- Model creation
- Message types
- Tool definitions
- Memory usage
- Agent execution
- Structured output

---

## 📋 Example Comparison

| Example | Level | Topics | Use Case |
|---------|-------|--------|----------|
| `1.model_access_methods_guide.py` | Beginner | Model access patterns | Learn different ways to use models |
| `2.simple_agent_basics.py` | Beginner | All fundamentals | **Start here** - Complete introduction |
| `3.advanced_agent_patterns.py` | Advanced | Production patterns | Build production-ready agents |

---

## 🎯 What You'll Learn

### Core Concepts

1. **Models**
   - Creating and configuring chat models
   - Temperature and token settings
   - Provider selection (DeepSeek, Qwen, GLM)

2. **Messages**
   - SystemMessage for agent instructions
   - HumanMessage for user input
   - AIMessage for model output
   - Conversation structure

3. **Tools**
   - Defining tools with @tool decorator
   - Writing effective docstrings
   - Type hints and validation
   - Error handling

4. **Memory**
   - Checkpointers for conversation history
   - thread_id for conversation isolation
   - MemorySaver, SqliteSaver, RedisSaver
   - Multi-user applications

5. **Agents**
   - ReAct pattern (Reason + Act)
   - create_agent function
   - Tool selection and execution
   - Multi-step reasoning

6. **Structured Output**
   - Pydantic schemas
   - JSON validation
   - Type-safe responses
   - API integrations

### Advanced Patterns

7. **Dynamic Model Selection**
   - Task complexity analysis
   - Automatic model switching
   - Cost optimization

8. **Dynamic Prompts**
   - Context-aware prompt generation
   - Template systems
   - Prompt optimization

9. **Production Best Practices**
   - Error handling
   - Retry logic
   - Logging and monitoring
   - Performance optimization

---

## 🔧 Common Patterns

### Pattern 1: Simple Question-Answer
```python
# No tools, just model
result = model.invoke([
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is Python?")
])
```

### Pattern 2: Agent with Tools
```python
# Agent decides when to use tools
result = agent.invoke(
    {"messages": [HumanMessage(content="Calculate 123 * 456")]},
    config={"configurable": {"thread_id": "conversation-1"}}
)
```

### Pattern 3: Multi-turn Conversation
```python
# Same thread_id = shared memory
config = {"configurable": {"thread_id": "user-1"}}

result1 = agent.invoke({"messages": [HumanMessage(content="Hello")]}, config)
result2 = agent.invoke({"messages": [HumanMessage(content="Remember me?")]}, config)
```

---

## 🐛 Troubleshooting

### Agent not using tools?
- Check tool docstrings are clear and descriptive
- Mention tools in system prompt
- Verify tools are passed to `create_agent()`

### Memory not working?
- Use same `thread_id` across invocations
- Pass `config` to `agent.invoke()`
- Provide checkpointer to `create_agent()`

### Import errors?
```python
# Correct imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
```

---

## 📚 Official Documentation

- **LangChain Models**: https://docs.langchain.com/oss/python/langchain/models
- **LangChain Messages**: https://docs.langchain.com/oss/python/langchain/messages
- **LangChain Tools**: https://docs.langchain.com/oss/python/langchain/tools
- **LangChain Memory**: https://docs.langchain.com/oss/python/langchain/short-term-memory
- **LangChain Agents**: https://docs.langchain.com/oss/python/langchain/agents
- **Structured Output**: https://docs.langchain.com/oss/python/langchain/structured-output

---

## 💡 Next Steps

After completing these examples:

1. **Customize Tools** - Create your own tools for specific tasks
2. **Experiment with Prompts** - Try different system prompts
3. **Test Memory** - Use different thread_ids for conversation isolation
4. **Define Schemas** - Create your own Pydantic schemas
5. **Build Your Agent** - Combine concepts to build your own application

---

## 🤝 Contributing

Found an issue or have a suggestion? Please open an issue or submit a pull request!

---

**Happy Learning! 🎓**

Start with `2.simple_agent_basics.py` and work your way through the examples. Each example builds on the previous one, gradually introducing more advanced concepts.
