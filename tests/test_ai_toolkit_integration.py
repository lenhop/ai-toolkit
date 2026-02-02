"""
Comprehensive Integration Tests for AI Toolkit

Tests all major modules together to verify they work correctly:
- Models
- Agents
- Tools
- Memory
- Messages
- Streaming
- RAG

Based on examples: 11_model_base.py, 13_agent_base.py, 16_tool_base.py, 17_memory_base.py, 18_streaming_base.py, rag_04_agent_workflow.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

print("=" * 80)
print("AI Toolkit Comprehensive Integration Tests")
print("=" * 80)

# Test imports
issues = []

try:
    from ai_toolkit.models import ModelManager
    print("✓ ModelManager imported")
except ImportError as e:
    issues.append(f"ModelManager import: {e}")
    print(f"❌ ModelManager import failed: {e}")

try:
    from ai_toolkit.agents import (
        create_agent_with_tools,
        create_agent_with_memory,
        create_streaming_agent,
        create_structured_output_agent,
        create_dynamic_model_selector,
        create_tool_error_handler,
    )
    print("✓ Agent helpers imported")
except ImportError as e:
    issues.append(f"Agent helpers import: {e}")
    print(f"❌ Agent helpers import failed: {e}")

try:
    from ai_toolkit.tools import (
        create_search_tool,
        create_weather_tool,
        create_calculator_tool,
    )
    print("✓ Tool utilities imported")
except ImportError as e:
    issues.append(f"Tool utilities import: {e}")
    print(f"❌ Tool utilities import failed: {e}")

try:
    from ai_toolkit.memory import MemoryManager
    print("✓ MemoryManager imported")
except ImportError as e:
    issues.append(f"MemoryManager import: {e}")
    print(f"❌ MemoryManager import failed: {e}")

try:
    from ai_toolkit.messages import MessageBuilder
    print("✓ MessageBuilder imported")
except ImportError as e:
    issues.append(f"MessageBuilder import: {e}")
    print(f"❌ MessageBuilder import failed: {e}")

try:
    from ai_toolkit.streaming import StreamHandler
    print("✓ StreamHandler imported")
except ImportError as e:
    issues.append(f"StreamHandler import: {e}")
    print(f"❌ StreamHandler import failed: {e}")

try:
    from ai_toolkit.rag import (
        load_web_document,
        split_document_recursive,
        create_vector_store,
        create_rag_agent,
        retrieve_document_only,
        retrieve_with_priority,
    )
    print("✓ RAG utilities imported")
except ImportError as e:
    issues.append(f"RAG utilities import: {e}")
    print(f"❌ RAG utilities import failed: {e}")


def test_end_to_end_workflow():
    """Test a complete end-to-end workflow."""
    print("\n" + "=" * 80)
    print("Test: End-to-End Workflow")
    print("=" * 80)
    
    if issues:
        print(f"⚠ Skipping test - {len(issues)} import issue(s) found")
        return
    
    try:
        # 1. Create model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            provider="deepseek",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2000
        )
        print("✓ Step 1: Model created")
        
        # 2. Create tools
        search = create_search_tool()
        weather = create_weather_tool()
        calc = create_calculator_tool()
        tools = [search, weather, calc]
        print("✓ Step 2: Tools created")
        
        # 3. Create agent with memory
        agent = create_agent_with_memory(
            model=model,
            tools=tools,
            checkpointer_type="inmemory",
            system_prompt="You are a helpful assistant."
        )
        print("✓ Step 3: Agent with memory created")
        
        # 4. Use agent
        config = {"configurable": {"thread_id": "integration-test"}}
        result = agent.invoke({
            "messages": [{"role": "user", "content": "What is 25 * 4?"}]
        }, config=config)
        
        print("✓ Step 4: Agent invoked")
        print(f"✓ Response: {str(result['messages'][-1].content)[:150]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_rag_workflow():
    """Test complete RAG workflow."""
    print("\n" + "=" * 80)
    print("Test: RAG Workflow")
    print("=" * 80)
    
    if issues:
        print(f"⚠ Skipping test - {len(issues)} import issue(s) found")
        return
    
    try:
        # Check if test document exists
        test_file = "/Users/hzz/Downloads/big_data_route.md"
        if not os.path.exists(test_file):
            print(f"⚠ Test file not found: {test_file}")
            print("  Skipping RAG workflow test")
            return
        
        # 1. Load document
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(test_file, encoding='utf-8')
        docs = loader.load()
        print(f"✓ Step 1: Document loaded ({len(docs)} docs)")
        
        # 2. Split document
        chunks = split_document_recursive(docs, chunk_size=1000, chunk_overlap=200)
        print(f"✓ Step 2: Document split ({len(chunks)} chunks)")
        
        # 3. Create embeddings and vector store
        from langchain_community.embeddings import DashScopeEmbeddings
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.environ.get('QWEN_API_KEY')
        )
        vector_store = create_vector_store(chunks, embeddings)
        print("✓ Step 3: Vector store created")
        
        # 4. Create model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        print("✓ Step 4: Model created")
        
        # 5. Test document-only retrieval
        result = retrieve_document_only(
            vector_store=vector_store,
            model=model,
            query="请详细比较Hadoop和Spark",
            k=3
        )
        print("✓ Step 5: Document-only retrieval completed")
        print(f"✓ Retrieved {result['num_chunks']} chunks")
        print(f"✓ Answer preview: {result['answer'][:150]}...")
        
        # 6. Test priority retrieval
        result2 = retrieve_with_priority(
            vector_store=vector_store,
            model=model,
            query="What is the capital of France?",
            k=3,
            use_ai_fallback=True
        )
        print("✓ Step 6: Priority retrieval completed")
        print(f"✓ Source: {result2['source']}")
        print(f"✓ Fallback used: {result2['fallback_used']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# Run tests
if __name__ == "__main__":
    test_end_to_end_workflow()
    test_rag_workflow()
    
    print("\n" + "=" * 80)
    print("Integration Tests Summary")
    print("=" * 80)
    
    if issues:
        print(f"\n❌ Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ All imports successful")
    
    print("\n" + "=" * 80)
    print("Integration Tests Completed")
    print("=" * 80)
