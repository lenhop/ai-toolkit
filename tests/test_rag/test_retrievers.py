"""
Test RAG Retrievers

Tests for ai_toolkit.rag.retrievers module:
- create_vector_store
- create_retrieval_tool
- create_rag_agent
- retrieve_document_only
- retrieve_with_priority

Based on examples: rag_04_agent_workflow.py, rag_05_retrieval_document_only.py, rag_06_priority_fallback.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

# Test imports
try:
    from ai_toolkit.models import ModelManager
    from ai_toolkit.rag import (
        load_web_document,
        split_document_recursive,
        create_vector_store,
        create_retrieval_tool,
        create_rag_agent,
        retrieve_document_only,
        retrieve_with_priority,
    )
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_core.messages import HumanMessage
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_create_vector_store():
    """Test create_vector_store function."""
    print("\n" + "=" * 80)
    print("Test 1: create_vector_store")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Load and split documents
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        docs = load_web_document(url)
        chunks = split_document_recursive(docs, chunk_size=1000, chunk_overlap=200)
        
        print(f"✓ Prepared {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.environ.get('QWEN_API_KEY')
        )
        
        print("✓ Embeddings created")
        
        # Create vector store
        vector_store = create_vector_store(chunks, embeddings, store_type="inmemory")
        
        print("✓ Vector store created")
        
        # Test: Search
        results = vector_store.similarity_search("What is an agent?", k=3)
        print(f"✓ Search returned {len(results)} results")
        print(f"✓ First result: {results[0].page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_create_retrieval_tool():
    """Test create_retrieval_tool function."""
    print("\n" + "=" * 80)
    print("Test 2: create_retrieval_tool")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup vector store
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        docs = load_web_document(url)
        chunks = split_document_recursive(docs, chunk_size=1000, chunk_overlap=200)
        
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.environ.get('QWEN_API_KEY')
        )
        
        vector_store = create_vector_store(chunks, embeddings)
        
        # Create retrieval tool
        retrieval_tool = create_retrieval_tool(
            vector_store,
            name="retrieve_context",
            description="Retrieve relevant information from the document.",
            k=3
        )
        
        print("✓ Retrieval tool created")
        
        # Test: Use in agent
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        from langchain.agents import create_agent
        agent = create_agent(model, tools=[retrieval_tool])
        
        result = agent.invoke({
            "messages": [HumanMessage(content="What is an agent?")]
        })
        
        print("✓ Retrieval tool used in agent")
        print(f"✓ Response preview: {str(result['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_create_rag_agent():
    """Test create_rag_agent function."""
    print("\n" + "=" * 80)
    print("Test 3: create_rag_agent")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup: Load, split, create vector store
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        docs = load_web_document(url)
        chunks = split_document_recursive(docs, chunk_size=1000, chunk_overlap=200)
        
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.environ.get('QWEN_API_KEY')
        )
        
        vector_store = create_vector_store(chunks, embeddings)
        
        # Create model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        # Create RAG agent
        agent = create_rag_agent(
            model=model,
            vector_store=vector_store,
            k=3
        )
        
        print("✓ RAG agent created")
        
        # Test: Ask question
        result = agent.invoke({
            "messages": [HumanMessage(content="What is an agent?")]
        })
        
        print("✓ RAG agent invoked")
        print(f"✓ Response preview: {str(result['messages'][-1].content)[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_retrieve_document_only():
    """Test retrieve_document_only function."""
    print("\n" + "=" * 80)
    print("Test 4: retrieve_document_only")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup: Load, split, create vector store
        test_file = "/Users/hzz/Downloads/big_data_route.md"
        if not os.path.exists(test_file):
            print(f"⚠ Test file not found: {test_file}")
            print("  Skipping test")
            return
        
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(test_file, encoding='utf-8')
        docs = loader.load()
        chunks = split_document_recursive(docs, chunk_size=1000, chunk_overlap=200)
        
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.environ.get('QWEN_API_KEY')
        )
        
        vector_store = create_vector_store(chunks, embeddings)
        
        # Create model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        # Test: Retrieve document only
        result = retrieve_document_only(
            vector_store=vector_store,
            model=model,
            query="请详细比较Hadoop和Spark",
            k=3
        )
        
        print("✓ Document-only retrieval completed")
        print(f"✓ Retrieved {result['num_chunks']} chunks")
        print(f"✓ Answer preview: {result['answer'][:200]}...")
        
        # Test: Query with no match
        result_no_match = retrieve_document_only(
            vector_store=vector_store,
            model=model,
            query="What is the weather like today?",
            k=3
        )
        
        print(f"✓ No-match query handled: {result_no_match['answer']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_retrieve_with_priority():
    """Test retrieve_with_priority function."""
    print("\n" + "=" * 80)
    print("Test 5: retrieve_with_priority")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Setup: Load, split, create vector store
        test_file = "/Users/hzz/Downloads/big_data_route.md"
        if not os.path.exists(test_file):
            print(f"⚠ Test file not found: {test_file}")
            print("  Skipping test")
            return
        
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(test_file, encoding='utf-8')
        docs = loader.load()
        chunks = split_document_recursive(docs, chunk_size=1000, chunk_overlap=200)
        
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=os.environ.get('QWEN_API_KEY')
        )
        
        vector_store = create_vector_store(chunks, embeddings)
        
        # Create model
        manager = ModelManager()
        model = manager.create_model(
            api_key=os.environ.get('QWEN_API_KEY'),
            provider="qwen",
            model="qwen-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        
        # Test: Query with document answer (should use document)
        result1 = retrieve_with_priority(
            vector_store=vector_store,
            model=model,
            query="请详细比较Hadoop和Spark",
            k=3,
            use_ai_fallback=True
        )
        
        print("✓ Priority retrieval completed")
        print(f"✓ Source: {result1['source']}")
        print(f"✓ Fallback used: {result1['fallback_used']}")
        print(f"✓ Answer preview: {result1['answer'][:200]}...")
        
        # Test: Query without document answer (should use AI fallback)
        result2 = retrieve_with_priority(
            vector_store=vector_store,
            model=model,
            query="What is the capital of France?",
            k=3,
            use_ai_fallback=True
        )
        
        print(f"✓ Fallback test - Source: {result2['source']}")
        print(f"✓ Fallback used: {result2['fallback_used']}")
        print(f"✓ Answer preview: {result2['answer'][:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing RAG Retrievers")
    print("=" * 80)
    
    test_create_vector_store()
    test_create_retrieval_tool()
    test_create_rag_agent()
    test_retrieve_document_only()
    test_retrieve_with_priority()
    
    print("\n" + "=" * 80)
    print("RAG Retriever Tests Completed")
    print("=" * 80)
