"""
RAG Priority Pattern with Source Tracking - Document First, AI Fallback

This example demonstrates a priority-based RAG pattern:
1. Try to answer from document first (document priority)
2. If document has no relevant information, fallback to AI model
3. Track and mark the source of each answer (document vs ai_model)

Key Features:
- Source tracking: Every answer is marked with its source
- Priority logic: Document first, then AI model fallback
- Fallback detection: Automatically detects when document has no answer
- Clear display: Visual markers show answer source
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from ai_toolkit.models import ModelManager
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

print("=" * 80)
print("RAG Priority Pattern - Document First with AI Fallback")
print("=" * 80)

# ============================================================================
# Common Setup: Load Document, Split, Create Embeddings & Vector Store
# ============================================================================

print("\n[Setup] Loading document and creating vector store...")
document_path = "/Users/hzz/Downloads/big_data_route.md"

if not os.path.exists(document_path):
    raise FileNotFoundError(f"Document not found: {document_path}")

# Step 1: Load document
loader = TextLoader(document_path, encoding='utf-8')
docs = loader.load()
print(f"✓ Loaded {len(docs)} document(s)")
print(f"✓ Total characters: {len(docs[0].page_content)}")

# Step 2: Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", "。", "，", " ", ""]  # Chinese-friendly separators
)
all_splits = splitter.split_documents(docs)
print(f"✓ Split document into {len(all_splits)} chunks")

# Step 3: Initialize embeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.environ.get('QWEN_API_KEY')
)
print("✓ Qwen embeddings initialized")

# Step 4: Create vector store
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
print(f"✓ Vector store created with {len(all_splits)} chunks")

# Step 5: Initialize chat model
manager = ModelManager()
model = manager.create_model(
    api_key=os.environ.get('QWEN_API_KEY'),
    provider="qwen",
    model="qwen-turbo",
    temperature=0.7,
    max_tokens=2000
)
print("✓ Qwen chat model initialized")

# ============================================================================
# Helper Function: Document Retrieval (Reused from rag_05)
# ============================================================================

def retrieve_from_document(query: str, k: int = 3, min_similarity: float = None) -> dict:
    """
    Retrieve answer from document only.
    
    Args:
        query: User's question
        k: Number of chunks to retrieve
        min_similarity: Optional minimum similarity score threshold
        
    Returns:
        dict with answer, retrieved_docs, and context
    """
    # Retrieve documents with similarity scores if threshold is provided
    if min_similarity is not None:
        try:
            if hasattr(vector_store, 'similarity_search_with_score'):
                docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
                relevant_docs = [
                    doc for doc, score in docs_with_scores 
                    if score >= min_similarity
                ]
                if not relevant_docs:
                    return {
                        "answer": "无相关文档信息",
                        "retrieved_docs": [],
                        "context_used": "",
                        "num_chunks": 0,
                        "similarity_scores": [score for _, score in docs_with_scores]
                    }
                retrieved_docs = relevant_docs
            else:
                retrieved_docs = vector_store.similarity_search(query, k=k)
        except Exception as e:
            retrieved_docs = vector_store.similarity_search(query, k=k)
    else:
        retrieved_docs = vector_store.similarity_search(query, k=k)
    
    # Check if no documents retrieved
    if not retrieved_docs:
        return {
            "answer": "无相关文档信息",
            "retrieved_docs": [],
            "context_used": "",
            "num_chunks": 0
        }
    
    # Build context from retrieved documents
    context = "\n\n".join(
        f"[Document Chunk {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )
    
    # Create strict prompt that ONLY uses context
    prompt = f"""You are a document-based Q&A assistant. Answer the question using ONLY the information provided in the document context below.

CRITICAL RULES:
1. You MUST answer using ONLY information from the provided context
2. You MUST NOT use any knowledge from your training data
3. If the context doesn't contain the answer, say: "无相关文档信息" (No relevant document information)
4. If the context is partially relevant, answer based on what IS in the context
5. Cite which document chunks you used in your answer

Document Context:
{context}

Question: {query}

Answer (based ONLY on the document context above):"""

    # Call model with the prompt
    response = model.invoke([HumanMessage(content=prompt)])
    
    return {
        "answer": response.content,
        "retrieved_docs": retrieved_docs,
        "context_used": context,
        "num_chunks": len(retrieved_docs)
    }


# ============================================================================
# Pattern: Document-First Priority with AI Fallback
# ============================================================================

def answer_with_priority(query: str, k: int = 3, min_similarity: float = None, 
                        use_ai_fallback: bool = True) -> dict:
    """
    Priority Pattern: Document first, then AI model fallback.
    
    This pattern implements priority logic:
    1. Try to answer from document first (PRIORITY)
    2. If document has no relevant information, fallback to AI model
    3. Mark the source of the answer (document or ai_model)
    
    Args:
        query: User's question
        k: Number of chunks to retrieve
        min_similarity: Optional minimum similarity score threshold (0.0-1.0)
        use_ai_fallback: Whether to use AI model when document has no answer
        
    Returns:
        dict with answer, source, retrieved_docs, and context
        - source: "document" or "ai_model" (marks where answer came from)
        - answer: The answer from document or AI model
        - fallback_used: Whether AI fallback was used
    """
    # Step 1: Try document retrieval first (PRIORITY)
    doc_result = retrieve_from_document(query, k=k, min_similarity=min_similarity)
    
    # Step 2: Check if document has relevant information
    # If answer is "无相关文档信息" or empty, document doesn't have answer
    doc_answer = doc_result.get("answer", "").strip()
    has_document_answer = (
        doc_answer and 
        doc_answer != "无相关文档信息" and
        doc_result.get("num_chunks", 0) > 0
    )
    
    if has_document_answer:
        # Document has the answer - return with document source
        return {
            "answer": doc_answer,
            "source": "document",
            "retrieved_docs": doc_result.get("retrieved_docs", []),
            "context_used": doc_result.get("context_used", ""),
            "num_chunks": doc_result.get("num_chunks", 0),
            "similarity_scores": doc_result.get("similarity_scores"),
            "fallback_used": False,
            "document_attempted": True
        }
    
    # Step 3: Document doesn't have answer
    # Return document empty message if fallback is disabled
    if not use_ai_fallback:
        return {
            "answer": "无相关文档信息",
            "source": "document",
            "retrieved_docs": [],
            "context_used": "",
            "num_chunks": 0,
            "fallback_used": False,
            "document_attempted": True
        }
    
    # Step 4: Fallback to AI model (document had no answer)
    ai_prompt = f"""You are a helpful AI assistant. Answer the following question based on your knowledge.

Question: {query}

Please provide a clear and helpful answer. If you're not certain about something, you can say so."""

    ai_response = model.invoke([HumanMessage(content=ai_prompt)])
    
    return {
        "answer": ai_response.content,
        "source": "ai_model",
        "retrieved_docs": doc_result.get("retrieved_docs", []),
        "context_used": "",
        "num_chunks": doc_result.get("num_chunks", 0),
        "fallback_used": True,
        "document_attempted": True,
        "document_answer": doc_answer if doc_answer else "无相关文档信息"
    }


# ============================================================================
# Enhanced Display Function with Source Marking
# ============================================================================

def display_answer_with_source(result: dict):
    """
    Display answer with clear source marking and visual indicators.
    
    Args:
        result: Result dict from answer_with_priority() or similar function
    """
    source = result.get("source", "unknown")
    answer = result.get("answer", "")
    fallback = result.get("fallback_used", False)
    num_chunks = result.get("num_chunks", 0)
    
    # Create visual markers based on source
    if source == "document":
        marker = "📄 [DOCUMENT]"
        indicator = "✓"
        source_desc = "Answer from document"
    elif source == "ai_model":
        marker = "🤖 [AI MODEL]"
        indicator = "⚠"
        source_desc = "Answer from AI model (fallback)"
    else:
        marker = "❓ [UNKNOWN]"
        indicator = "?"
        source_desc = "Unknown source"
    
    print(f"\n{marker} {indicator} Source: {source.upper()}")
    print(f"   {source_desc}")
    
    if fallback:
        print(f"   ⚠ Fallback was used (document had no relevant information)")
        if result.get("document_attempted"):
            print(f"   📋 Document response: {result.get('document_answer', 'N/A')}")
    
    print(f"\nAnswer:\n{answer}")
    
    # Show document context if available
    if num_chunks > 0 and result.get("retrieved_docs"):
        print(f"\n[Document Context Used]")
        print(f"   Retrieved {num_chunks} chunks")
        for j, doc in enumerate(result.get("retrieved_docs", [])[:2], 1):
            print(f"   Chunk {j}: {doc.page_content[:100]}...")
    
    # Show similarity scores if available
    if result.get("similarity_scores"):
        scores = result["similarity_scores"]
        print(f"\n[Similarity Scores]")
        print(f"   {scores}")


# ============================================================================
# Test Questions
# ============================================================================

test_questions = [
    (
        "请详细比较Hadoop、Spark、Kafka和Flink这四个大数据框架的核心定位、"
        "处理模型、延迟特点和主要使用场景。它们如何协同工作？",
        "Comprehensive comparison - should use DOCUMENT"
    ),
    (
        "Flink相比Spark Streaming有什么优势？为什么Flink的延迟可以达到毫秒级？"
        "请解释Flink的事件时间处理机制。",
        "Flink advantages - should use DOCUMENT"
    ),
    (
        "如果我已经掌握了Kafka和Flink，下一步应该学习什么？"
        "请根据文档中的学习路径建议，给出详细的学习方向。",
        "Learning path - should use DOCUMENT"
    ),
    (
        "What is the capital of France?",
        "General knowledge - should use AI MODEL (fallback)"
    ),
    (
        "Python编程语言的基本语法是什么？",
        "General programming question - should use AI MODEL (fallback)"
    ),
    (
        "What is the weather like today?",
        "Current events - should use AI MODEL (fallback)"
    ),
]

# ============================================================================
# Testing: Priority Pattern with Source Tracking
# ============================================================================

print("\n" + "=" * 80)
print("Testing: Priority Pattern with Source Tracking")
print("=" * 80)
print("\nThis pattern tries document first, then falls back to AI model.\n")
print("Source markers:")
print("  📄 [DOCUMENT] = Answer from document")
print("  🤖 [AI MODEL] = Answer from AI model (fallback)")
print()

for i, (query, description) in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"[Test {i}] {description}")
    print(f"Query: {query}")
    print("-" * 80)
    
    try:
        # Test with fallback enabled
        result = answer_with_priority(query, k=3, use_ai_fallback=True)
        display_answer_with_source(result)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Testing: Without Fallback (Document Only)
# ============================================================================

print("\n\n" + "=" * 80)
print("Testing: Document Only (No Fallback)")
print("=" * 80)
print("\nTesting with use_ai_fallback=False - should return '无相关文档信息'\n")

test_cases_no_fallback = [
    ("What is the capital of France?", "General knowledge question"),
    ("请详细比较Hadoop和Spark", "Document-related question"),
]

for i, (query, description) in enumerate(test_cases_no_fallback, 1):
    print(f"\n[Test {i}] {description}")
    print(f"Query: {query}")
    print("-" * 80)
    
    try:
        result = answer_with_priority(query, k=3, use_ai_fallback=False)
        display_answer_with_source(result)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("\nKey Features:")
print("  ✓ Priority Logic: Document first, then AI model fallback")
print("  ✓ Source Tracking: Every answer marked with source (document/ai_model)")
print("  ✓ Fallback Detection: Automatically detects when document has no answer")
print("  ✓ Visual Markers: Clear indicators show answer source")
print("  ✓ Configurable: Can disable fallback to return document-only responses")
print("\nReturn Dictionary Structure:")
print("  {")
print("    'answer': 'The actual answer',")
print("    'source': 'document' | 'ai_model',  # Source tracking")
print("    'retrieved_docs': [...],")
print("    'num_chunks': 3,")
print("    'fallback_used': True | False,  # Whether fallback was used")
print("    'document_attempted': True,  # Whether document was tried first")
print("    'document_answer': '...'  # What document returned")
print("  }")
print("\nUsage:")
print("  result = answer_with_priority(query, k=3, use_ai_fallback=True)")
print("  display_answer_with_source(result)")
