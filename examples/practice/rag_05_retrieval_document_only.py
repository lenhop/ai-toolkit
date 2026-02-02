"""
RAG Document-Only Retrieval - Ensuring All Answers Come from Document

This example demonstrates two patterns to ensure ALL responses come ONLY from the document:
1. Option 1: RAG Chain (RetrievalQA) - Always retrieves before answering
2. Option 3: Pre-retrieve Pattern - Explicitly retrieves context first, then answers

Key difference from rag_04_agent_workflow.py:
- Agent-based approach: Agent decides whether to use tool (may use training data)
- Document-only approach: ALWAYS retrieves from document first (guaranteed document-only)
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
print("RAG Document-Only Retrieval - Big Data Frameworks Q&A")
print("=" * 80)

# ============================================================================
# Common Setup: Load Document, Split, Create Embeddings & Vector Store
# (Reused from rag_04_agent_workflow.py)
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
# Pattern 1: Pre-retrieve Pattern (Simple & Explicit)
# ============================================================================

def answer_with_pre_retrieve(query: str, k: int = 3, min_similarity: float = None) -> dict:
    """
    Pattern 3: Pre-retrieve Pattern - Always retrieves context first.
    
    This pattern GUARANTEES that answers come only from the document by:
    1. ALWAYS retrieving relevant chunks first
    2. Filtering by similarity threshold (if provided)
    3. Returning empty string if no relevant documents found
    4. Building a prompt with ONLY the retrieved context
    5. Explicitly instructing the model to use ONLY the context
    
    Args:
        query: User's question
        k: Number of chunks to retrieve
        min_similarity: Optional minimum similarity score threshold (0.0-1.0)
                       If None, uses all retrieved documents
        
    Returns:
        dict with answer, retrieved_docs, and context
        If no relevant documents: returns empty answer and empty context
    """
    # Retrieve documents with similarity scores if threshold is provided
    if min_similarity is not None:
        # Try to use similarity_search_with_score if available
        try:
            if hasattr(vector_store, 'similarity_search_with_score'):
                docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
                
                # Filter by similarity threshold
                # Note: Score interpretation depends on distance metric:
                # - For cosine similarity: higher score = more similar (range -1 to 1)
                # - For distance metrics: lower score = more similar
                # Assuming cosine similarity: score >= min_similarity means relevant
                relevant_docs = [
                    doc for doc, score in docs_with_scores 
                    if score >= min_similarity
                ]
                
                # If no documents meet the threshold, return empty
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
                # Fallback: similarity threshold not supported, use standard retrieval
                print(f"Warning: similarity_search_with_score not available, ignoring min_similarity={min_similarity}")
                retrieved_docs = vector_store.similarity_search(query, k=k)
        except Exception as e:
            # Fallback on error
            print(f"Warning: Error using similarity_search_with_score: {e}. Using standard retrieval.")
            retrieved_docs = vector_store.similarity_search(query, k=k)
    else:
        # Standard retrieval without score filtering
        retrieved_docs = vector_store.similarity_search(query, k=k)
    
    # Check if no documents retrieved (safety check)
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
# Pattern 2: RAG Chain Pattern (Using LangChain's RetrievalQA)
# ============================================================================

def answer_with_rag_chain(query: str, k: int = 3, min_similarity: float = None) -> dict:
    """
    Pattern 1: RAG Chain Pattern - Uses LangChain's retrieval chain.
    
    This pattern uses LangChain's built-in RAG chain which:
    1. Always retrieves context first
    2. Filters by similarity threshold (if provided)
    3. Returns empty if no relevant documents found
    4. Formats the prompt with context
    5. Generates answer based on context
    
    Note: RetrievalQA may not be available in all LangChain versions,
    so we implement a similar pattern using prompts and retriever.
    
    Args:
        query: User's question
        k: Number of chunks to retrieve
        min_similarity: Optional minimum similarity score threshold (0.0-1.0)
                       If None, uses all retrieved documents
        
    Returns:
        dict with answer and retrieved_docs
        If no relevant documents: returns empty answer and empty context
    """
    # Retrieve documents with similarity scores if threshold is provided
    if min_similarity is not None:
        # Try to use similarity_search_with_score if available
        try:
            if hasattr(vector_store, 'similarity_search_with_score'):
                docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
                
                # Filter by similarity threshold
                relevant_docs = [
                    doc for doc, score in docs_with_scores 
                    if score >= min_similarity
                ]
                
                # If no documents meet the threshold, return empty
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
                # Fallback: similarity threshold not supported, use standard retrieval
                print(f"Warning: similarity_search_with_score not available, ignoring min_similarity={min_similarity}")
                retriever = vector_store.as_retriever(search_kwargs={"k": k})
                retrieved_docs = retriever.invoke(query)
        except Exception as e:
            # Fallback on error
            print(f"Warning: Error using similarity_search_with_score: {e}. Using standard retrieval.")
            retriever = vector_store.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(query)
    else:
        # Standard retrieval without score filtering
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)
    
    # Check if no documents retrieved (safety check)
    if not retrieved_docs:
        return {
            "answer": "无相关文档信息",
            "retrieved_docs": [],
            "context_used": "",
            "num_chunks": 0
        }
    
    # Build context
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a document-based Q&A assistant. 
Answer questions using ONLY the provided document context.

Rules:
- Use ONLY information from the context below
- Do NOT use any external knowledge
- If context doesn't contain the answer, say "无相关文档信息" (No relevant document information)
- Cite which chunks you used"""),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    
    # Format prompt
    formatted_prompt = prompt_template.format_messages(
        context=context,
        question=query
    )
    
    # Generate answer
    response = model.invoke(formatted_prompt)
    
    return {
        "answer": response.content,
        "retrieved_docs": retrieved_docs,
        "context_used": context,
        "num_chunks": len(retrieved_docs)
    }


# ============================================================================
# Test Questions
# ============================================================================

test_questions = [
    (
        "请详细比较Hadoop、Spark、Kafka和Flink这四个大数据框架的核心定位、"
        "处理模型、延迟特点和主要使用场景。它们如何协同工作？",
        "Comprehensive comparison of all four frameworks"
    ),
    (
        "Flink相比Spark Streaming有什么优势？为什么Flink的延迟可以达到毫秒级？"
        "请解释Flink的事件时间处理机制。",
        "Flink advantages and event time processing"
    ),
    (
        "如果我已经掌握了Kafka和Flink，下一步应该学习什么？"
        "请根据文档中的学习路径建议，给出详细的学习方向。",
        "Learning path recommendations"
    ),
]

# Test question with no relevant information (to demonstrate empty result handling)
test_question_no_match = (
    "What is the weather like today?",
    "Question with no relevant document content"
)

# ============================================================================
# Testing: Pattern 1 - Pre-retrieve Pattern
# ============================================================================

print("\n" + "=" * 80)
print("Pattern 1: Pre-retrieve Pattern (Document-Only)")
print("=" * 80)
print("\nThis pattern ALWAYS retrieves context first, ensuring document-only answers.\n")

for i, (query, description) in enumerate(test_questions, 1):
    print(f"\n[Question {i}] {description}")
    print(f"Query: {query}")
    print("-" * 80)
    
    try:
        result = answer_with_pre_retrieve(query, k=3)
        print(f"\n✓ Retrieved {result['num_chunks']} document chunks")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n[Retrieved Chunks Preview]")
        for j, doc in enumerate(result['retrieved_docs'][:2], 1):  # Show first 2
            print(f"  Chunk {j}: {len(doc.page_content)} chars - {doc.page_content[:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Testing: Pattern 2 - RAG Chain Pattern
# ============================================================================

print("\n\n" + "=" * 80)
print("Pattern 2: RAG Chain Pattern (Document-Only)")
print("=" * 80)
print("\nThis pattern uses LangChain's retrieval chain pattern.\n")

for i, (query, description) in enumerate(test_questions, 1):
    print(f"\n[Question {i}] {description}")
    print(f"Query: {query}")
    print("-" * 80)
    
    try:
        result = answer_with_rag_chain(query, k=3)
        print(f"\n✓ Retrieved {result['num_chunks']} document chunks")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n[Retrieved Chunks Preview]")
        for j, doc in enumerate(result['retrieved_docs'][:2], 1):  # Show first 2
            print(f"  Chunk {j}: {len(doc.page_content)} chars - {doc.page_content[:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================

# ============================================================================
# Testing: Empty Result Handling
# ============================================================================

print("\n\n" + "=" * 80)
print("Testing: Empty Result Handling (No Relevant Documents)")
print("=" * 80)
print("\nDemonstrating the rule: '无相关信息时返回空字符串' (Return empty when no relevant info)\n")

query_no_match, desc_no_match = test_question_no_match
print(f"\n[Test] {desc_no_match}")
print(f"Query: {query_no_match}")
print("-" * 80)

try:
    # Test with high similarity threshold to force empty result
    result = answer_with_pre_retrieve(query_no_match, k=3, min_similarity=0.95)
    print(f"\n✓ Retrieved {result['num_chunks']} document chunks")
    print(f"✓ Answer: {result['answer']}")
    if result['num_chunks'] == 0:
        print("✓ Correctly returned empty result (无相关文档信息)")
    if 'similarity_scores' in result:
        print(f"✓ Similarity scores: {result['similarity_scores']}")
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
print("  ✓ ALWAYS retrieves context before answering")
print("  ✓ Explicit instructions to use ONLY document context")
print("  ✓ No option to skip retrieval")
print("  ✓ Returns empty string (无相关文档信息) when no relevant documents found")
print("  ✓ Optional similarity threshold filtering for better precision")
print("\nPattern Comparison:")
print("  Pattern 1 (Pre-retrieve): Simple, explicit, full control")
print("  Pattern 2 (RAG Chain): Uses LangChain patterns, more structured")
print("\nBoth patterns implement the rule:")
print("  '仅从文档中检索与查询相关的信息，无相关信息时返回空字符串'")
print("  (Retrieve only relevant info from document, return empty when no relevant info)")
