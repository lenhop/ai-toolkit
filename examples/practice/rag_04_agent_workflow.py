"""
Complete RAG Workflow Example - Big Data Frameworks Q&A

This example demonstrates a complete RAG (Retrieval Augmented Generation) workflow:
1. Load markdown document
2. Split into chunks
3. Create embeddings using Qwen
4. Store in vector database
5. Create agent with retrieval tool
6. Answer questions based on the document
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from ai_toolkit.models import ModelManager
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

print("=" * 80)
print("Complete RAG Workflow - Big Data Frameworks Q&A")
print("=" * 80)

# ============================================================================
# Step 1: Load Document
# ============================================================================
print("\n[Step 1] Loading document...")
document_path = "/Users/hzz/Downloads/big_data_route.md"

if not os.path.exists(document_path):
    raise FileNotFoundError(f"Document not found: {document_path}")

# Load markdown file
loader = TextLoader(document_path, encoding='utf-8')
docs = loader.load()
print(f"✓ Loaded {len(docs)} document(s)")
print(f"✓ Total characters: {len(docs[0].page_content)}")
print(f"✓ Source: {docs[0].metadata.get('source', 'N/A')}")

# ============================================================================
# Step 2: Split Document into Chunks
# ============================================================================
print("\n[Step 2] Splitting document into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Each chunk has ~1000 characters
    chunk_overlap=200,    # 200 characters overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", "。", "，", " ", ""]  # Chinese-friendly separators
)

all_splits = splitter.split_documents(docs)
print(f"✓ Split document into {len(all_splits)} chunks")
print(f"✓ Average chunk size: {sum(len(chunk.page_content) for chunk in all_splits) // len(all_splits)} characters")
print(f"\nFirst chunk preview:")
print(f"  Length: {len(all_splits[0].page_content)} characters")
print(f"  Content: {all_splits[0].page_content[:150]}...")

# ============================================================================
# Step 3: Initialize Embeddings (Qwen)
# ============================================================================
print("\n[Step 3] Initializing Qwen embeddings...")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # Qwen embedding model via DashScope
    dashscope_api_key=os.environ.get('QWEN_API_KEY')
)
print("✓ Qwen embeddings initialized")

# ============================================================================
# Step 4: Create Vector Store and Add Documents
# ============================================================================
print("\n[Step 4] Creating vector store and adding documents...")
vector_store = InMemoryVectorStore(embeddings)

# Add all document chunks to vector store
print("  Adding chunks to vector store...", end=" ", flush=True)
vector_store.add_documents(all_splits)
print("✓ Done")
print(f"✓ Vector store contains {len(all_splits)} document chunks")

# ============================================================================
# Step 5: Initialize Chat Model (Qwen)
# ============================================================================
print("\n[Step 5] Initializing Qwen chat model...")
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
# Step 6: Create Retrieval Tool
# ============================================================================
print("\n[Step 6] Creating retrieval tool...")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> tuple:
    """
    Retrieve relevant information from the big data frameworks document to help answer a query.
    
    This tool searches the vector store for the most relevant document chunks related to the query
    and returns formatted context for the LLM to use in generating answers.
    
    Args:
        query: The user's question or query string
        
    Returns:
        A tuple containing:
        - serialized: Formatted text with source and content for the LLM
        - retrieved_docs: Original document objects for programmatic use
    """
    # Retrieve top-k most relevant documents (increase k for more context)
    retrieved_docs = vector_store.similarity_search(query, k=3)
    
    # Format retrieved documents for LLM consumption
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs

print("✓ Retrieval tool created")

# ============================================================================
# Step 7: Create RAG Agent
# ============================================================================
print("\n[Step 7] Creating RAG agent...")
tools = [retrieve_context]

system_prompt = (
    "You are a helpful assistant specialized in big data frameworks and technologies. "
    "You have access to a comprehensive document about Hadoop, Spark, Kafka, and Flink. "
    "When answering questions:\n"
    "1. Use the retrieve_context tool to find relevant information from the document\n"
    "2. Base your answers on the retrieved context\n"
    "3. Provide accurate, detailed answers with specific examples when possible\n"
    "4. If the document doesn't contain the answer, say so clearly\n"
    "5. You can answer in Chinese or English based on the user's language preference"
)

agent = create_agent(model, tools, system_prompt=system_prompt)
print("✓ RAG agent created")

# ============================================================================
# Step 8: Test with Relevant Questions
# ============================================================================
print("\n" + "=" * 80)
print("Testing RAG System")
print("=" * 80)

# Question 1: Core comparison question
query1 = (
    "请详细比较Hadoop、Spark、Kafka和Flink这四个大数据框架的核心定位、"
    "处理模型、延迟特点和主要使用场景。它们如何协同工作？"
)

print(f"\n[Question 1] {query1}")
print("-" * 80)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query1}]},
    stream_mode="values",
):
    if "messages" in event and event["messages"]:
        last_message = event["messages"][-1]
        if hasattr(last_message, 'pretty_print'):
            last_message.pretty_print()
        elif hasattr(last_message, 'content'):
            print(last_message.content)

# Question 2: Specific framework question
query2 = (
    "Flink相比Spark Streaming有什么优势？为什么Flink的延迟可以达到毫秒级？"
    "请解释Flink的事件时间处理机制。"
)

print(f"\n\n[Question 2] {query2}")
print("-" * 80)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query2}]},
    stream_mode="values",
):
    if "messages" in event and event["messages"]:
        last_message = event["messages"][-1]
        if hasattr(last_message, 'pretty_print'):
            last_message.pretty_print()
        elif hasattr(last_message, 'content'):
            print(last_message.content)

# Question 3: Learning path question
query3 = (
    "如果我已经掌握了Kafka和Flink，下一步应该学习什么？"
    "请根据文档中的学习路径建议，给出详细的学习方向。"
)

print(f"\n\n[Question 3] {query3}")
print("-" * 80)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query3}]},
    stream_mode="values",
):
    if "messages" in event and event["messages"]:
        last_message = event["messages"][-1]
        if hasattr(last_message, 'pretty_print'):
            last_message.pretty_print()
        elif hasattr(last_message, 'content'):
            print(last_message.content)

print("\n" + "=" * 80)
print("RAG Workflow Completed!")
print("=" * 80)
print("\nSummary:")
print(f"  - Document loaded: {document_path}")
print(f"  - Total chunks: {len(all_splits)}")
print(f"  - Embedding model: Qwen text-embedding-v3")
print(f"  - Chat model: Qwen qwen-turbo")
print(f"  - Vector store: InMemoryVectorStore")
print(f"  - Questions tested: 3")
