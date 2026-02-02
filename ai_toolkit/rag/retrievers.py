"""
Retrieval Utilities - RAG retrieval and agent helpers

This module provides utilities for creating RAG agents and retrieval tools.

Functions:
    - create_vector_store(): Create vector store with embeddings
    - create_retrieval_tool(): Create retrieval tool for agent
    - create_rag_agent(): Complete RAG agent factory
    - retrieve_with_priority(): Document-first with AI fallback
    - retrieve_document_only(): Document-only retrieval

Based on: examples/practice/rag_04_agent_workflow.py, rag_05_retrieval_document_only.py, rag_06_priority_fallback.py
LangChain Version: 1.0
Python Version: 3.11

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Optional, Any, Dict, Tuple
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain.tools import tool


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    store_type: str = "inmemory"
) -> VectorStore:
    """
    Create vector store with embeddings.
    
    Based on examples/practice/rag_04_agent_workflow.py
    
    Args:
        documents: List of document chunks to store
        embeddings: Embeddings model (e.g., DashScopeEmbeddings for Qwen)
        store_type: Type of vector store ("inmemory" or others)
    
    Returns:
        Vector store instance
    
    Example:
        >>> from langchain_community.embeddings import DashScopeEmbeddings
        >>> from ai_toolkit.rag import load_web_document, split_document_recursive
        >>> 
        >>> # Load and split documents
        >>> docs = load_web_document("https://example.com")
        >>> chunks = split_document_recursive(docs, chunk_size=1000)
        >>> 
        >>> # Create embeddings
        >>> embeddings = DashScopeEmbeddings(
        ...     model="text-embedding-v3",
        ...     dashscope_api_key="your-api-key"
        ... )
        >>> 
        >>> # Create vector store
        >>> vector_store = create_vector_store(chunks, embeddings)
        >>> 
        >>> # Search
        >>> results = vector_store.similarity_search("query", k=3)
    
    Note:
        - InMemory: Fast, lost on restart
        - For production, consider Chroma, Pinecone, or Weaviate
        - Embeddings model should match your use case (language, domain)
    """
    if store_type == "inmemory":
        from langchain_core.vectorstores import InMemoryVectorStore
        
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents)
        return vector_store
    else:
        raise ValueError(f"Unknown store_type: {store_type}. Currently only 'inmemory' is supported.")


def create_retrieval_tool(
    vector_store: VectorStore,
    name: str = "retrieve_context",
    description: str = "Retrieve relevant information from the document to help answer a query.",
    k: int = 3
):
    """
    Create a retrieval tool for RAG agent.
    
    Based on examples/practice/rag_04_agent_workflow.py
    
    Args:
        vector_store: Vector store to retrieve from
        name: Tool name (default: "retrieve_context")
        description: Tool description for the agent
        k: Number of documents to retrieve (default: 3)
    
    Returns:
        Retrieval tool
    
    Example:
        >>> from ai_toolkit.rag import create_vector_store, create_retrieval_tool
        >>> 
        >>> # Create vector store
        >>> vector_store = create_vector_store(chunks, embeddings)
        >>> 
        >>> # Create retrieval tool
        >>> retrieval_tool = create_retrieval_tool(
        ...     vector_store,
        ...     description="Search the knowledge base for relevant information.",
        ...     k=3
        ... )
        >>> 
        >>> # Use in agent
        >>> agent = create_agent(model, tools=[retrieval_tool])
    
    Note:
        - Returns formatted context with source and content
        - Also returns original documents as artifact
        - Agent can use context to answer questions
    """
    @tool(name, description=description, response_format="content_and_artifact")
    def retrieve_context(query: str) -> Tuple[str, List[Document]]:
        """
        Retrieve relevant information from the document.
        
        Args:
            query: The user's question or query string
            
        Returns:
            A tuple containing:
            - serialized: Formatted text with source and content for the LLM
            - retrieved_docs: Original document objects for programmatic use
        """
        # Retrieve top-k most relevant documents
        retrieved_docs = vector_store.similarity_search(query, k=k)
        
        # Format retrieved documents for LLM consumption
        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        
        return serialized, retrieved_docs
    
    return retrieve_context


def create_rag_agent(
    model: BaseChatModel,
    vector_store: VectorStore,
    system_prompt: Optional[str] = None,
    k: int = 3,
    **kwargs
) -> Any:
    """
    Create a complete RAG agent with retrieval tool.
    
    Based on examples/practice/rag_04_agent_workflow.py
    
    Args:
        model: LangChain chat model
        vector_store: Vector store with document chunks
        system_prompt: Optional system prompt (default: RAG-optimized prompt)
        k: Number of documents to retrieve (default: 3)
        **kwargs: Additional arguments passed to create_agent()
    
    Returns:
        RAG agent
    
    Example:
        >>> from ai_toolkit.models import ModelManager
        >>> from ai_toolkit.rag import (
        ...     load_web_document,
        ...     split_document_recursive,
        ...     create_vector_store,
        ...     create_rag_agent
        ... )
        >>> from langchain_community.embeddings import DashScopeEmbeddings
        >>> 
        >>> # 1. Load and split documents
        >>> docs = load_web_document("https://example.com/article")
        >>> chunks = split_document_recursive(docs, chunk_size=1000)
        >>> 
        >>> # 2. Create embeddings and vector store
        >>> embeddings = DashScopeEmbeddings(
        ...     model="text-embedding-v3",
        ...     dashscope_api_key="your-api-key"
        ... )
        >>> vector_store = create_vector_store(chunks, embeddings)
        >>> 
        >>> # 3. Create model
        >>> manager = ModelManager()
        >>> model = manager.create_model("qwen", model="qwen-turbo")
        >>> 
        >>> # 4. Create RAG agent
        >>> agent = create_rag_agent(model, vector_store, k=3)
        >>> 
        >>> # 5. Ask questions
        >>> result = agent.invoke({
        ...     "messages": [{"role": "user", "content": "What is the main topic?"}]
        ... })
        >>> print(result["messages"][-1].content)
    
    Note:
        - Agent automatically uses retrieval tool to find relevant context
        - System prompt guides agent to use retrieved information
        - Adjust k based on document size and query complexity
    """
    from langchain.agents import create_agent
    
    # Create retrieval tool
    retrieval_tool = create_retrieval_tool(vector_store, k=k)
    
    # Default RAG-optimized system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant with access to a document knowledge base. "
            "When answering questions:\n"
            "1. Use the retrieve_context tool to find relevant information from the document\n"
            "2. Base your answers on the retrieved context\n"
            "3. Provide accurate, detailed answers with specific examples when possible\n"
            "4. If the document doesn't contain the answer, say so clearly\n"
            "5. Cite the sources when providing information"
        )
    
    # Create agent with retrieval tool
    agent = create_agent(
        model=model,
        tools=[retrieval_tool],
        system_prompt=system_prompt,
        **kwargs
    )
    
    return agent


def retrieve_document_only(
    vector_store: VectorStore,
    model: BaseChatModel,
    query: str,
    k: int = 3,
    min_similarity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Retrieve answer from document only (no AI training data).
    
    Based on examples/practice/rag_05_retrieval_document_only.py
    
    Args:
        vector_store: Vector store to retrieve from
        model: Model for generating answer from context
        query: User's question
        k: Number of chunks to retrieve (default: 3)
        min_similarity: Optional minimum similarity threshold (0.0-1.0)
    
    Returns:
        dict with answer, retrieved_docs, context_used, num_chunks
    
    Example:
        >>> result = retrieve_document_only(
        ...     vector_store=vector_store,
        ...     model=model,
        ...     query="What is the main topic?",
        ...     k=3
        ... )
        >>> 
        >>> print(f"Answer: {result['answer']}")
        >>> print(f"Retrieved {result['num_chunks']} chunks")
    
    Note:
        - ALWAYS retrieves from document first
        - Returns empty if no relevant documents found
        - Guarantees answer comes ONLY from document
    """
    from langchain_core.messages import HumanMessage
    
    # Retrieve documents
    if min_similarity is not None:
        try:
            if hasattr(vector_store, 'similarity_search_with_score'):
                docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
                retrieved_docs = [
                    doc for doc, score in docs_with_scores 
                    if score >= min_similarity
                ]
            else:
                retrieved_docs = vector_store.similarity_search(query, k=k)
        except Exception:
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


def retrieve_with_priority(
    vector_store: VectorStore,
    model: BaseChatModel,
    query: str,
    k: int = 3,
    min_similarity: Optional[float] = None,
    use_ai_fallback: bool = True
) -> Dict[str, Any]:
    """
    Priority pattern: Document first, then AI model fallback.
    
    Based on examples/practice/rag_06_priority_fallback.py
    
    Args:
        vector_store: Vector store to retrieve from
        model: Model for generating answers
        query: User's question
        k: Number of chunks to retrieve (default: 3)
        min_similarity: Optional minimum similarity threshold (0.0-1.0)
        use_ai_fallback: Whether to use AI model when document has no answer
    
    Returns:
        dict with answer, source, retrieved_docs, fallback_used
        - source: "document" or "ai_model" (marks where answer came from)
    
    Example:
        >>> # Try document first, fallback to AI if needed
        >>> result = retrieve_with_priority(
        ...     vector_store=vector_store,
        ...     model=model,
        ...     query="What is the capital of France?",
        ...     use_ai_fallback=True
        ... )
        >>> 
        >>> print(f"Answer: {result['answer']}")
        >>> print(f"Source: {result['source']}")  # "document" or "ai_model"
        >>> print(f"Fallback used: {result['fallback_used']}")
    
    Note:
        - Always tries document first (priority)
        - Falls back to AI model if document has no answer
        - Source tracking shows where answer came from
        - Set use_ai_fallback=False for document-only mode
    """
    from langchain_core.messages import HumanMessage
    
    # Step 1: Try document retrieval first
    doc_result = retrieve_document_only(
        vector_store, model, query, k, min_similarity
    )
    
    # Step 2: Check if document has relevant information
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
            "fallback_used": False,
            "document_attempted": True
        }
    
    # Step 3: Document doesn't have answer
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
    
    # Step 4: Fallback to AI model
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
