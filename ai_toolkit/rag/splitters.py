"""
Document Splitters - Split documents into chunks

This module provides document splitting utilities for RAG systems.

Functions:
    - split_document_recursive(): Split with RecursiveCharacterTextSplitter
    - split_with_overlap(): Split with configurable overlap
    - split_for_chinese(): Chinese-optimized splitting

Based on: examples/practice/rag_03_split_document.py
LangChain Version: 1.0
Python Version: 3.11

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Optional
from langchain_core.documents import Document


def split_document_recursive(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    add_start_index: bool = True,
    **kwargs
) -> List[Document]:
    """
    Split documents using RecursiveCharacterTextSplitter.
    
    Based on examples/practice/rag_03_split_document.py
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        add_start_index: Track index in original document (default: True)
        **kwargs: Additional arguments for RecursiveCharacterTextSplitter
    
    Returns:
        List of split document chunks
    
    Example:
        >>> from ai_toolkit.rag import load_web_document, split_document_recursive
        >>> 
        >>> # Load document
        >>> docs = load_web_document("https://example.com")
        >>> 
        >>> # Split into chunks
        >>> chunks = split_document_recursive(
        ...     docs,
        ...     chunk_size=1000,
        ...     chunk_overlap=200
        ... )
        >>> 
        >>> print(f"Split into {len(chunks)} chunks")
        >>> print(f"First chunk: {chunks[0].page_content[:200]}...")
    
    Note:
        - Tries to split on natural boundaries (paragraphs, sentences)
        - Overlap helps maintain context between chunks
        - add_start_index tracks position in original document
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
        length_function=len,
        **kwargs
    )
    
    return splitter.split_documents(documents)


def split_with_overlap(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> List[Document]:
    """
    Split documents with custom separators and overlap.
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        separators: Custom separators (default: ["\n\n", "\n", " ", ""])
    
    Returns:
        List of split document chunks
    
    Example:
        >>> # Split with default separators
        >>> chunks = split_with_overlap(docs, chunk_size=500)
        >>> 
        >>> # Split with custom separators
        >>> chunks = split_with_overlap(
        ...     docs,
        ...     chunk_size=500,
        ...     separators=["\n\n", "\n", ". ", " "]
        ... )
    
    Note:
        - Separators are tried in order
        - First separator that fits is used
        - Falls back to character-level splitting if needed
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    
    return splitter.split_documents(documents)


def split_for_chinese(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents optimized for Chinese text.
    
    Based on examples/practice/rag_04_agent_workflow.py
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
    
    Returns:
        List of split document chunks
    
    Example:
        >>> # Load Chinese document
        >>> docs = load_web_document("https://example.com/chinese-article")
        >>> 
        >>> # Split with Chinese-friendly separators
        >>> chunks = split_for_chinese(
        ...     docs,
        ...     chunk_size=1000,
        ...     chunk_overlap=200
        ... )
        >>> 
        >>> print(f"Split into {len(chunks)} chunks")
    
    Note:
        - Uses Chinese punctuation as separators
        - Separators: \n\n, \n, 。, ，, space, ""
        - Better preserves Chinese sentence boundaries
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Chinese-friendly separators
    separators = ["\n\n", "\n", "。", "，", " ", ""]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    
    return splitter.split_documents(documents)
