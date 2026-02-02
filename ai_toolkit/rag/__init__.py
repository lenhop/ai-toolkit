"""
RAG Module - Retrieval Augmented Generation utilities

This module provides utilities for building RAG (Retrieval Augmented Generation) systems.

Submodules:
    - loaders: Document loaders for various formats
    - splitters: Document splitting utilities
    - retrievers: Retrieval and RAG agent helpers

Key Functions:
    - load_web_document(): Load from URL
    - load_pdf_document(): Load PDF files
    - split_document_recursive(): Split with RecursiveCharacterTextSplitter
    - create_vector_store(): Create vector store with embeddings
    - create_retrieval_tool(): Create retrieval tool for agent
    - create_rag_agent(): Complete RAG agent factory

Author: AI Toolkit Team
Version: 1.0.0
"""

from .loaders import (
    load_web_document,
    load_pdf_document,
    load_json_document,
    load_csv_document,
)

from .splitters import (
    split_document_recursive,
    split_with_overlap,
    split_for_chinese,
)

from .retrievers import (
    create_vector_store,
    create_retrieval_tool,
    create_rag_agent,
    retrieve_with_priority,
    retrieve_document_only,
)

__all__ = [
    # Loaders
    'load_web_document',
    'load_pdf_document',
    'load_json_document',
    'load_csv_document',
    # Splitters
    'split_document_recursive',
    'split_with_overlap',
    'split_for_chinese',
    # Retrievers
    'create_vector_store',
    'create_retrieval_tool',
    'create_rag_agent',
    'retrieve_with_priority',
    'retrieve_document_only',
]
