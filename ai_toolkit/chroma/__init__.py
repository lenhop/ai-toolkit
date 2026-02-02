"""
Chroma Vector Database Toolkit

专业级 Chroma 向量数据库工具包，为 RAG 和 Agent 项目提供简洁易用的接口。

核心功能：
    - create_chroma_client(): 创建 Chroma 客户端
    - create_chroma_store(): 创建 Chroma VectorStore（一步到位）
    - ChromaStore: 高级 VectorStore 封装类

设计原则：
    - 简洁：3个核心文件，覆盖90%使用场景
    - 实用：基于实际 RAG/Agent 项目经验
    - 易用：高级封装，降低使用门槛
    - 兼容：完全兼容 LangChain API

Author: AI Toolkit Team
Version: 1.0.0
"""

from .client import (
    create_chroma_client,
    get_or_create_collection,
    list_collections,
)

from .store import (
    create_chroma_store,
    from_documents,
    load_store,
    ChromaStore,
)

from .utils import (
    documents_to_chroma_format,
    chroma_to_documents,
    validate_collection_name,
    validate_metadata,
    generate_document_ids,
    format_search_results,
)

__all__ = [
    # Client
    'create_chroma_client',
    'get_or_create_collection',
    'list_collections',
    # Store
    'create_chroma_store',
    'from_documents',
    'load_store',
    'ChromaStore',
    # Utils
    'documents_to_chroma_format',
    'chroma_to_documents',
    'validate_collection_name',
    'validate_metadata',
    'generate_document_ids',
    'format_search_results',
]
