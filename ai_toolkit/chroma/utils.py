"""
Chroma Utility Functions

工具函数模块：提供辅助功能。

核心功能：
    - 数据转换：LangChain Documents ↔ Chroma 格式
    - 验证工具：集合名称、元数据验证
    - 辅助函数：ID 生成、结果格式化

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import uuid
import re


def documents_to_chroma_format(
    documents: List[Document]
) -> Dict[str, List]:
    """
    将 LangChain Documents 转换为 Chroma 格式
    
    Args:
        documents: LangChain Document 列表
    
    Returns:
        包含 ids, documents, metadatas 的字典
    
    Example:
        >>> chroma_data = documents_to_chroma_format(docs)
        >>> collection.add(**chroma_data)
    """
    ids = []
    texts = []
    metadatas = []
    
    for doc in documents:
        ids.append(doc.id if hasattr(doc, 'id') and doc.id else str(uuid.uuid4()))
        texts.append(doc.page_content)
        metadatas.append(doc.metadata or {})
    
    return {
        "ids": ids,
        "documents": texts,
        "metadatas": metadatas
    }


def chroma_to_documents(
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None
) -> List[Document]:
    """
    将 Chroma 格式转换为 LangChain Documents
    
    Args:
        ids: 文档 ID 列表
        documents: 文档内容列表
        metadatas: 元数据列表（可选）
    
    Returns:
        LangChain Document 列表
    
    Example:
        >>> results = collection.get(ids=["doc1", "doc2"])
        >>> docs = chroma_to_documents(
        ...     ids=results["ids"],
        ...     documents=results["documents"],
        ...     metadatas=results["metadatas"]
        ... )
    """
    if metadatas is None:
        metadatas = [{}] * len(documents)
    
    return [
        Document(
            page_content=doc,
            metadata={**meta, "id": doc_id}
        )
        for doc_id, doc, meta in zip(ids, documents, metadatas)
    ]


def validate_collection_name(name: str) -> bool:
    """
    验证集合名称格式
    
    Chroma 集合名称规则：
        - 只能包含小写字母、数字、下划线、连字符
        - 不能以数字开头
        - 长度限制：1-63 字符
    
    Args:
        name: 集合名称
    
    Returns:
        是否有效
    
    Raises:
        ValueError: 如果名称无效
    
    Example:
        >>> validate_collection_name("ecommerce_docs")  # ✅
        >>> validate_collection_name("123docs")  # ❌ ValueError
    """
    if not name:
        raise ValueError("Collection name cannot be empty")
    
    if len(name) > 63:
        raise ValueError("Collection name must be <= 63 characters")
    
    # Chroma 集合名称规则：小写字母、数字、下划线、连字符
    pattern = r'^[a-z][a-z0-9_-]*$'
    if not re.match(pattern, name):
        raise ValueError(
            f"Invalid collection name: {name}. "
            "Must start with lowercase letter and contain only "
            "lowercase letters, numbers, underscores, and hyphens."
        )
    
    return True


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    验证元数据格式
    
    Chroma 元数据规则：
        - 值必须是字符串、数字、布尔值或 None
        - 键必须是字符串
    
    Args:
        metadata: 元数据字典
    
    Returns:
        是否有效
    
    Raises:
        ValueError: 如果元数据无效
    
    Example:
        >>> validate_metadata({"platform": "amazon", "count": 10})  # ✅
        >>> validate_metadata({"platform": {"nested": "value"}})  # ❌ ValueError
    """
    valid_types = (str, int, float, bool, type(None))
    
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"Metadata key must be string, got {type(key)}")
        
        if not isinstance(value, valid_types):
            raise ValueError(
                f"Metadata value must be str/int/float/bool/None, "
                f"got {type(value)} for key '{key}'"
            )
    
    return True


def generate_document_ids(count: int, prefix: str = "doc") -> List[str]:
    """
    生成文档 ID 列表
    
    Args:
        count: 需要生成的 ID 数量
        prefix: ID 前缀（默认: "doc"）
    
    Returns:
        ID 列表
    
    Example:
        >>> ids = generate_document_ids(5, prefix="chunk")
        >>> # ['chunk_0', 'chunk_1', 'chunk_2', 'chunk_3', 'chunk_4']
    """
    return [f"{prefix}_{i}" for i in range(count)]


def format_search_results(
    results: List[Document],
    include_score: bool = False,
    scores: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    格式化搜索结果
    
    Args:
        results: 文档列表
        include_score: 是否包含分数
        scores: 分数列表（如果 include_score=True）
    
    Returns:
        格式化后的结果列表
    
    Example:
        >>> results = store.similarity_search("query", k=5)
        >>> formatted = format_search_results(results)
        >>> 
        >>> # 带分数
        >>> results_with_score = store.similarity_search_with_score("query", k=5)
        >>> docs, scores = zip(*results_with_score)
        >>> formatted = format_search_results(docs, include_score=True, scores=scores)
    """
    formatted = []
    
    for i, doc in enumerate(results):
        item = {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        
        if include_score and scores:
            item["score"] = scores[i]
        
        formatted.append(item)
    
    return formatted
