"""
Chroma VectorStore Wrapper

Chroma VectorStore 封装模块：提供高级接口，简化使用。

核心功能：
    - create_chroma_store(): 一步创建 VectorStore（最常用）
    - from_documents(): 从文档直接创建
    - load_store(): 加载已有存储
    - ChromaStore: 高级封装类（灵活控制）

设计理念：
    - 所有操作返回 LangChain VectorStore 兼容对象
    - 提供高级封装，隐藏底层细节
    - 支持链式调用和流畅 API

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from chromadb import Client

from .client import create_chroma_client, get_or_create_collection
from .utils import generate_document_ids, validate_collection_name


def create_chroma_store(
    documents: Optional[List[Document]] = None,
    embeddings: Optional[Embeddings] = None,
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
    client: Optional[Client] = None,
    **kwargs
) -> Chroma:
    """
    一步创建 Chroma VectorStore（最常用）
    
    这是最简洁的创建方式，自动处理所有细节。
    
    Args:
        documents: 文档列表（可选，可后续添加）
        embeddings: Embeddings 模型（必需）
        collection_name: 集合名称（默认: "default"）
        persist_directory: 持久化目录（None 则使用内存模式）
        client: 自定义客户端（可选，覆盖 persist_directory）
        **kwargs: 其他 Chroma 配置参数
    
    Returns:
        LangChain Chroma VectorStore 实例
    
    Example:
        >>> from langchain_community.embeddings import DashScopeEmbeddings
        >>> from ai_toolkit.chroma import create_chroma_store
        >>> 
        >>> # 一步创建
        >>> store = create_chroma_store(
        ...     documents=chunks,
        ...     embeddings=DashScopeEmbeddings(...),
        ...     collection_name="ecommerce_docs",
        ...     persist_directory="./data/chroma"
        ... )
        >>> 
        >>> # 直接使用 LangChain API
        >>> results = store.similarity_search("query", k=5)
    """
    if embeddings is None:
        raise ValueError("embeddings is required")
    
    # 验证集合名称
    validate_collection_name(collection_name)
    
    # 创建或获取客户端
    if client is None:
        if persist_directory:
            client = create_chroma_client(
                mode="persistent",
                persist_directory=persist_directory
            )
        else:
            client = create_chroma_client(mode="memory")
    
    # 创建 VectorStore
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client,
        **kwargs
    )
    
    # 如果有文档，添加文档
    if documents:
        vectorstore.add_documents(documents)
    
    return vectorstore


def from_documents(
    documents: List[Document],
    embeddings: Embeddings,
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
    **kwargs
) -> Chroma:
    """
    从文档直接创建 VectorStore（便捷方法）
    
    这是 create_chroma_store() 的便捷别名，明确表示从文档创建。
    
    Args:
        documents: 文档列表（必需）
        embeddings: Embeddings 模型（必需）
        collection_name: 集合名称
        persist_directory: 持久化目录
        **kwargs: 其他配置参数
    
    Returns:
        LangChain Chroma VectorStore 实例
    
    Example:
        >>> store = from_documents(
        ...     documents=chunks,
        ...     embeddings=embeddings,
        ...     collection_name="docs"
        ... )
    """
    return create_chroma_store(
        documents=documents,
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        **kwargs
    )


def load_store(
    collection_name: str,
    embeddings: Embeddings,
    persist_directory: Optional[str] = None,
    client: Optional[Client] = None,
    **kwargs
) -> Chroma:
    """
    加载已有的 VectorStore
    
    用于加载之前创建的存储，不添加新文档。
    
    Args:
        collection_name: 集合名称（必需）
        embeddings: Embeddings 模型（必需，需与创建时一致）
        persist_directory: 持久化目录
        client: 自定义客户端
        **kwargs: 其他配置参数
    
    Returns:
        LangChain Chroma VectorStore 实例
    
    Example:
        >>> # 加载已有存储
        >>> store = load_store(
        ...     collection_name="ecommerce_docs",
        ...     embeddings=embeddings,
        ...     persist_directory="./data/chroma"
        ... )
    """
    return create_chroma_store(
        documents=None,  # 不添加新文档
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        client=client,
        **kwargs
    )


class ChromaStore:
    """
    Chroma VectorStore 高级封装类
    
    提供更灵活的控制和高级功能。
    适合需要精细控制的场景。
    
    Example:
        >>> from ai_toolkit.chroma import ChromaStore
        >>> 
        >>> store = ChromaStore(
        ...     collection_name="docs",
        ...     embeddings=embeddings,
        ...     persist_directory="./data/chroma"
        ... )
        >>> 
        >>> # 批量添加
        >>> store.batch_add(documents=chunks, batch_size=100)
        >>> 
        >>> # 带过滤搜索
        >>> results = store.search_with_filter(
        ...     query="product",
        ...     filter={"platform": "amazon"},
        ...     k=5
        ... )
    """
    
    def __init__(
        self,
        collection_name: str,
        embeddings: Embeddings,
        persist_directory: Optional[str] = None,
        client: Optional[Client] = None,
        **kwargs
    ):
        """
        初始化 ChromaStore
        
        Args:
            collection_name: 集合名称
            embeddings: Embeddings 模型
            persist_directory: 持久化目录
            client: 自定义客户端
            **kwargs: 其他配置参数
        """
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        
        # 创建 VectorStore
        self.store = create_chroma_store(
            documents=None,
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client=client,
            **kwargs
        )
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        添加文档
        
        Args:
            documents: 文档列表
            ids: 文档 ID 列表（可选，自动生成）
            **kwargs: 其他参数
        
        Returns:
            文档 ID 列表
        """
        if ids is None:
            ids = generate_document_ids(len(documents))
        
        return self.store.add_documents(
            documents=documents,
            ids=ids,
            **kwargs
        )
    
    def batch_add(
        self,
        documents: List[Document],
        batch_size: int = 100,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        批量添加文档（性能优化）
        
        Args:
            documents: 文档列表
            batch_size: 批次大小（默认: 100）
            ids: 文档 ID 列表（可选）
            **kwargs: 其他参数
        
        Returns:
            所有文档 ID 列表
        """
        if ids is None:
            ids = generate_document_ids(len(documents))
        
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_ids = self.store.add_documents(
                documents=batch_docs,
                ids=batch_ids,
                **kwargs
            )
            all_ids.extend(batch_ids)
        
        return all_ids
    
    def search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            **kwargs: 其他参数
        
        Returns:
            文档列表
        """
        return self.store.similarity_search(query=query, k=k, **kwargs)
    
    def search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        带相似度分数的搜索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            **kwargs: 其他参数
        
        Returns:
            (文档, 分数) 元组列表
        """
        return self.store.similarity_search_with_score(query=query, k=k, **kwargs)
    
    def search_with_filter(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """
        带元数据过滤的搜索
        
        Args:
            query: 查询文本
            filter: 元数据过滤条件（Chroma where 格式）
            k: 返回文档数量
            **kwargs: 其他参数
        
        Returns:
            文档列表
        
        Example:
            >>> # 搜索特定平台的文档
            >>> results = store.search_with_filter(
            ...     query="return policy",
            ...     filter={"platform": "amazon"},
            ...     k=5
            ... )
        """
        if filter:
            return self.store.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs
            )
        else:
            return self.search(query=query, k=k, **kwargs)
    
    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        删除文档
        
        Args:
            ids: 文档 ID 列表（按 ID 删除）
            filter: 元数据过滤条件（按过滤删除）
            **kwargs: 其他参数
        
        Note:
            ids 和 filter 至少提供一个
        """
        if ids:
            self.store.delete(ids=ids, **kwargs)
        elif filter:
            self.store.delete(filter=filter, **kwargs)
        else:
            raise ValueError("Either ids or filter must be provided")
    
    def update_documents(
        self,
        documents: List[Document],
        ids: List[str],
        **kwargs
    ) -> None:
        """
        更新文档
        
        Args:
            documents: 新文档内容
            ids: 文档 ID 列表（需与 documents 长度一致）
            **kwargs: 其他参数
        """
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")
        
        # Chroma 使用 upsert 更新
        self.store.add_documents(documents=documents, ids=ids, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        collection = self.store._collection
        count = collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
        }
    
    def clear(self) -> None:
        """
        清空集合中的所有文档
        """
        collection = self.store._collection
        # 获取所有 ID 并删除
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    
    def __getattr__(self, name):
        """
        代理到底层 VectorStore 的所有方法
        允许直接访问 LangChain Chroma 的所有功能
        """
        return getattr(self.store, name)
