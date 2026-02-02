"""
Chroma Client Management

客户端管理模块：创建和管理 Chroma 客户端和集合。

核心功能：
    - create_chroma_client(): 创建客户端（统一入口）
    - get_or_create_collection(): 获取或创建集合（常用操作）
    - list_collections(): 列出所有集合

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import Optional, List, Dict, Any
import chromadb
from chromadb import Client, PersistentClient, HttpClient
from chromadb.config import Settings


def create_chroma_client(
    mode: str = "persistent",
    persist_directory: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    **kwargs
) -> Client:
    """
    创建 Chroma 客户端（统一入口）
    
    支持三种模式：
        - "memory": 内存模式（测试用）
        - "persistent": 持久化模式（推荐，生产环境）
        - "http": HTTP 客户端模式（远程服务器）
    
    Args:
        mode: 客户端模式 ("memory" | "persistent" | "http")
        persist_directory: 持久化目录（persistent 模式必需）
        host: 服务器地址（http 模式必需）
        port: 服务器端口（http 模式，默认 8000）
        **kwargs: 其他 Chroma 配置参数
    
    Returns:
        Chroma 客户端实例
    
    Example:
        >>> # 持久化模式（推荐）
        >>> client = create_chroma_client(
        ...     mode="persistent",
        ...     persist_directory="./data/chroma"
        ... )
        >>> 
        >>> # 内存模式（测试）
        >>> client = create_chroma_client(mode="memory")
        >>> 
        >>> # HTTP 模式（远程服务器）
        >>> client = create_chroma_client(
        ...     mode="http",
        ...     host="localhost",
        ...     port=8000
        ... )
    """
    if mode == "memory":
        return chromadb.Client()
    
    elif mode == "persistent":
        if persist_directory is None:
            raise ValueError("persist_directory is required for persistent mode")
        return PersistentClient(path=persist_directory, **kwargs)
    
    elif mode == "http":
        if host is None:
            raise ValueError("host is required for http mode")
        port = port or 8000
        return HttpClient(host=host, port=port, **kwargs)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'memory', 'persistent', or 'http'")


def get_or_create_collection(
    client: Client,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    获取或创建集合（常用操作）
    
    如果集合不存在则创建，存在则返回。
    这是最常用的集合操作。
    
    Args:
        client: Chroma 客户端实例
        name: 集合名称
        metadata: 集合元数据
        **kwargs: 其他集合配置参数（如 distance_function）
    
    Returns:
        Collection 实例
    
    Note:
        ChromaDB 的 distance 参数在较新版本中已移除，
        距离函数在查询时自动处理。
    
    Example:
        >>> client = create_chroma_client(persist_directory="./data/chroma")
        >>> collection = get_or_create_collection(
        ...     client=client,
        ...     name="ecommerce_docs",
        ...     metadata={"description": "E-commerce documents"}
        ... )
    """
    try:
        # 尝试获取现有集合
        collection = client.get_collection(name=name)
        return collection
    except Exception:
        # 集合不存在，创建新集合
        # 注意：ChromaDB 0.4.x 版本中，distance 参数已移除
        collection = client.create_collection(
            name=name,
            metadata=metadata,
            **kwargs
        )
        return collection


def list_collections(client: Client) -> List[str]:
    """
    列出所有集合名称
    
    Args:
        client: Chroma 客户端实例
    
    Returns:
        集合名称列表
    
    Example:
        >>> client = create_chroma_client(persist_directory="./data/chroma")
        >>> collections = list_collections(client)
        >>> print(f"Collections: {collections}")
    """
    return [col.name for col in client.list_collections()]
