"""
Chroma VectorStore Wrapper

Chroma VectorStore wrapper module: provides a high-level interface and simplifies usage.

Core features:
    - create_chroma_store(): One-step VectorStore creation (most common)
    - from_documents(): Create from documents directly
    - load_store(): Load an existing store
    - ChromaStore: High-level wrapper class (flexible control)

Design principles:
    - All operations return LangChain VectorStore-compatible objects
    - High-level encapsulation hides low-level details
    - Supports chained calls and a fluent API

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


def get_chroma_collection(vectorstore: Chroma) -> Any:
    """
    Return the underlying Chroma Collection from a LangChain Chroma VectorStore.

    Use this when you need raw Chroma operations (get by ids/where/where_document,
    query with query_embeddings, add with precomputed embeddings, delete) via
    the chroma utils: collection_count, get_collection_items, add_to_collection,
    query_collection, delete_from_collection.

    Args:
        vectorstore: LangChain Chroma instance (e.g. from create_chroma_store).

    Returns:
        Chroma Collection instance.

    Example:
        >>> store = create_chroma_store(documents=docs, embeddings=embeddings, ...)
        >>> collection = get_chroma_collection(store)
        >>> ids = get_collection_ids(collection)
        >>> items = get_collection_items(collection, limit=5)
    """
    return vectorstore._collection


def create_chroma_store(
    documents: Optional[List[Document]] = None,
    embeddings: Optional[Embeddings] = None,
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
    client: Optional[Client] = None,
    **kwargs
) -> Chroma:
    """
    Create a Chroma VectorStore in one step (most common usage).

    This is the simplest creation path; it handles all details automatically.

    Args:
        documents: List of documents (optional; can be added later).
        embeddings: Embeddings model (required).
        collection_name: Collection name (default: "default").
        persist_directory: Persist directory (None for in-memory mode).
        client: Custom client (optional; overrides persist_directory).
        **kwargs: Other Chroma configuration parameters.

    Returns:
        LangChain Chroma VectorStore instance.

    Example:
        >>> from langchain_community.embeddings import DashScopeEmbeddings
        >>> from ai_toolkit.chroma import create_chroma_store
        >>>
        >>> store = create_chroma_store(
        ...     documents=chunks,
        ...     embeddings=DashScopeEmbeddings(...),
        ...     collection_name="ecommerce_docs",
        ...     persist_directory="./data/chroma"
        ... )
        >>> results = store.similarity_search("query", k=5)
    """
    if embeddings is None:
        raise ValueError("embeddings is required")

    validate_collection_name(collection_name)

    if client is None:
        if persist_directory:
            client = create_chroma_client(
                mode="persistent",
                persist_directory=persist_directory
            )
        else:
            client = create_chroma_client(mode="memory")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client,
        **kwargs
    )

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
    Create a VectorStore directly from documents (convenience method).

    Convenience alias for create_chroma_store() that makes document-based creation explicit.

    Args:
        documents: List of documents (required).
        embeddings: Embeddings model (required).
        collection_name: Collection name.
        persist_directory: Persist directory.
        **kwargs: Other configuration parameters.

    Returns:
        LangChain Chroma VectorStore instance.

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
    Load an existing VectorStore.

    Use to load a previously created store without adding new documents.

    Args:
        collection_name: Collection name (required).
        embeddings: Embeddings model (required; must match the one used at creation).
        persist_directory: Persist directory.
        client: Custom client.
        **kwargs: Other configuration parameters.

    Returns:
        LangChain Chroma VectorStore instance.

    Example:
        >>> store = load_store(
        ...     collection_name="ecommerce_docs",
        ...     embeddings=embeddings,
        ...     persist_directory="./data/chroma"
        ... )
    """
    return create_chroma_store(
        documents=None,
        embeddings=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
        client=client,
        **kwargs
    )


class ChromaStore:
    """
    High-level Chroma VectorStore wrapper class.

    Provides more flexible control and advanced features.
    Suited for scenarios that need fine-grained control.

    Example:
        >>> from ai_toolkit.chroma import ChromaStore
        >>>
        >>> store = ChromaStore(
        ...     collection_name="docs",
        ...     embeddings=embeddings,
        ...     persist_directory="./data/chroma"
        ... )
        >>> store.batch_add(documents=chunks, batch_size=100)
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
        Initialize ChromaStore.

        Args:
            collection_name: Collection name.
            embeddings: Embeddings model.
            persist_directory: Persist directory.
            client: Custom client.
            **kwargs: Other configuration parameters.
        """
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.persist_directory = persist_directory

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
        Add documents.

        Args:
            documents: List of documents.
            ids: List of document IDs (optional; auto-generated if not provided).
            **kwargs: Other parameters.

        Returns:
            List of document IDs.
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
        Add documents in batches (performance optimization).

        Args:
            documents: List of documents.
            batch_size: Batch size (default: 100).
            ids: List of document IDs (optional).
            **kwargs: Other parameters.

        Returns:
            List of all document IDs.
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
        Similarity search.

        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Other parameters.

        Returns:
            List of documents.
        """
        return self.store.similarity_search(query=query, k=k, **kwargs)
    
    def search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Similarity search with scores.

        Args:
            query: Query text.
            k: Number of documents to return.
            **kwargs: Other parameters.

        Returns:
            List of (document, score) tuples.
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
        Similarity search with metadata filter.

        Args:
            query: Query text.
            filter: Metadata filter (Chroma where format).
            k: Number of documents to return.
            **kwargs: Other parameters.

        Returns:
            List of documents.

        Example:
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
        Delete documents.

        Args:
            ids: List of document IDs (delete by id).
            filter: Metadata filter (delete by filter).
            **kwargs: Other parameters.

        Note:
            At least one of ids or filter must be provided.
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
        Update documents.

        Args:
            documents: New document content.
            ids: List of document IDs (must match length of documents).
            **kwargs: Other parameters.
        """
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")

        self.store.add_documents(documents=documents, ids=ids, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary of statistics.
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
        Clear all documents in the collection.
        """
        collection = self.store._collection
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    
    def __getattr__(self, name):
        """
        Delegate to the underlying VectorStore methods.
        Allows direct access to all LangChain Chroma functionality.
        """
        return getattr(self.store, name)
