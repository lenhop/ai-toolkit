"""
Chroma Utility Functions

Utility module: provides helper functionality.

Core features:
    - Data conversion: LangChain Documents <-> Chroma format
    - Validation: collection name, metadata validation
    - Helpers: ID generation, result formatting
    - Collection common operations: count/get/add/query/delete (Chroma Collection API)

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
    Convert LangChain Documents to Chroma format.

    Args:
        documents: List of LangChain Documents.

    Returns:
        Dictionary with keys ids, documents, metadatas.

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
    Convert Chroma format to LangChain Documents.

    Args:
        ids: List of document IDs.
        documents: List of document contents.
        metadatas: List of metadata dicts (optional).

    Returns:
        List of LangChain Documents.

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
    Validate collection name format.

    Chroma collection name rules:
        - Only lowercase letters, digits, underscores, hyphens
        - Cannot start with a digit
        - Length 1-63 characters

    Args:
        name: Collection name.

    Returns:
        True if valid.

    Raises:
        ValueError: If name is invalid.

    Example:
        >>> validate_collection_name("ecommerce_docs")
        >>> validate_collection_name("123docs")  # ValueError
    """
    if not name:
        raise ValueError("Collection name cannot be empty")

    if len(name) > 63:
        raise ValueError("Collection name must be <= 63 characters")

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
    Validate metadata format.

    Chroma metadata rules:
        - Values must be str, int, float, bool, or None
        - Keys must be strings

    Args:
        metadata: Metadata dictionary.

    Returns:
        True if valid.

    Raises:
        ValueError: If metadata is invalid.

    Example:
        >>> validate_metadata({"platform": "amazon", "count": 10})
        >>> validate_metadata({"platform": {"nested": "value"}})  # ValueError
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
    Generate a list of document IDs.

    Args:
        count: Number of IDs to generate.
        prefix: ID prefix (default: "doc").

    Returns:
        List of IDs.

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
    Format search results.

    Args:
        results: List of documents.
        include_score: Whether to include score.
        scores: List of scores (if include_score=True).

    Returns:
        List of formatted result dicts.

    Example:
        >>> results = store.similarity_search("query", k=5)
        >>> formatted = format_search_results(results)
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


# ---------------------------------------------------------------------------
# Chroma collection common operations (raw Collection API helpers)
# ---------------------------------------------------------------------------


def collection_count(collection: Any) -> int:
    """
    Return the number of items in a Chroma collection.

    Args:
        collection: Chroma Collection instance (from get_or_create_collection).

    Returns:
        Item count.

    Example:
        >>> client = create_chroma_client(mode="persistent", persist_directory="./data/chroma")
        >>> coll = get_or_create_collection(client, "my_docs")
        >>> n = collection_count(coll)
    """
    return collection.count()


def get_collection_ids(collection: Any) -> List[str]:
    """
    Return all document IDs in the collection without loading other fields.

    Args:
        collection: Chroma Collection instance.

    Returns:
        List of ids.

    Example:
        >>> ids = get_collection_ids(collection)
    """
    return collection.get(include=[])["ids"]


def get_collection_items(
    collection: Any,
    ids: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
    limit: int = 100000,
    offset: int = 0,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get items from a Chroma collection with optional filters and pagination.

    Args:
        collection: Chroma Collection instance.
        ids: Optional list of ids to fetch. None means no id filter.
        where: Optional metadata filter (e.g. {"source": "file.pdf"}, {"page": 0}).
        where_document: Optional document content filter (e.g. {"$contains": "keyword"}).
        limit: Max number of results (default 100000).
        offset: Pagination offset (e.g. offset=10 skips first 10).
        include: Fields to return (ids are always returned). Valid: "metadatas",
                 "documents", "embeddings", "uris", "data". Default ["metadatas", "documents"].

    Returns:
        Dict with keys ids, metadatas, documents, embeddings (as per include).

    Example:
        >>> items = get_collection_items(collection, limit=5, include=["ids", "documents"])
        >>> items = get_collection_items(collection, where={"page": 0}, limit=3)
        >>> items = get_collection_items(collection, where_document={"$contains": "fee"})
        >>> page2 = get_collection_items(collection, limit=10, offset=10)
    """
    if include is None:
        include = ["metadatas", "documents"]
    kwargs = {"limit": limit, "offset": offset, "include": include}
    if ids is not None:
        kwargs["ids"] = ids
    if where is not None:
        kwargs["where"] = where
    if where_document is not None:
        kwargs["where_document"] = where_document
    return collection.get(**kwargs)


def add_to_collection(
    collection: Any,
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    embeddings: Optional[List[List[float]]] = None,
) -> None:
    """
    Add documents to a Chroma collection (with optional precomputed embeddings).

    Args:
        collection: Chroma Collection instance.
        ids: Document ids (same length as documents).
        documents: Text content for each document.
        metadatas: Optional metadata dict per document. Defaults to [{}] * len(documents).
        embeddings: Optional precomputed vectors. If None, collection must have an
                    embedding function configured.

    Example:
        >>> add_to_collection(collection, ids=["d1", "d2"], documents=["text1", "text2"],
        ...                   metadatas=[{"page": 0}, {"page": 1}], embeddings=vectors)
    """
    if len(ids) != len(documents):
        raise ValueError("ids and documents must have the same length")
    if metadatas is None:
        metadatas = [{}] * len(documents)
    if len(metadatas) != len(documents):
        raise ValueError("metadatas length must match documents")
    kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
    if embeddings is not None:
        if len(embeddings) != len(documents):
            raise ValueError("embeddings length must match documents")
        kwargs["embeddings"] = embeddings
    collection.add(**kwargs)


def query_collection(
    collection: Any,
    query_embeddings: Optional[List[List[float]]] = None,
    query_texts: Optional[List[str]] = None,
    n_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Semantic search over a Chroma collection by vector or text.

    Args:
        collection: Chroma Collection instance.
        query_embeddings: Optional list of query vectors (use when embedding externally).
        query_texts: Optional list of query strings (requires collection embedding function).
        n_results: Number of results per query (default 10).
        where: Optional metadata filter.
        include: Fields to return (default ["documents", "metadatas", "distances"]).

    Returns:
        Dict with keys ids, metadatas, documents, distances (as per include).
        Each value is a list of lists (one per query).

    Raises:
        ValueError: If neither query_embeddings nor query_texts is provided.

    Example:
        >>> res = query_collection(collection, query_embeddings=[query_vector], n_results=5)
        >>> res = query_collection(collection, query_texts=["FBA customer service"], n_results=3)
    """
    if query_embeddings is None and query_texts is None:
        raise ValueError("Provide at least one of query_embeddings or query_texts")
    if include is None:
        include = ["documents", "metadatas", "distances"]
    kwargs = {"n_results": n_results, "include": include}
    if query_embeddings is not None:
        kwargs["query_embeddings"] = query_embeddings
    if query_texts is not None:
        kwargs["query_texts"] = query_texts
    if where is not None:
        kwargs["where"] = where
    return collection.query(**kwargs)


def delete_from_collection(
    collection: Any,
    ids: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Delete items from a Chroma collection by ids or metadata filter.

    Args:
        collection: Chroma Collection instance.
        ids: Optional list of ids to delete.
        where: Optional metadata filter (delete all matching).

    Raises:
        ValueError: If neither ids nor where is provided.

    Example:
        >>> delete_from_collection(collection, ids=["doc_1", "doc_2"])
        >>> delete_from_collection(collection, where={"source": "obsolete.pdf"})
    """
    if ids is None and where is None:
        raise ValueError("Provide at least one of ids or where")
    if ids is not None:
        collection.delete(ids=ids)
    else:
        collection.delete(where=where)
