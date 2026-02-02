"""
Document Loaders - Load documents from various sources

This module provides simplified document loading functions.

Functions:
    - load_web_document(): Load from URL with CheerioWebBaseLoader
    - load_pdf_document(): Load PDF files
    - load_json_document(): Load JSON files
    - load_csv_document(): Load CSV files

Based on: examples/practice/rag_01_loader_base.py
LangChain Version: 1.0
Python Version: 3.11

Author: AI Toolkit Team
Version: 1.0.0
"""

from typing import List, Optional
from langchain_core.documents import Document


def load_web_document(
    url: str,
    selector: Optional[str] = None,
    **kwargs
) -> List[Document]:
    """
    Load document from web URL using CheerioWebBaseLoader.
    
    Based on examples/practice/rag_01_loader_base.py
    
    Args:
        url: URL to load
        selector: Optional CSS selector to extract specific elements (e.g., "p" for paragraphs)
        **kwargs: Additional arguments for WebBaseLoader
    
    Returns:
        List of Document objects
    
    Example:
        >>> # Load entire page
        >>> docs = load_web_document("https://example.com")
        >>> 
        >>> # Load only paragraphs
        >>> docs = load_web_document(
        ...     "https://example.com",
        ...     selector="p"
        ... )
        >>> 
        >>> print(f"Loaded {len(docs)} documents")
        >>> print(f"Content: {docs[0].page_content[:200]}...")
    
    Note:
        - Uses BeautifulSoup for HTML parsing
        - Does not execute JavaScript (use Playwright for dynamic content)
        - Returns one document per webpage
    """
    from langchain_community.document_loaders import WebBaseLoader
    import bs4
    
    if selector:
        # Use BeautifulSoup selector
        bs4_strainer = bs4.SoupStrainer(selector)
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"parse_only": bs4_strainer},
            **kwargs
        )
    else:
        # Load entire page
        loader = WebBaseLoader(web_paths=[url], **kwargs)
    
    return loader.load()


def load_pdf_document(
    file_path: str,
    loader_type: str = "pypdf"
) -> List[Document]:
    """
    Load PDF document.
    
    Based on examples/practice/rag_01_loader_base.py
    
    Args:
        file_path: Path to PDF file
        loader_type: Type of PDF loader ("pypdf" or "unstructured")
    
    Returns:
        List of Document objects (one per page for pypdf)
    
    Example:
        >>> # Load with PyPDF (recommended)
        >>> docs = load_pdf_document("document.pdf", loader_type="pypdf")
        >>> 
        >>> # Load with Unstructured (more advanced)
        >>> docs = load_pdf_document("document.pdf", loader_type="unstructured")
        >>> 
        >>> print(f"Loaded {len(docs)} pages")
        >>> print(f"First page: {docs[0].page_content[:200]}...")
    
    Note:
        - pypdf: Fast, one document per page
        - unstructured: Slower, better structure extraction
        - Requires: pip install pypdf (for pypdf)
        - Requires: pip install unstructured (for unstructured)
    """
    if loader_type == "pypdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif loader_type == "unstructured":
        from langchain_community.document_loaders import UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(file_path)
    else:
        raise ValueError(f"Unknown loader_type: {loader_type}. Use 'pypdf' or 'unstructured'.")
    
    return loader.load()


def load_json_document(
    file_path: str,
    jq_schema: str = ".",
    text_content: bool = False
) -> List[Document]:
    """
    Load JSON document.
    
    Based on examples/practice/rag_01_loader_base.py
    
    Args:
        file_path: Path to JSON file
        jq_schema: JQ schema to extract content (default: "." for entire file)
        text_content: Whether JSON values are text (default: False)
    
    Returns:
        List of Document objects
    
    Example:
        >>> # Load entire JSON
        >>> docs = load_json_document("data.json")
        >>> 
        >>> # Extract specific field
        >>> docs = load_json_document(
        ...     "data.json",
        ...     jq_schema=".content"
        ... )
        >>> 
        >>> print(f"Loaded {len(docs)} documents")
        >>> print(f"Content: {docs[0].page_content}")
    
    Note:
        - Uses jq syntax for field extraction
        - jq_schema="." loads entire JSON
        - jq_schema=".field" extracts specific field
        - Requires: pip install jq (for jq support)
    """
    from langchain_community.document_loaders import JSONLoader
    
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=jq_schema,
        text_content=text_content
    )
    
    return loader.load()


def load_csv_document(
    file_path: str,
    **kwargs
) -> List[Document]:
    """
    Load CSV document.
    
    Based on examples/practice/rag_01_loader_base.py
    
    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments for CSVLoader
    
    Returns:
        List of Document objects (one per row)
    
    Example:
        >>> # Load CSV
        >>> docs = load_csv_document("data.csv")
        >>> 
        >>> print(f"Loaded {len(docs)} rows")
        >>> for i, doc in enumerate(docs[:3], 1):
        ...     print(f"Row {i}: {doc.page_content[:100]}...")
    
    Note:
        - Each row becomes a separate document
        - Header row is excluded
        - Metadata includes row number and source file
    """
    from langchain_community.document_loaders import CSVLoader
    
    loader = CSVLoader(file_path=file_path, **kwargs)
    
    return loader.load()
