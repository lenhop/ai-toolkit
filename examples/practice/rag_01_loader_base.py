"""
RAG Document Loaders - Comprehensive Examples

This module demonstrates various document loaders for RAG (Retrieval Augmented Generation) applications.
Supports: PDF, JSON, JSONL, CSV, DOCX, PPTX, and Web pages (Cheerio & WebBaseLoader).

CheerioWebBaseLoader:
    Python implementation matching the JavaScript CheerioWebBaseLoader API from @langchain/community.
    Reference: https://docs.langchain.com/oss/javascript/integrations/document_loaders/web_loaders/web_cheerio
    
    JavaScript API:
        import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio"
        const loader = new CheerioWebBaseLoader(url, { selector: "p" })
        const docs = await loader.load()
    
    Python API (equivalent):
        from rag_01_loader_base import CheerioWebBaseLoader
        loader = CheerioWebBaseLoader(url, selector="p")
        docs = loader.load()

Required Dependencies:
    - langchain-community>=0.0.20
    - beautifulsoup4 (for web loaders - Cheerio equivalent)
    - requests (for web loaders)

Optional Dependencies (for specific loaders):
    - pypdf (for PyPDFLoader)
    - unstructured (for UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader)
    - docx2txt (for Docx2txtLoader)
    
Install optional dependencies:
    pip install pypdf unstructured docx2txt
"""

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
    JSONLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.document_loaders.web_base import WebBaseLoader as BaseWebLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from typing import Optional, List
import bs4
import json
import os


# ============================================================================
# 1. Web Page Loaders
# ============================================================================

class CheerioWebBaseLoader(BaseWebLoader):
    """
    Web loader with selector-based extraction using BeautifulSoup (Cheerio equivalent).
    
    Python implementation of CheerioWebBaseLoader from @langchain/community.
    Uses BeautifulSoup for HTML parsing and CSS selector support, matching the JavaScript API.
    
    Reference: https://docs.langchain.com/oss/javascript/integrations/document_loaders/web_loaders/web_cheerio
    
    Features:
    - Fast and lightweight HTML parsing (no browser simulation)
    - CSS selector support for targeted content extraction
    - One document per webpage
    - Does not execute JavaScript (use Playwright/Puppeteer loaders for dynamic content)
    
    Example:
        >>> loader = CheerioWebBaseLoader("https://example.com", selector="p")
        >>> docs = loader.load()
    """
    
    def __init__(
        self,
        web_path: str,
        selector: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Cheerio web loader.
        
        Args:
            web_path: URL string to load (can be single URL or list of URLs)
            selector: Optional CSS selector to extract specific elements (e.g., "p" for paragraphs)
            **kwargs: Additional arguments passed to WebBaseLoader:
                - header_template: Custom headers dict
                - verify_ssl: Whether to verify SSL certificates (default: True)
                - continue_on_failure: Whether to continue if one URL fails (default: False)
                - autoset_encoding: Auto-detect encoding (default: True)
                - encoding: Manual encoding override
                - requests_per_second: Rate limiting (default: 2)
                - default_parser: HTML parser to use (default: "html.parser")
        
        Note: This matches the JavaScript API where selector is passed in config:
            JavaScript: new CheerioWebBaseLoader(url, { selector: "p" })
            Python: CheerioWebBaseLoader(url, selector="p")
        """
        # Support both single URL and list of URLs (like JavaScript version)
        if isinstance(web_path, str):
            web_paths = [web_path]
        else:
            web_paths = web_path
        
        super().__init__(web_paths=web_paths, **kwargs)
        self.selector = selector
    
    def load(self) -> List[Document]:
        """
        Load documents from web path(s) with optional selector filtering.
        
        Creates one document per webpage. If selector is provided, only extracts
        content matching the CSS selector.
        
        Returns:
            List of Document objects, one per webpage
            
        Example:
            >>> loader = CheerioWebBaseLoader("https://example.com", selector="p")
            >>> docs = loader.load()
            >>> print(docs[0].page_content)  # Only paragraph content
            >>> print(docs[0].metadata)  # {'source': 'https://example.com'}
        """
        import requests
        
        all_docs = []
        
        # Process each URL (supporting multiple URLs like JavaScript version)
        for url in self.web_paths:
            try:
                # Fetch HTML content with proper headers
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                # Merge with any custom headers from kwargs
                if hasattr(self, 'header_template') and self.header_template:
                    headers.update(self.header_template)
                
                response = requests.get(
                    url,
                    headers=headers,
                    verify=getattr(self, 'verify_ssl', True),
                    timeout=30
                )
                response.raise_for_status()
                html_content = response.text
                
                # Parse with BeautifulSoup (equivalent to Cheerio)
                soup = BeautifulSoup(html_content, getattr(self, 'default_parser', 'html.parser'))
                
                # Apply selector if provided (matches JavaScript behavior)
                if self.selector:
                    # Use CSS selector to find matching elements
                    selected_elements = soup.select(self.selector)
                    # Extract text from selected elements, preserving structure
                    text_parts = []
                    for elem in selected_elements:
                        text = elem.get_text(separator=" ", strip=True)
                        if text:  # Only add non-empty text
                            text_parts.append(text)
                    text_content = "\n".join(text_parts)
                else:
                    # Extract all text if no selector (default behavior)
                    text_content = soup.get_text(separator=" ", strip=True)
                
                # Create metadata matching JavaScript format
                metadata = {
                    "source": url
                }
                
                # Create document (one per webpage, matching JavaScript behavior)
                doc = Document(page_content=text_content, metadata=metadata)
                all_docs.append(doc)
                
            except Exception as e:
                # Handle errors based on continue_on_failure setting
                if getattr(self, 'continue_on_failure', False):
                    print(f"Warning: Failed to load {url}: {e}")
                    continue
                else:
                    raise RuntimeError(f"Failed to load {url}: {e}") from e
        
        return all_docs


def example_web_loader_cheerio():
    """
    Example: Load web page using CheerioWebBaseLoader (BeautifulSoup selector).
    
    Matches JavaScript API: new CheerioWebBaseLoader(url, { selector: "p" })
    Reference: https://docs.langchain.com/oss/javascript/integrations/document_loaders/web_loaders/web_cheerio
    """
    print("=" * 80)
    print("Example 1: Web Loader - CheerioWebBaseLoader (BeautifulSoup)")
    print("=" * 80)
    
    # Example 1a: Load without selector (extracts all text)
    print("\n1a. Loading without selector (all content):")
    loader_all = CheerioWebBaseLoader(
        web_path="https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    docs_all = loader_all.load()
    print(f"   ✓ Loaded {len(docs_all)} document(s)")
    print(f"   ✓ Total characters: {len(docs_all[0].page_content)}")
    
    # Example 1b: Load with selector (only paragraphs)
    print("\n1b. Loading with selector='p' (paragraphs only):")
    p_tag_selector = "p"
    loader = CheerioWebBaseLoader(
        web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
        selector=p_tag_selector
    )
    docs = loader.load()
    
    # Verify and display results
    assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
    print(f"   ✓ Loaded {len(docs)} document(s)")
    print(f"   ✓ Total characters: {len(docs[0].page_content)}")
    print(f"   ✓ Metadata: {docs[0].metadata}")
    print(f"   ✓ Content reduction: {len(docs_all[0].page_content) - len(docs[0].page_content)} characters filtered")
    print(f"\n   First 200 characters:\n   {docs[0].page_content[:200]}...\n")


def example_web_loader_webbase():
    """Example: Load web page using WebBaseLoader with BeautifulSoup selector."""
    print("=" * 80)
    print("Example 2: Web Loader - WebBaseLoader with BeautifulSoup")
    print("=" * 80)
    
    # Define CSS selector
    p_tag_selector = "p"
    
    # Create loader with BeautifulSoup selector
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    bs4_strainer = bs4.SoupStrainer(p_tag_selector)
    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs={
            "parse_only": bs4_strainer
        }
    )
    
    # Load documents
    docs = loader.load()
    
    # Verify and display results
    assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
    print(f"✓ Loaded {len(docs)} document(s)")
    print(f"✓ Total characters: {len(docs[0].page_content)}")
    print(f"\nFirst 200 characters:\n{docs[0].page_content[:200]}...\n")


# ============================================================================
# 2. PDF Loaders
# ============================================================================

def example_pdf_loader_pypdf(file_path: str = "sample.pdf"):
    """
    Example: Load PDF using PyPDFLoader.
    
    Args:
        file_path: Path to PDF file
    """
    print("=" * 80)
    print("Example 3: PDF Loader - PyPDFLoader")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        print("  Create a PDF file or update the file_path parameter\n")
        return
    
    # Create loader
    loader = PyPDFLoader(file_path)
    
    # Load documents (each page becomes a document)
    docs = loader.load()
    
    print(f"✓ Loaded {len(docs)} page(s) from PDF")
    if docs:
        print(f"✓ First page characters: {len(docs[0].page_content)}")
        print(f"✓ Metadata: {docs[0].metadata}")
        print(f"\nFirst 200 characters:\n{docs[0].page_content[:200]}...\n")


def example_pdf_loader_unstructured(file_path: str = "sample.pdf"):
    """
    Example: Load PDF using UnstructuredPDFLoader (more advanced).
    
    Args:
        file_path: Path to PDF file
    """
    print("=" * 80)
    print("Example 4: PDF Loader - UnstructuredPDFLoader")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        print("  Create a PDF file or update the file_path parameter\n")
        return
    
    # Create loader
    loader = UnstructuredPDFLoader(file_path)
    
    # Load documents
    docs = loader.load()
    
    print(f"✓ Loaded {len(docs)} document(s) from PDF")
    if docs:
        print(f"✓ Total characters: {len(docs[0].page_content)}")
        print(f"✓ Metadata: {docs[0].metadata}")
        print(f"\nFirst 200 characters:\n{docs[0].page_content[:200]}...\n")


# ============================================================================
# 3. JSON Loader
# ============================================================================

def example_json_loader(file_path: str = "sample.json"):
    """
    Example: Load JSON file using JSONLoader.
    
    Args:
        file_path: Path to JSON file
    """
    print("=" * 80)
    print("Example 5: JSON Loader")
    print("=" * 80)
    
    # Create sample JSON file if it doesn't exist
    if not os.path.exists(file_path):
        sample_data = {
            "title": "Sample Document",
            "content": "This is a sample JSON document for RAG testing.",
            "author": "AI Toolkit",
            "date": "2024-01-01"
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        print(f"✓ Created sample JSON file: {file_path}")
    
    # Create loader - specify jq_schema to extract content
    # jq_schema defines which fields to extract (using jq syntax)
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.content',  # Extract the 'content' field
        text_content=False  # Set to True if JSON values are text
    )
    
    # Load documents
    docs = loader.load()
    
    print(f"✓ Loaded {len(docs)} document(s) from JSON")
    if docs:
        print(f"✓ Total characters: {len(docs[0].page_content)}")
        print(f"✓ Metadata: {docs[0].metadata}")
        print(f"\nContent:\n{docs[0].page_content}\n")


# ============================================================================
# 4. JSONL (JSON Lines) Loader
# ============================================================================

def example_jsonl_loader(file_path: str = "sample.jsonl"):
    """
    Example: Load JSONL file using JSONLoader.
    
    JSONL format: Each line is a separate JSON object.
    Note: JSONLoader processes the entire file, so we need to handle JSONL manually.
    
    Args:
        file_path: Path to JSONL file
    """
    print("=" * 80)
    print("Example 6: JSONL (JSON Lines) Loader")
    print("=" * 80)
    
    # Create sample JSONL file if it doesn't exist
    if not os.path.exists(file_path):
        sample_data = [
            {"id": 1, "text": "First document in JSONL format."},
            {"id": 2, "text": "Second document in JSONL format."},
            {"id": 3, "text": "Third document in JSONL format."}
        ]
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        print(f"✓ Created sample JSONL file: {file_path}")
    
    # Load JSONL manually (each line is a separate JSON object)
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Extract text field (or use entire object as string)
                content = data.get('text', json.dumps(data))
                metadata = {
                    "source": file_path,
                    "line_number": line_num,
                    **{k: v for k, v in data.items() if k != 'text'}
                }
                docs.append(Document(page_content=content, metadata=metadata))
            except json.JSONDecodeError as e:
                print(f"⚠ Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"✓ Loaded {len(docs)} document(s) from JSONL")
    for i, doc in enumerate(docs[:3], 1):  # Show first 3
        print(f"  Document {i}: {len(doc.page_content)} characters")
        print(f"    Content: {doc.page_content[:100]}...")
        print(f"    Metadata: {doc.metadata}")
    print()


# ============================================================================
# 5. CSV Loader
# ============================================================================

def example_csv_loader(file_path: str = "sample.csv"):
    """
    Example: Load CSV file using CSVLoader.
    
    Args:
        file_path: Path to CSV file
    """
    print("=" * 80)
    print("Example 7: CSV Loader")
    print("=" * 80)
    
    # Create sample CSV file if it doesn't exist
    if not os.path.exists(file_path):
        import csv
        sample_data = [
            ["id", "title", "content"],
            ["1", "Document 1", "This is the first document content."],
            ["2", "Document 2", "This is the second document content."],
            ["3", "Document 3", "This is the third document content."]
        ]
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)
        print(f"✓ Created sample CSV file: {file_path}")
    
    # Create loader
    # Each row becomes a document (excluding header)
    loader = CSVLoader(file_path=file_path)
    
    # Load documents
    docs = loader.load()
    
    print(f"✓ Loaded {len(docs)} row(s) from CSV")
    for i, doc in enumerate(docs[:3], 1):  # Show first 3
        print(f"  Row {i}: {len(doc.page_content)} characters")
        print(f"    Content: {doc.page_content[:100]}...")
        print(f"    Metadata: {doc.metadata}")
    print()


# ============================================================================
# 6. DOCX Loader
# ============================================================================

def example_docx_loader_unstructured(file_path: str = "sample.docx"):
    """
    Example: Load DOCX file using UnstructuredWordDocumentLoader.
    
    Args:
        file_path: Path to DOCX file
    """
    print("=" * 80)
    print("Example 8: DOCX Loader - UnstructuredWordDocumentLoader")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        print("  Create a DOCX file or update the file_path parameter\n")
        return
    
    # Create loader
    loader = UnstructuredWordDocumentLoader(file_path)
    
    # Load documents
    docs = loader.load()
    
    print(f"✓ Loaded {len(docs)} document(s) from DOCX")
    if docs:
        print(f"✓ Total characters: {len(docs[0].page_content)}")
        print(f"✓ Metadata: {docs[0].metadata}")
        print(f"\nFirst 200 characters:\n{docs[0].page_content[:200]}...\n")


def example_docx_loader_docx2txt(file_path: str = "sample.docx"):
    """
    Example: Load DOCX file using Docx2txtLoader (simpler alternative).
    
    Args:
        file_path: Path to DOCX file
    """
    print("=" * 80)
    print("Example 9: DOCX Loader - Docx2txtLoader")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        print("  Create a DOCX file or update the file_path parameter\n")
        return
    
    # Create loader
    loader = Docx2txtLoader(file_path)
    
    # Load documents
    docs = loader.load()
    
    print(f"✓ Loaded {len(docs)} document(s) from DOCX")
    if docs:
        print(f"✓ Total characters: {len(docs[0].page_content)}")
        print(f"✓ Metadata: {docs[0].metadata}")
        print(f"\nFirst 200 characters:\n{docs[0].page_content[:200]}...\n")


# ============================================================================
# 7. PPTX Loader
# ============================================================================

def example_pptx_loader(file_path: str = "sample.pptx"):
    """
    Example: Load PPTX file using UnstructuredPowerPointLoader.
    
    Args:
        file_path: Path to PPTX file
    """
    print("=" * 80)
    print("Example 10: PPTX Loader - UnstructuredPowerPointLoader")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        print("  Create a PPTX file or update the file_path parameter\n")
        return
    
    # Create loader
    loader = UnstructuredPowerPointLoader(file_path)
    
    # Load documents (each slide becomes a document)
    docs = loader.load()
    
    print(f"✓ Loaded {len(docs)} slide(s) from PPTX")
    if docs:
        print(f"✓ First slide characters: {len(docs[0].page_content)}")
        print(f"✓ Metadata: {docs[0].metadata}")
        print(f"\nFirst 200 characters:\n{docs[0].page_content[:200]}...\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAG Document Loaders - Comprehensive Examples")
    print("=" * 80 + "\n")
    
    # Web loaders (always work - no file needed)
    example_web_loader_cheerio()
    example_web_loader_webbase()
    
    # File-based loaders (require files)
    # Uncomment and provide file paths to test:
    
    # example_pdf_loader_pypdf("path/to/your/file.pdf")
    # example_pdf_loader_unstructured("path/to/your/file.pdf")
    # example_json_loader("sample.json")
    # example_jsonl_loader("sample.jsonl")
    # example_csv_loader("sample.csv")
    # example_docx_loader_unstructured("path/to/your/file.docx")
    # example_docx_loader_docx2txt("path/to/your/file.docx")
    # example_pptx_loader("path/to/your/file.pptx")
    
    print("=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nNote: File-based loaders require actual files.")
    print("JSON, JSONL, and CSV examples will create sample files if they don't exist.")
    print("For PDF, DOCX, and PPTX, provide your own files or update file paths.\n")
