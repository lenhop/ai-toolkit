"""
Test RAG Loaders

Tests for ai_toolkit.rag.loaders module:
- load_web_document
- load_pdf_document
- load_json_document
- load_csv_document

Based on examples: rag_01_loader_base.py, rag_02_website_loader.py
"""

import os
import sys
import tempfile
import json
import csv
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

# Test imports
try:
    from ai_toolkit.rag import (
        load_web_document,
        load_pdf_document,
        load_json_document,
        load_csv_document,
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_load_web_document():
    """Test load_web_document function."""
    print("\n" + "=" * 80)
    print("Test 1: load_web_document")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Test: Load entire page
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        docs = load_web_document(url)
        
        print(f"✓ Loaded {len(docs)} document(s)")
        print(f"✓ Total characters: {len(docs[0].page_content)}")
        print(f"✓ Metadata: {docs[0].metadata}")
        
        # Test: Load with selector
        docs_with_selector = load_web_document(url, selector="p")
        
        print(f"✓ Loaded with selector: {len(docs_with_selector)} document(s)")
        print(f"✓ Content reduction: {len(docs[0].page_content) - len(docs_with_selector[0].page_content)} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_load_json_document():
    """Test load_json_document function."""
    print("\n" + "=" * 80)
    print("Test 2: load_json_document")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "title": "Test Document",
                "content": "This is a test JSON document for RAG testing.",
                "author": "AI Toolkit",
                "date": "2024-01-01"
            }, f)
            temp_path = f.name
        
        try:
            # Test: Load entire JSON
            docs = load_json_document(temp_path)
            print(f"✓ Loaded {len(docs)} document(s)")
            print(f"✓ Content: {docs[0].page_content[:100]}...")
            
            # Test: Load with jq_schema
            docs_filtered = load_json_document(temp_path, jq_schema=".content")
            print(f"✓ Loaded with filter: {len(docs_filtered)} document(s)")
            print(f"✓ Filtered content: {docs_filtered[0].page_content[:100]}...")
            
        finally:
            os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_load_csv_document():
    """Test load_csv_document function."""
    print("\n" + "=" * 80)
    print("Test 3: load_csv_document")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "title", "content"])
            writer.writerow(["1", "Document 1", "This is the first document content."])
            writer.writerow(["2", "Document 2", "This is the second document content."])
            writer.writerow(["3", "Document 3", "This is the third document content."])
            temp_path = f.name
        
        try:
            # Test: Load CSV
            docs = load_csv_document(temp_path)
            print(f"✓ Loaded {len(docs)} row(s)")
            
            for i, doc in enumerate(docs[:3], 1):
                print(f"  Row {i}: {len(doc.page_content)} chars")
                print(f"    Content: {doc.page_content[:80]}...")
                print(f"    Metadata: {doc.metadata}")
            
        finally:
            os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_load_pdf_document():
    """Test load_pdf_document function."""
    print("\n" + "=" * 80)
    print("Test 4: load_pdf_document")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    # Check if test PDF exists
    test_pdf = "/Users/hzz/Downloads/big_data_route.md"  # Not a PDF, but we'll check
    pdf_path = test_pdf.replace('.md', '.pdf')
    
    if not os.path.exists(pdf_path):
        print(f"⚠ PDF file not found: {pdf_path}")
        print("  Skipping PDF test - create a PDF file to test")
        return
    
    try:
        # Test: Load PDF with PyPDF
        docs = load_pdf_document(pdf_path, loader_type="pypdf")
        print(f"✓ Loaded {len(docs)} page(s) from PDF")
        if docs:
            print(f"✓ First page: {len(docs[0].page_content)} chars")
            print(f"✓ Metadata: {docs[0].metadata}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("  Note: PDF loading requires pypdf or unstructured package")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing RAG Loaders")
    print("=" * 80)
    
    test_load_web_document()
    test_load_json_document()
    test_load_csv_document()
    test_load_pdf_document()
    
    print("\n" + "=" * 80)
    print("RAG Loader Tests Completed")
    print("=" * 80)
