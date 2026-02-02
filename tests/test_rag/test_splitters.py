"""
Test RAG Splitters

Tests for ai_toolkit.rag.splitters module:
- split_document_recursive
- split_with_overlap
- split_for_chinese

Based on examples: rag_03_split_document.py, rag_04_agent_workflow.py
"""

import os
import sys
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
        split_document_recursive,
        split_with_overlap,
        split_for_chinese,
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_OK = False


def test_split_document_recursive():
    """Test split_document_recursive function."""
    print("\n" + "=" * 80)
    print("Test 1: split_document_recursive")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Load test document
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        docs = load_web_document(url)
        
        print(f"✓ Loaded {len(docs)} document(s)")
        
        # Test: Split with default parameters
        chunks = split_document_recursive(
            docs,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        print(f"✓ Split into {len(chunks)} chunks")
        print(f"✓ Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
        print(f"✓ First chunk preview: {chunks[0].page_content[:150]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_split_with_overlap():
    """Test split_with_overlap function."""
    print("\n" + "=" * 80)
    print("Test 2: split_with_overlap")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Load test document
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        docs = load_web_document(url)
        
        # Test: Split with custom separators
        chunks = split_with_overlap(
            docs,
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        print(f"✓ Split into {len(chunks)} chunks")
        print(f"✓ Chunk size: {len(chunks[0].page_content)} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_split_for_chinese():
    """Test split_for_chinese function."""
    print("\n" + "=" * 80)
    print("Test 3: split_for_chinese")
    print("=" * 80)
    
    if not IMPORTS_OK:
        print("⚠ Skipping test - imports failed")
        return
    
    try:
        # Load test document (using markdown file if available)
        test_file = "/Users/hzz/Downloads/big_data_route.md"
        if os.path.exists(test_file):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(test_file, encoding='utf-8')
            docs = loader.load()
            
            print(f"✓ Loaded {len(docs)} document(s)")
            
            # Test: Split for Chinese
            chunks = split_for_chinese(
                docs,
                chunk_size=1000,
                chunk_overlap=200
            )
            
            print(f"✓ Split into {len(chunks)} chunks")
            print(f"✓ Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
            print(f"✓ First chunk preview: {chunks[0].page_content[:150]}...")
        else:
            print(f"⚠ Test file not found: {test_file}")
            print("  Skipping Chinese splitter test")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing RAG Splitters")
    print("=" * 80)
    
    test_split_document_recursive()
    test_split_with_overlap()
    test_split_for_chinese()
    
    print("\n" + "=" * 80)
    print("RAG Splitter Tests Completed")
    print("=" * 80)
