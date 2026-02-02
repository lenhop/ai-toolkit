"""
Simple Web Loader Example - Direct Python Translation

This is a simpler, more direct translation of the JavaScript CheerioWebBaseLoader code.
"""

from bs4 import BeautifulSoup
from langchain_core.documents import Document
import requests


def load_web_with_selector(url: str, selector: str = None):
    """
    Load web page content with optional CSS selector filtering.
    
    Args:
        url: URL to load
        selector: CSS selector to extract specific elements (e.g., "p" for paragraphs)
    
    Returns:
        List of Document objects (always returns single document)
    """
    # Fetch HTML content
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    html_content = response.text
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Apply selector if provided
    if selector:
        selected_elements = soup.select(selector)
        # Extract text from selected elements
        text_content = "\n".join([elem.get_text(strip=True) for elem in selected_elements])
    else:
        # Extract all text if no selector
        text_content = soup.get_text(strip=True)
    
    # Create metadata
    metadata = {
        "source": url,
        "selector": selector
    }
    
    # Create and return document
    return [Document(page_content=text_content, metadata=metadata)]


# Example usage - equivalent to the JavaScript code
if __name__ == "__main__":
    # Equivalent to: const pTagSelector = "p"
    p_tag_selector = "p"
    
    # Equivalent to: new CheerioWebBaseLoader(url, { selector: pTagSelector })
    # Direct equivalent:
    docs = load_web_with_selector(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        selector=p_tag_selector
    )
    
    # Equivalent to: console.assert(docs.length === 1)
    assert len(docs) == 1, f"Expected 1 document, got {len(docs)}"
    
    # Equivalent to: console.log(`Total characters: ${docs[0].pageContent.length}`)
    print(f"Total characters: {len(docs[0].page_content)}")
    print(f"Document metadata: {docs[0].metadata}")
    print(f"\nFirst 500 characters of content:\n{docs[0].page_content[:500]}...")
