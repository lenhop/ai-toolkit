from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4

# 1. Load the document
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer("p")
    }
)

docs = loader.load()
print(f"✓ Loaded {len(docs)} document(s)")
print(f"✓ Total characters: {len(docs[0].page_content)}")

# 2. Split the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,  # track index in original document
)

all_splits = splitter.split_documents(docs)
print(f"\n✓ Split blog post into {len(all_splits)} sub-documents.")

# 3. Inspect the splits
print(f"\nFirst split preview:")
print(f"  Length: {len(all_splits[0].page_content)} characters")
print(f"  Content: {all_splits[0].page_content[:200]}...")
