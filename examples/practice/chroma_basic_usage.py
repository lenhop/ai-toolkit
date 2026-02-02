"""
Chroma 工具包基础使用示例

演示 Chroma 向量数据库工具包的核心功能。

Author: AI Toolkit Team
Version: 1.0.0
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量
load_dotenv()

# 导入 Chroma 工具包
from ai_toolkit.chroma import (
    create_chroma_store,
    ChromaStore,
    create_chroma_client,
    get_or_create_collection,
)


def example_1_simple_usage():
    """示例1：最简单的使用方式（一步创建）"""
    print("\n" + "=" * 60)
    print("示例1：最简单的使用方式")
    print("=" * 60)
    
    # 创建测试文档
    documents = [
        Document(
            page_content="Amazon 退货政策：30天内可以退货",
            metadata={"platform": "amazon", "doc_type": "policy"}
        ),
        Document(
            page_content="eBay 退货政策：14天内可以退货",
            metadata={"platform": "ebay", "doc_type": "policy"}
        ),
    ]
    
    # 创建 Embeddings（使用 DashScope/Qwen）
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    # 一步创建 Chroma VectorStore
    store = create_chroma_store(
        documents=documents,
        embeddings=embeddings,
        collection_name="ecommerce_policies",
        persist_directory="./data/chroma"  # 持久化存储
    )
    
    print("✅ VectorStore 创建成功")
    
    # 搜索
    results = store.similarity_search("退货政策", k=2)
    print(f"✅ 搜索成功，找到 {len(results)} 个结果")
    for i, doc in enumerate(results, 1):
        print(f"  结果 {i}: {doc.page_content[:50]}...")
        print(f"  元数据: {doc.metadata}")


def example_2_advanced_usage():
    """示例2：高级使用（ChromaStore 类）"""
    print("\n" + "=" * 60)
    print("示例2：高级使用（ChromaStore 类）")
    print("=" * 60)
    
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    # 创建 ChromaStore（更灵活的控制）
    store = ChromaStore(
        collection_name="ecommerce_products",
        embeddings=embeddings,
        persist_directory="./data/chroma"
    )
    
    # 批量添加文档
    documents = [
        Document(
            page_content=f"产品 {i} 的描述信息",
            metadata={"platform": "amazon", "product_id": f"prod_{i}"}
        )
        for i in range(10)
    ]
    
    ids = store.batch_add(documents=documents, batch_size=5)
    print(f"✅ 批量添加成功，添加了 {len(ids)} 个文档")
    
    # 带元数据过滤的搜索
    results = store.search_with_filter(
        query="产品",
        filter={"platform": "amazon"},
        k=3
    )
    print(f"✅ 带过滤搜索成功，找到 {len(results)} 个结果")
    
    # 获取统计信息
    stats = store.get_stats()
    print(f"✅ 统计信息: {stats}")


def example_3_rag_integration():
    """示例3：与 RAG 模块集成"""
    print("\n" + "=" * 60)
    print("示例3：与 RAG 模块集成")
    print("=" * 60)
    
    from ai_toolkit.rag import (
        load_web_document,
        split_document_recursive,
        create_vector_store
    )
    
    # 1. 加载文档
    docs = load_web_document("https://www.example.com", selector="p")
    print(f"✅ 文档加载成功: {len(docs)} 个文档")
    
    # 2. 切分文档
    chunks = split_document_recursive(docs, chunk_size=500, chunk_overlap=50)
    print(f"✅ 文档切分成功: {len(chunks)} 个块")
    
    # 3. 创建 Embeddings
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v2",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    # 4. 通过 RAG 模块创建 Chroma 存储
    store = create_vector_store(
        documents=chunks,
        embeddings=embeddings,
        store_type="chroma",  # 使用 Chroma
        collection_name="web_docs",
        persist_directory="./data/chroma"
    )
    
    print("✅ Chroma VectorStore 创建成功（通过 RAG 模块）")
    
    # 5. 搜索
    results = store.similarity_search("example", k=3)
    print(f"✅ 搜索成功，找到 {len(results)} 个结果")


def example_4_client_management():
    """示例4：客户端管理"""
    print("\n" + "=" * 60)
    print("示例4：客户端管理")
    print("=" * 60)
    
    from ai_toolkit.chroma import create_chroma_client, list_collections
    
    # 创建客户端
    client = create_chroma_client(
        mode="persistent",
        persist_directory="./data/chroma"
    )
    print("✅ 客户端创建成功")
    
    # 列出所有集合
    collections = list_collections(client)
    print(f"✅ 集合列表: {collections}")
    
    # 获取或创建集合
    collection = get_or_create_collection(
        client=client,
        name="test_collection",
        metadata={"description": "Test collection"}
    )
    print(f"✅ 集合获取/创建成功: {collection.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Chroma 工具包使用示例")
    print("=" * 60)
    
    # 注意：需要配置 DASHSCOPE_API_KEY 环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️  警告: 未配置 DASHSCOPE_API_KEY，部分示例可能无法运行")
        print("   请设置环境变量或使用 FakeEmbeddings 进行测试")
    
    try:
        example_1_simple_usage()
    except Exception as e:
        print(f"❌ 示例1失败: {e}")
    
    try:
        example_2_advanced_usage()
    except Exception as e:
        print(f"❌ 示例2失败: {e}")
    
    try:
        example_3_rag_integration()
    except Exception as e:
        print(f"❌ 示例3失败: {e}")
    
    try:
        example_4_client_management()
    except Exception as e:
        print(f"❌ 示例4失败: {e}")
    
    print("\n" + "=" * 60)
    print("示例执行完成")
    print("=" * 60)
