# Chroma Vector Database Toolkit

> 专业级 Chroma 向量数据库工具包，为 RAG 和 Agent 项目提供简洁易用的接口

## 简介

Chroma Toolkit 是专为 RAG 和 Agent 项目设计的 Chroma 向量数据库工具包。提供简洁的 API，隐藏底层复杂性，让开发者专注于业务逻辑。

## 设计原则

- **简洁优先**：3个核心文件，覆盖90%使用场景
- **实用导向**：基于实际 RAG/Agent 项目经验
- **易于使用**：提供高级封装，降低使用门槛
- **灵活扩展**：支持高级场景，但不增加复杂度

## 快速开始

### 安装依赖

```bash
pip install chromadb langchain-community
```

### 基础使用（最简单）

```python
from ai_toolkit.chroma import create_chroma_store
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# 创建文档
documents = [
    Document(page_content="文档内容", metadata={"source": "test"})
]

# 创建 Embeddings
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key="your-api-key"
)

# 一步创建 VectorStore
store = create_chroma_store(
    documents=documents,
    embeddings=embeddings,
    collection_name="my_docs",
    persist_directory="./data/chroma"
)

# 搜索
results = store.similarity_search("查询", k=5)
```

### 高级使用（灵活控制）

```python
from ai_toolkit.chroma import ChromaStore

# 创建 ChromaStore
store = ChromaStore(
    collection_name="docs",
    embeddings=embeddings,
    persist_directory="./data/chroma"
)

# 批量添加
store.batch_add(documents=chunks, batch_size=100)

# 带过滤搜索
results = store.search_with_filter(
    query="product",
    filter={"platform": "amazon"},
    k=5
)
```

## 核心功能

### 1. 客户端管理 (`client.py`)

```python
from ai_toolkit.chroma import create_chroma_client, get_or_create_collection

# 创建客户端
client = create_chroma_client(
    mode="persistent",  # memory | persistent | http
    persist_directory="./data/chroma"
)

# 获取或创建集合
collection = get_or_create_collection(
    client=client,
    name="ecommerce_docs"
)
```

### 2. VectorStore 封装 (`store.py`)

**快速创建**：
```python
from ai_toolkit.chroma import create_chroma_store

store = create_chroma_store(
    documents=chunks,
    embeddings=embeddings,
    collection_name="docs",
    persist_directory="./data/chroma"
)
```

**高级控制**：
```python
from ai_toolkit.chroma import ChromaStore

store = ChromaStore(
    collection_name="docs",
    embeddings=embeddings,
    persist_directory="./data/chroma"
)

# 批量操作
store.batch_add(documents, batch_size=100)

# 带过滤搜索
results = store.search_with_filter(
    query="query",
    filter={"platform": "amazon"},
    k=5
)

# 获取统计
stats = store.get_stats()
```

### 3. 工具函数 (`utils.py`)

```python
from ai_toolkit.chroma import (
    validate_collection_name,
    generate_document_ids,
    validate_metadata
)

# 验证集合名称
validate_collection_name("ecommerce_docs")

# 生成文档 ID
ids = generate_document_ids(10, prefix="chunk")

# 验证元数据
validate_metadata({"platform": "amazon", "count": 10})
```

## 与 RAG 模块集成

```python
from ai_toolkit.rag import (
    load_web_document,
    split_document_recursive,
    create_vector_store
)

# 加载和切分文档
docs = load_web_document("https://example.com")
chunks = split_document_recursive(docs, chunk_size=1000)

# 创建 Chroma VectorStore（通过 RAG 模块）
store = create_vector_store(
    documents=chunks,
    embeddings=embeddings,
    store_type="chroma",  # 使用 Chroma
    collection_name="web_docs",
    persist_directory="./data/chroma"
)
```

## API 参考

### create_chroma_store()

一步创建 Chroma VectorStore。

**参数**：
- `documents`: 文档列表（可选）
- `embeddings`: Embeddings 模型（必需）
- `collection_name`: 集合名称（默认: "default"）
- `persist_directory`: 持久化目录（None 则内存模式）
- `client`: 自定义客户端（可选）

**返回**：LangChain Chroma VectorStore 实例

### ChromaStore 类

高级封装类，提供更多控制。

**主要方法**：
- `add_documents()`: 添加文档
- `batch_add()`: 批量添加（性能优化）
- `search()`: 相似度搜索
- `search_with_score()`: 带分数搜索
- `search_with_filter()`: 带过滤搜索
- `delete_documents()`: 删除文档
- `get_stats()`: 获取统计信息
- `clear()`: 清空集合

## 使用场景

### IC-RAG-Agent 典型场景

```python
from ai_toolkit.chroma import create_chroma_store
from ai_toolkit.rag import load_web_document, split_document_recursive

# 1. 加载平台文档
docs = load_web_document("https://amazon.com/policy")

# 2. 切分文档
chunks = split_document_recursive(docs, chunk_size=1000)

# 3. 添加元数据
for chunk in chunks:
    chunk.metadata.update({
        "platform": "amazon",
        "doc_type": "policy"
    })

# 4. 创建 Chroma 存储
store = create_chroma_store(
    documents=chunks,
    embeddings=embeddings,
    collection_name="amazon_policies",
    persist_directory="./data/chroma"
)

# 5. 检索（带过滤）
results = store.similarity_search(
    query="return policy",
    filter={"platform": "amazon"},
    k=3
)
```

## 注意事项

1. **NumPy 版本兼容性**：
   - ChromaDB 0.4.x 需要 NumPy < 2.0.0
   - 已在 requirements.txt 中约束

2. **持久化目录**：
   - 生产环境建议使用持久化模式
   - 确保目录有写入权限

3. **集合名称**：
   - 只能包含小写字母、数字、下划线、连字符
   - 不能以数字开头

## 版本信息

- **版本**: 1.0.0
- **ChromaDB**: >= 0.4.0
- **LangChain**: >= 0.1.0
- **Python**: >= 3.11

---

**文档版本**: v1.0  
**最后更新**: 2025-01-23
