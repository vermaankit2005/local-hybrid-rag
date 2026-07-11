# 🔍 Local Hybrid RAG

A local **Hybrid Retrieval-Augmented Generation (RAG)** system that combines **BM25 keyword search** and **semantic vector search** on top of a self-hosted [OpenSearch](https://opensearch.org/) instance. Documents are sourced from Wikipedia, embedded via OpenRouter, and answers are generated using DeepSeek v4 flash.

---

## 🏗️ Architecture

```
Wikipedia API
     │
     ▼
WikipediaDocumentLoader   ←── fetches & converts HTML → Markdown
     │
     ▼
MarkdownDocumentTextSplitter  ←── splits by Markdown headings + size
     │
     ▼
OpenSearchVectorStore         ←── embeds + stores (BM25 + knn_vector)
     │                                   (text-embedding-3-large via OpenRouter)
     ▼
 OpenSearch (Docker)
     │
     ▼
DocumentRetrieval.hybrid_search()   ←── 40% BM25  +  60% semantic
     │                                   (min-max normalised, arithmetic mean)
     ▼
RAG Chain (LangChain)               ←── DeepSeek v4 flash via OpenRouter
     │
     ▼
  Answer (Markdown)
```

---

## ✨ Features

- **Hybrid search** — combines BM25 keyword and KNN vector search via OpenSearch's native `hybrid` query + normalisation pipeline
- **Wikipedia ingestion** — loads any Wikipedia topic, converts pages to clean Markdown, and chunks by heading structure
- **OpenRouter embeddings** — uses `text-embedding-3-large` (3072-dim) via the OpenRouter API
- **LLM answering** — DeepSeek v4 flash answers questions strictly from retrieved context
- **Fully local** — OpenSearch runs in Docker; no cloud search service needed
- **LangChain / LangGraph** — built on LangChain for easy chain composition

---

## 📁 Project Structure

```
local-hybrid-rag/
├── config/
│   ├── constants.py          # OpenSearch connection + model constants
│   └── setup_opensearch.py   # Client, embedding model, LLM, index & pipeline setup
├── index/
│   ├── document_loader.py    # WikipediaDocumentLoader
│   ├── document_splitter.py  # MarkdownDocumentTextSplitter
│   ├── ingest.py             # OpenSearchVectorStore (embed + index)
│   └── ingest_pipeline.py    # End-to-end ingestion script
├── retrieval/
│   ├── document_retrieval.py # DocumentRetrieval.hybrid_search()
│   └── retrieval_pipeline.py # RAG chain (retrieve → prompt → LLM)
├── docker-compose.yml        # OpenSearch single-node dev setup
├── main.py
└── pyproject.toml
```

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) (or pip)
- Docker & Docker Compose

### 1. Clone the repo

```bash
git clone https://github.com/vermaankit2005/local-hybrid-rag.git
cd local-hybrid-rag
```

### 2. Install dependencies

```bash
uv sync
# or: pip install -e .
```

### 3. Set up environment variables

Create a `.env` file in the project root (never commit this):

```env
OPENROUTER_API_KEY=sk-or-...
```

### 4. Start OpenSearch

```bash
docker-compose up -d
```

OpenSearch will be available at `http://localhost:9200`.

### 5. Create the index & hybrid pipeline

```bash
python config/setup_opensearch.py
```

### 6. Ingest Wikipedia documents

```bash
cd index
python ingest_pipeline.py
```

This loads up to 10 Wikipedia articles on *Mickey Mouse* (configurable), chunks them, embeds each chunk, and indexes everything into OpenSearch.

### 7. Run the RAG pipeline

```bash
cd retrieval
python retrieval_pipeline.py
```

Or call it programmatically:

```python
from retrieval.retrieval_pipeline import query

answer = query("Who created Mickey Mouse and when?")
print(answer)
```

---

### Hybrid search weights

Defined in `config/setup_opensearch.py` (`_setup_hybrid_pipeline`):

```python
"weights": [0.4, 0.6]  # [BM25 keyword, semantic vector]
```

Adjust to tune the balance between keyword precision and semantic recall.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Search & storage | [OpenSearch 2.13](https://opensearch.org/) |
| Embeddings | `text-embedding-3-large` via [OpenRouter](https://openrouter.ai/) |
| LLM | `deepseek-v4-flash` via [OpenRouter](https://openrouter.ai/) |
| Orchestration | [LangChain](https://python.langchain.com/) / [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Data source | [Wikipedia](https://pypi.org/project/wikipedia/) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Containerisation | Docker Compose |

---

## 📄 License

MIT
