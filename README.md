# 🔍 Local Hybrid RAG

A local **Hybrid Retrieval-Augmented Generation (RAG)** system that combines **BM25 keyword search** and **semantic vector search** on top of a self-hosted [OpenSearch](https://opensearch.org/) instance. Documents are sourced from Wikipedia, embedded via OpenRouter, reranked with a cross-encoder, and answered by DeepSeek v4 flash.

---

## 🏗️ Architecture

### Ingestion

```
Wikipedia API
     │
     ▼
WikipediaDocumentLoader        ←── fetches & converts HTML → Markdown
     │
     ▼
OpenSearchDocumentStore.find_unique_documents()
     │                          ←── skips unchanged pages, re-ingests
     │                              pages whose revision_id changed
     ▼
MarkdownDocumentTextSplitter   ←── splits by Markdown headings + size
     │
     ├──────────────► docs index    (full document snapshot + revision_id)
     │
     └──────────────► chunks index  (text for BM25 + knn_vector embedding)
                                     text-embedding-3-small, 1536-dim
```

### Retrieval

```
User question
     │
     ▼
LLM query expansion            ←── generates 3–5 sub-questions (multi-query)
     │
     ▼
hybrid_search() per sub-query  ←── 40% BM25 + 60% semantic
     │                             (min-max normalised, arithmetic mean)
     ▼
flatten + deduplicate
     │
     ▼
OpenRouter reranker            ←── cohere/rerank-4-pro, keeps top-N
     │
     ▼
RAG chain (LangChain)          ←── DeepSeek v4 flash via OpenRouter
     │
     ▼
  Answer (Markdown)
```

---

## ✨ Features

- **Hybrid search** — combines BM25 keyword and KNN vector search via OpenSearch's native `hybrid` query + normalisation pipeline
- **Multi-query retrieval** — an LLM expands one question into several sub-questions, each retrieved separately, then deduplicated
- **Cross-encoder reranking** — `cohere/rerank-4-pro` via OpenRouter reorders the merged candidates before they reach the LLM
- **Two-index design** — full documents live in a docs index, chunks in a vector index, so originals stay retrievable
- **Revision-aware ingestion** — unchanged Wikipedia pages are skipped; changed pages have their old document *and* chunks deleted before re-ingestion, so no duplicates build up
- **Wikipedia ingestion** — loads any Wikipedia topic, converts pages to clean Markdown, and chunks by heading structure
- **Fully local** — OpenSearch runs in Docker; no cloud search service needed
- **LangChain / LangGraph** — built on LangChain for easy chain composition

---

## 📁 Project Structure

```
local-hybrid-rag/
├── config/
│   ├── constants.py            # OpenSearch connection, index names, model constants
│   ├── setup_opensearch.py     # Client, index & hybrid-pipeline setup
│   └── utils.py                # Embedding model + LLM factories
├── index/
│   ├── document_loader.py      # WikipediaDocumentLoader
│   ├── document_splitter.py    # MarkdownDocumentTextSplitter
│   ├── opensearch_ingest.py    # OpenSearchDocumentStore (dedupe, index docs + chunks)
│   └── ingest_pipeline.py      # End-to-end ingestion script
├── retrieval/
│   ├── document_retrieval.py   # OpenSearchDocumentRetrieval.hybrid_search()
│   ├── reranker_setup.py       # OpenRouterReranker HTTP client
│   ├── reranker.py             # rerank_documents()
│   └── retrieval_pipeline.py   # RAG chains (standard + multi-query)
├── tests/
│   └── test_ingest.py          # Integration tests for revision-aware dedup
├── opensearch_visualization.ipynb  # Notebook for inspecting the indices
├── docker-compose.yml          # OpenSearch single-node dev setup
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

### 5. Create the indices & hybrid pipeline

```bash
python -m config.setup_opensearch
```

Creates the docs index, the KNN-enabled chunks index, and the `hybrid-pipeline` search pipeline. Safe to re-run — existing resources are left alone.

### 6. Ingest Wikipedia documents

```bash
python -m index.ingest_pipeline
```

Loads up to 25 Wikipedia articles on *Mickey Mouse and Friends* (configurable in `config/constants.py`), skips pages already indexed at the same revision, chunks the rest, embeds each chunk, and indexes everything.

### 7. Run the RAG pipeline

```bash
python -m retrieval.retrieval_pipeline
```

Or call it programmatically:

```python
from retrieval.retrieval_pipeline import standard_search, multi_query_search

# single-query hybrid retrieval
print(standard_search("Who created Mickey Mouse and when?"))

# query expansion + dedup + rerank
print(multi_query_search("Tell me about Mickey Mouse?"))
```

### Run the tests

The tests are integration tests — OpenSearch must be running and the indices created.

```bash
python -m unittest discover tests
```

---

## ⚙️ Configuration

All knobs live in `config/constants.py`:

| Constant | Purpose |
|---|---|
| `RAG_TOPIC` | Wikipedia topic to ingest |
| `MAX_DOCUMENTS` | Max articles to pull from Wikipedia |
| `OPENSEARCH_INDEX_DOCS` | Index holding full document snapshots |
| `OPENSEARCH_INDEX_CHUNKS` | Index holding chunks + embeddings |
| `TEXT_EMBEDDING_LARGE` | Embedding model (`openai/text-embedding-3-small`) |
| `OPENROUTER_EMBEDDING_DIMENSION` | Must match the embedding model (1536) |
| `LLM_MODEL` | Answering model (`deepseek/deepseek-v4-flash`) |

### Hybrid search weights

Defined in `config/setup_opensearch.py` (`_setup_hybrid_pipeline`):

```python
"weights": [0.4, 0.6]  # [BM25 keyword, semantic vector]
```

Adjust to tune the balance between keyword precision and semantic recall. Note that the pipeline is only created if it doesn't already exist — delete it before re-running to pick up new weights.

### Reranking

`retrieval/reranker_setup.py` calls OpenRouter's `/rerank` endpoint with `cohere/rerank-4-pro` and returns the top `top_n` (default 5) documents.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Search & storage | [OpenSearch 2.13](https://opensearch.org/) |
| Embeddings | `text-embedding-3-small` via [OpenRouter](https://openrouter.ai/) |
| Reranking | `cohere/rerank-4-pro` via [OpenRouter](https://openrouter.ai/) |
| LLM | `deepseek-v4-flash` via [OpenRouter](https://openrouter.ai/) |
| Orchestration | [LangChain](https://python.langchain.com/) / [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Data source | [Wikipedia](https://pypi.org/project/wikipedia/) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Containerisation | Docker Compose |

---

## 📄 License

MIT
