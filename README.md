# 🔍 Local Hybrid RAG

A local **Hybrid Retrieval-Augmented Generation (RAG)** system that combines **BM25 keyword search** and **semantic vector search** on top of a self-hosted [OpenSearch](https://opensearch.org/) instance. Documents are sourced from Wikipedia, embedded via OpenRouter, reranked with a cross-encoder, and answered by DeepSeek v4 flash.

---

## 🏗️ Architecture

### Ingestion

```
Wikipedia API
     │
     ▼
WikipediaDocumentLoader        ←── fetches & converts HTML → Markdown,
     │                             then strips [edit] links, images and
     │                             link URLs (keeping the link text)
     ▼
OpenSearchDocumentStore.find_unique_documents()
     │                          ←── skips unchanged pages, re-ingests
     │                              pages whose revision_id changed
     ▼
MarkdownDocumentTextSplitterHybrid
     │                          ←── pass 1: split on H1/H2/H3
     │                              pass 2: split each section to ~1000 chars
     │                              (150 overlap), prefixing every chunk with
     │                              its "H1 > H2 > H3" heading breadcrumb
     │
     ├──────────────► chunks index  (text for BM25 + knn_vector embedding)
     │                               text-embedding-3-small, 1536-dim
     │
     └──────────────► docs index    (full document snapshot + revision_id)
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
     │                             keyword arm matches text + metadata.title
     │                             returns LangChain Documents (text + metadata)
     ▼
flatten + deduplicate          ←── by metadata.chunk_id, keeps first occurrence
     │                             and preserves retrieval order
     ▼
OpenRouter reranker            ←── cohere/rerank-4-pro, keeps top 5
     │
     ▼
RAG chain (LangChain)          ←── DeepSeek v4 flash via OpenRouter
     │                             answers from context only, or says
     │                             it doesn't know; cites source URLs
     │                             when the context carries them
     ▼
  Answer
```

---

## ✨ Features

- **Hybrid search** — combines BM25 keyword and KNN vector search via OpenSearch's native `hybrid` query + normalisation pipeline; the keyword arm is a `multi_match` across chunk text *and* the page title, so title-worded questions still rank
- **Multi-query retrieval** — an LLM expands one question into several sub-questions, each retrieved separately, then deduplicated
- **Metadata all the way through** — retrieval returns LangChain `Document` objects, so title, source URL and heading breadcrumb survive reranking and reach the prompt; dedup keys off `chunk_id` instead of comparing raw text
- **Cross-encoder reranking** — `cohere/rerank-4-pro` via OpenRouter reorders the merged candidates before they reach the LLM
- **Swappable answering LLM** — DeepSeek v4 flash via OpenRouter, or `gpt-oss-120b` on Groq (`get_grok_llm()`) when you want a faster answer leg
- **Two-index design** — full documents live in a docs index, chunks in a vector index, so originals stay retrievable
- **Revision-aware ingestion** — unchanged Wikipedia pages are skipped; changed pages have their old document *and* chunks deleted before re-ingestion, so no duplicates build up
- **Heading-aware hybrid chunking** — chunks are split on Markdown headings *then* by size, and each one carries its `H1 > H2 > H3` breadcrumb inline, so an isolated chunk still says what section it came from
- **Wikipedia ingestion** — loads any Wikipedia topic and converts pages to clean Markdown, stripping `[edit]` links, images and URL noise that would otherwise be embedded
- **Fully local** — OpenSearch runs in Docker; no cloud search service needed
- **LangChain / LangGraph** — built on LangChain for easy chain composition

---

## 📁 Project Structure

```
local-hybrid-rag/
├── config/
│   ├── constants.py            # OpenSearch connection, index names, model constants
│   ├── setup_opensearch.py     # Client, index & hybrid-pipeline setup
│   └── utils.py                # Embedding model + LLM factories (OpenRouter, Groq)
├── index/
│   ├── document_loader.py      # WikipediaDocumentLoader + Markdown cleanup
│   ├── document_splitter.py    # MarkdownDocumentTextSplitterHybrid (headings + size)
│   ├── __document_splitter___.py  # superseded heading-only splitter, kept for reference
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
GROQ_API_KEY=gsk_...        # only needed if you use get_grok_llm()
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

### Chunking

Chunk sizing is set when constructing the splitter in `index/ingest_pipeline.py`:

```python
MarkdownDocumentTextSplitterHybrid(chunk_size=1000, chunk_overlap=150)
```

Headings split first, so a section shorter than `chunk_size` stays in one piece.

### Hybrid search weights

Defined in `config/setup_opensearch.py` (`_setup_hybrid_pipeline`):

```python
"weights": [0.4, 0.6]  # [BM25 keyword, semantic vector]
```

Adjust to tune the balance between keyword precision and semantic recall. Note that the pipeline is only created if it doesn't already exist — delete it before re-running to pick up new weights.

### Reranking

`retrieval/reranker_setup.py` calls OpenRouter's `/rerank` endpoint with `cohere/rerank-4-pro` and returns the top `top_n` (default 5) documents. It sends only `page_content` to the API, then maps the returned indices back to the original `Document` objects so metadata is never lost. `top_n` is clamped to the number of candidates, so a short candidate list won't fail the request.

### Answering LLM

`config/utils.py` exposes two factories:

| Factory | Model | Notes |
|---|---|---|
| `get_llm_model()` | `deepseek/deepseek-v4-flash` via OpenRouter | used for query expansion and the multi-query answer |
| `get_grok_llm()` | `openai/gpt-oss-120b` via Groq | used by `standard_search()`; reasoning hidden |

Both run at `temperature=0.4` to keep answers grounded.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Search & storage | [OpenSearch 2.13](https://opensearch.org/) |
| Embeddings | `text-embedding-3-small` via [OpenRouter](https://openrouter.ai/) |
| Reranking | `cohere/rerank-4-pro` via [OpenRouter](https://openrouter.ai/) |
| LLM | `deepseek-v4-flash` via [OpenRouter](https://openrouter.ai/), `gpt-oss-120b` via [Groq](https://groq.com/) |
| Orchestration | [LangChain](https://python.langchain.com/) / [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Data source | [Wikipedia](https://pypi.org/project/wikipedia/) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Containerisation | Docker Compose |

---

## 📄 License

MIT
