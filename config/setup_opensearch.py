import logging
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from opensearchpy import OpenSearch

from config.constants import OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX_CHUNKS, OPENSEARCH_USERNAME, \
    OPENSEARCH_PASSWORD, OPENROUTER_EMBEDDING_DIMENSION, OPENROUTER_BASE_URL, TEXT_EMBEDDING_LARGE, \
    OPENSEARCH_INDEX_DOCS

logger = logging.getLogger(__name__)

client = None  # Global variable to hold the OpenSearch client instance

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def get_opensearch_client() -> OpenSearch:
    """
    Returns an OpenSearch client instance.

    :return: An OpenSearch client.
    """
    global client
    if client is None:
        client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            use_ssl=False,
            verify_certs=False
        )
    return client


def get_embedding_model():
    """
    Returns an instance of OpenAIEmbeddings configured with the OpenRouter API key and model.
    :return: An instance of OpenAIEmbeddings.
    """
    return OpenAIEmbeddings(
        openai_api_base=OPENROUTER_BASE_URL,
        model=TEXT_EMBEDDING_LARGE,
        openai_api_key=OPENROUTER_API_KEY
    )


def get_llm_model() -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_base=OPENROUTER_BASE_URL,
        model="deepseek/deepseek-v4-flash",
        openai_api_key=OPENROUTER_API_KEY,
    )


def _setup_doc_and_chunk_opensearch_index():
    """
    Sets up the OpenSearch index, if it doesn't already exist.
    Check both text search and vector search are enabled for the index.

    :return: A tuple containing the OpenSearch client and the index name.
    """
    client = get_opensearch_client()

    doc_index_name = OPENSEARCH_INDEX_DOCS
    doc_index_body = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},  # your custom document id
                "revision_id": {"type": "integer"},  # revision id at document level
                "document": {  # full LangChain document snapshot
                    "properties": {
                        "page_content": {"type": "text"},
                        "metadata": {"type": "object", "dynamic": True}
                    }
                }
            }
        }
    }

    chunk_index_name = OPENSEARCH_INDEX_CHUNKS
    chunk_index_body = {
        "settings": {
            "index.knn": True,  # 👈 turns ON vector search for this index
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},  # unique identifier for each document
                "text": {"type": "text"},  # for BM25 keyword search
                "embedding": {  # for semantic search
                    "type": "knn_vector",
                    "dimension": int(OPENROUTER_EMBEDDING_DIMENSION),  # 👈 MUST match your embedding model
                    "method": {
                        "name": "hnsw",  # fast approximate nearest neighbor
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    }
                },
                "metadata": {
                    "properties": {
                        "title": {"type": "text"},
                        "pageid": {"type": "integer"},
                        "source": {"type": "keyword"},
                        "page_url": {"type": "keyword"},
                        "revision_id": {"type": "integer"},
                        "chunk_id": {"type": "keyword"},
                    }
                },  # for storing metadata
            }
        }
    }

    # Setting up the OpenSearch documnent index if it doesn't already exist
    if not client.indices.exists(index=doc_index_name):
        try:
            client.indices.create(index=doc_index_name, body=doc_index_body)
            print("Created OpenSearch document index '%s'" % doc_index_name)
        except Exception:
            print("Failed creating OpenSearch document index '%s'" % doc_index_name)
            raise
    else:
        print("OpenSearch document index '%s' already exists" % doc_index_name)

    # Setting up the OpenSearch chunk index if it doesn't already exist
    if not client.indices.exists(index=chunk_index_name):
        try:
            client.indices.create(index=chunk_index_name, body=chunk_index_body)
            print("Created OpenSearch chunk index '%s'" % chunk_index_name)
        except Exception:
            print("Failed creating OpenSearch index '%s'" % chunk_index_name)
            raise
    else:
        print("OpenSearch chunk index '%s' already exists" % chunk_index_name)

    return client, chunk_index_name


def _setup_hybrid_pipeline():
    pipeline = {
        "description": "hybrid normalization",
        "phase_results_processors": [{
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {"weights": [0.4, 0.6]},  # [keyword, semantic]
                },
            }
        }],
    }

    client = get_opensearch_client()

    hybrid_pipeline_exists = True
    try:
        result = client.transport.perform_request(
            "GET", "/_search/pipeline/hybrid-pipeline"
        )
        print("Hybrid pipeline already exists: %s" % result)
    except Exception:
        hybrid_pipeline_exists = False

    if not hybrid_pipeline_exists:
        print("Hybrid pipeline does not exist, creating it.")
        client.transport.perform_request(
            "PUT", "/_search/pipeline/hybrid-pipeline", body=pipeline
        )


if __name__ == "__main__":
    _setup_doc_and_chunk_opensearch_index()
    _setup_hybrid_pipeline()
