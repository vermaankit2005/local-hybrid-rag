import logging
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from opensearchpy import OpenSearch

from config.constants import OPENSEARCH_HOST, OPENSEARCH_PORT, OPENSEARCH_INDEX, OPENSEARCH_USERNAME, \
    OPENSEARCH_PASSWORD, OPENROUTER_EMBEDDING_DIMENSION, OPENROUTER_BASE_URL, TEXT_EMBEDDING_LARGE

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
        model="openai/gpt-4o-mini",
        openai_api_key=OPENROUTER_API_KEY,
    )


def _setup_opensearch_index():
    """
    Sets up the OpenSearch index, if it doesn't already exist.
    Check both text search and vector search are enabled for the index.

    :return: A tuple containing the OpenSearch client and the index name.
    """
    client = get_opensearch_client()

    index_name = OPENSEARCH_INDEX
    index_body = {
        "settings": {
            "index.knn": True,  # 👈 turns ON vector search for this index
        },
        "mappings": {
            "properties": {
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

    if not client.indices.exists(index=index_name):
        try:
            client.indices.create(index=index_name, body=index_body)
            print("Created OpenSearch index '%s'" % index_name)
        except Exception:
            print("Failed creating OpenSearch index '%s'" % index_name)
            raise
    else:
        print("OpenSearch index '%s' already exists" % index_name)

    return client, index_name


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

    # result = client.transport.perform_request(
    #     "DELETE", "/_search/pipeline/hybrid-pipeline"
    # )
    #
    # print("Deleted existing hybrid pipeline: %s", result)

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
    _setup_opensearch_index()
    _setup_hybrid_pipeline()
