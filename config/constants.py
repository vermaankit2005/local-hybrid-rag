from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_BASE_URL = "http://localhost:9200"  # Base URL for the OpenSearch instance
OPENSEARCH_HOST = "localhost"  # Hostname for the OpenSearch instance
OPENSEARCH_PORT = 9200  # Port number for OpenSearch
OPENSEARCH_INDEX_CHUNKS = "mickey_mouse_wiki_articles_v1"  # Index name for storing document chunks in OpenSearch
OPENSEARCH_INDEX_DOCS = "mickey_mouse_wiki_articles_docs_v1"  # Index name for storing original documents in OpenSearch

OPENSEARCH_USERNAME = "admin"  # Username for OpenSearch authentication
OPENSEARCH_PASSWORD = "MyStrongPass123!"  # Password # API key for OpenRouter
OPENROUTER_EMBEDDING_DIMENSION = "3072"  # Dimension of the embedding vector for the chosen model



TEXT_EMBEDDING_LARGE = "openai/text-embedding-3-large"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"