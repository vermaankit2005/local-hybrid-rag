import logging
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.constants import OPENSEARCH_BASE_URL, OPENSEARCH_INDEX
from config.setup_opensearch import get_embedding_model

load_dotenv()
logger = logging.getLogger(__name__)


class OpenSearchVectorStore:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    def __init__(self):
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")

        self.embedding_model = get_embedding_model()
        logger.info("Initialized OpenSearch embedding model")

    def ingest_documents(self, docs: list[Document]) -> VectorStoreRetriever:
        """
        Ingests a list of documents into the OpenSearch vector store.
        :param documents: List of Document objects to be ingested.
        :return: VectorStoreRetriever for querying the ingested documents.
        """

        vector_store = OpenSearchVectorSearch.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            opensearch_url=OPENSEARCH_BASE_URL,
            index_name=OPENSEARCH_INDEX,
            use_ssl=False,
            verify_certs=False,
            vector_field="embedding",  # matches Step 3 mapping
            text_field="text",
        )
        logger.info(f"Ingested {len(docs)} chunks into OpenSearch index {OPENSEARCH_INDEX}")

        return vector_store.as_retriever(search_kwargs={"k": 3})  # Return a retriever for querying the vector store
