import logging
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.constants import OPENSEARCH_BASE_URL, OPENSEARCH_INDEX_CHUNKS, OPENSEARCH_INDEX_DOCS
from config.setup_opensearch import get_embedding_model, get_opensearch_client

load_dotenv()
logger = logging.getLogger(__name__)


class OpenSearchVectorStore:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    def __init__(self):
        self.embedding_model = None

    @staticmethod
    def ingest_documents(docs: list[Document]):
        """
        Ingests a list of full documents into the OpenSearch document index.
        :param docs: List of Document objects to be ingested.
        """
        client = get_opensearch_client()

        for doc in docs:
            doc_id = doc.metadata.get("doc_id", doc.metadata.get("pageid"))
            revision_id = doc.metadata.get("revision_id", doc.metadata.get("revision_id"))

            if doc_id is None:
                raise ValueError("Document metadata must include 'pageid'.")
            if revision_id is None:
                raise ValueError("Document metadata must include 'revision_id'.")

            payload = {
                "doc_id": str(doc_id),
                "revision_id": int(revision_id),
                "document": {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            }

            client.index(
                index=OPENSEARCH_INDEX_DOCS,
                id=str(doc_id),
                body=payload,
                refresh=False,
            )
        logger.info("Ingested %s full documents into OpenSearch index %s", len(docs), OPENSEARCH_INDEX_DOCS)


    def ingest_chunks(self, docs: list[Document]) -> VectorStoreRetriever:
        """
        Ingests a list of document chunks into the OpenSearch vector store.
        :param docs: List of Document objects to be ingested.
        :return: VectorStoreRetriever for querying the ingested document chunks.
        """

        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.")
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
            logger.info("Initialized OpenSearch embedding model")

        vector_store = OpenSearchVectorSearch.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            opensearch_url=OPENSEARCH_BASE_URL,
            index_name=OPENSEARCH_INDEX_CHUNKS,
            use_ssl=False,
            verify_certs=False,
            vector_field="embedding",  # matches Step 3 mapping
            text_field="text",
        )
        logger.info(f"Ingested {len(docs)} chunks into OpenSearch index {OPENSEARCH_INDEX_CHUNKS}")

        return vector_store.as_retriever(search_kwargs={"k": 3})  # Return a retriever for querying the vector store
