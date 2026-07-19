import logging
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config.constants import OPENSEARCH_BASE_URL, OPENSEARCH_INDEX_CHUNKS, OPENSEARCH_INDEX_DOCS
from config.setup_opensearch import get_opensearch_client
from config.utils import get_embedding_model

load_dotenv()
logger = logging.getLogger(__name__)


class OpenSearchDocumentStore:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    def __init__(self):
        self.embedding_model = None
        self.client = get_opensearch_client()

    def _delete_document_and_chunks(self, doc_id: str):
        """
        Delete a document and its associated chunks from OpenSearch.
        :param doc_id: The ID of the document to be deleted.
        """
        self.client.delete_by_query(
            index=OPENSEARCH_INDEX_DOCS,
            body={"query": {"match": {"doc_id": doc_id}}},
        )
        self.client.delete_by_query(
            index=OPENSEARCH_INDEX_CHUNKS,
            body={"query": {"match": {"metadata.pageid": doc_id}}},
        )

    def ingest_documents(self, docs: list[Document]):
        """
        Ingests a list of full documents into the OpenSearch document index.
        :param docs: List of Document objects to be ingested.
        """

        for doc in docs:
            doc_id = doc.metadata.get("pageid")
            revision_id = doc.metadata.get("revision_id")

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

            self.client.index(
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
            bulk_size=3000,  # Adjust based on your needs
        )
        logger.info(f"Ingested {len(docs)} chunks into OpenSearch index {OPENSEARCH_INDEX_CHUNKS}")

        return vector_store.as_retriever(search_kwargs={"k": 3})  # Return a retriever for querying the vector store

    def find_unique_documents(self, docs: list[Document]) -> list[Document]:
        """
        Check if the document exist,
        if "NO" then add it to unique_doc_list,
        if "YES" then we need to check
            the revision_id is same from as existing one,
            if "YES" then skip it,
            if "NO" then delete the existing document and all its chunks and add the document to unique_doc_list
        :param docs: List of Document objects to be deduplicated.
        :return: List of unique Document objects.
        """
        unique_doc_list = []

        for doc in docs:
            new_doc_id = doc.metadata.get("pageid")
            new_revision_id = doc.metadata.get("revision_id")

            if new_doc_id is None:
                raise ValueError("Document must contain metadata.pageid")
            if new_revision_id is None:
                raise ValueError("Document must contain metadata.revision_id")

            # Check if the document already exists in OpenSearch
            if not self.client.exists(index=OPENSEARCH_INDEX_DOCS, id=str(new_doc_id)):
                logger.info(f"Document with pageid {new_doc_id} does not exist. Adding to unique_doc_list.")
                unique_doc_list.append(doc)
            else:
                existing_doc = self.client.get(index=OPENSEARCH_INDEX_DOCS, id=str(new_doc_id))
                existing_revision_id = existing_doc["_source"]["revision_id"]
                if existing_revision_id == new_revision_id:
                    logger.info(
                        f"Document with pageid {new_doc_id} and revision_id {new_revision_id} already exists. Skipping.")
                    continue
                else:
                    logger.info(
                        f"Document with pageid {new_doc_id} exists but has a different revision_id. Deleting existing document and its chunks.")
                    self._delete_document_and_chunks(new_doc_id)
                    unique_doc_list.append(doc)

        return unique_doc_list
