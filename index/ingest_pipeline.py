import logging
import sys

from config.constants import RAG_TOPIC, MAX_DOCUMENTS
from index.document_loader import WikipediaDocumentLoader
from index.document_splitter import MarkdownDocumentTextSplitterHybrid
from index.opensearch_ingest import OpenSearchDocumentStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting ingestion pipeline")

    vector_store = OpenSearchDocumentStore()
    # Load documents based on the pipeline configuration
    documents = WikipediaDocumentLoader.load_wiki_documents(
        topic=RAG_TOPIC,
        max_docs=MAX_DOCUMENTS,
    )
    logger.info("Loaded %s documents from Wikipedia", len(documents))

    unique_documents = vector_store.find_unique_documents(documents)

    if not unique_documents:
        logger.error("No unique documents found for ingestion")
    else:
        logger.info("Found %s unique documents for ingestion", len(unique_documents))
        logger.info("%s duplicate documents will be skipped", len(documents) - len(unique_documents))
        # Split documents into chunks
        splitter = MarkdownDocumentTextSplitterHybrid()
        all_chunks = []

        for doc in unique_documents:
            chunks = splitter.split_documents(doc)
            all_chunks.extend(chunks)

        logger.info("Prepared %s chunks for ingestion", len(all_chunks))

        # Ingest chunks into the vector store
        retriever = vector_store.ingest_chunks(all_chunks)
        vector_store.ingest_documents(unique_documents)
        logger.info("Ingestion completed successfully")

except Exception:
    logger.exception("Ingestion pipeline failed")
