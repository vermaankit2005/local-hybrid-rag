import logging
import sys

from document_loader import WikipediaDocumentLoader
from document_splitter import MarkdownDocumentTextSplitter
from ingest import OpenSearchVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting ingestion pipeline")

    # Load documents based on the pipeline configuration
    documents = WikipediaDocumentLoader.load_wiki_documents(
        topic="Mickey Mouse",
        max_docs=25,
    )
    logger.info("Loaded %s documents from Wikipedia", len(documents))

    # Split documents into chunks
    splitter = MarkdownDocumentTextSplitter()

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_documents(doc)
        all_chunks.extend(chunks)
    logger.info("Prepared %s chunks for ingestion", len(all_chunks))

    # Ingest chunks into the vector store
    vector_store = OpenSearchVectorStore()
    vector_store.ingest_documents(documents)
    retriever = vector_store.ingest_chunks(all_chunks)
    logger.info("Ingestion completed successfully")

except Exception:
    logger.exception("Ingestion pipeline failed")
