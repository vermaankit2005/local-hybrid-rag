import logging

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


class MarkdownDocumentTextSplitter(MarkdownHeaderTextSplitter):
    """A text splitter that splits Markdown documents based on headers."""

    def __init__(self):
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

    def split_documents(self, document: Document) -> list[Document]:
        output = []

        for i, doc_chunk in enumerate(self.splitter.split_text(document.page_content)):
            doc_chunk.metadata = document.metadata.copy()  # Preserve original metadata
            doc_chunk.metadata["chunk_id"] = f"{document.metadata['pageid']}_{i}"  # Add chunk ID
            output.append(doc_chunk)

        logger.info(
            "Split document pageid=%s into %s chunks",
            document.metadata.get("pageid"),
            len(output),
        )
        return output