import logging
import sys

import wikipedia
from altair.utils import Optional
from langchain_core.documents import Document
from markdownify import markdownify

sys.stdout.reconfigure(encoding="utf-8")
logger = logging.getLogger(__name__)


class WikipediaDocumentLoader():

    @staticmethod
    def load_wiki_documents(topic: str, max_docs: Optional[int] = 25) -> list[Document]:
        """
        Load documents from Wikipedia and return them as a list of Documents.

        Args:
            topic (str): The topic to search for on Wikipedia.
            max_docs (int): The maximum number of documents to retrieve.

        Returns:
            list[Document]: Documents containing Markdown content and Wikipedia metadata.
        """

        wikipedia.set_lang("en")
        wikipedia.set_user_agent("rag-from-scratch/0.1 (example@gmail.com)")
        logger.info("Loading up to %s Wikipedia documents for topic '%s'", max_docs, topic)

        documents = []
        for title in wikipedia.search(topic, results=max_docs):
            page = wikipedia.page(title, auto_suggest=False, preload=True)
            page_content = markdownify(page.html(), heading_style="ATX")
            if not page_content.strip():
                raise ValueError(f"Could not convert Wikipedia page '{title}' to Markdown.")
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "title": page.title,
                        "pageid": page.pageid,
                        "source": page.url,
                        "page_url": page.url,
                        "revision_id": page.revision_id
                    },
                )
            )
        logger.info("Loaded %s Wikipedia documents for topic '%s'", len(documents), topic)
        return documents
