import logging
import sys
from typing import Optional

import wikipedia
from langchain_core.documents import Document
from markdownify import markdownify
from wikipedia.exceptions import DisambiguationError, PageError, WikipediaException

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
            try:
                page = wikipedia.page(title, auto_suggest=False, preload=False)
            except DisambiguationError as exc:
                logger.warning(
                    "Skipping ambiguous Wikipedia result '%s' with %s options",
                    title,
                    len(exc.options),
                )
                continue
            except PageError:
                logger.warning("Skipping missing Wikipedia page '%s'", title)
                continue
            except WikipediaException as exc:
                logger.warning("Skipping Wikipedia page '%s' due to API error: %s", title, exc)
                continue

            page_content = markdownify(page.html(), heading_style="ATX")
            if not page_content.strip():
                logger.warning("Skipping Wikipedia page '%s' due to empty markdown content", title)
                continue

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

        if not documents:
            raise ValueError(
                f"No usable Wikipedia pages found for topic '{topic}'. "
                "Try a more specific topic."
            )
        logger.info("Loaded %s Wikipedia documents for topic '%s'", len(documents), topic)
        return documents
