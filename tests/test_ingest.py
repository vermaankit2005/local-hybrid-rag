import logging
import sys
import unittest

from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from config.constants import OPENSEARCH_INDEX_DOCS
from index.ingest import OpenSearchVectorStore

TEST_PAGE_ID = "test-integration-99999"


class TestFindUniqueDocumentsIntegration(unittest.TestCase):

    def setUp(self):
        self.store = OpenSearchVectorStore()
        self.client = self.store.client
        self._cleanup()
        # Seed an existing document with revision_id=1
        self.client.index(
            index=OPENSEARCH_INDEX_DOCS,
            id=TEST_PAGE_ID,
            body={
                "doc_id": TEST_PAGE_ID,
                "revision_id": 1,
                "document": {
                    "page_content": "Original content",
                    "metadata": {"pageid": TEST_PAGE_ID, "revision_id": 1},
                },
            },
            refresh=True,
        )

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        if self.client.exists(index=OPENSEARCH_INDEX_DOCS, id=TEST_PAGE_ID):
            self.client.delete(index=OPENSEARCH_INDEX_DOCS, id=TEST_PAGE_ID, refresh=True)

    def test_document_exists_with_changed_revision(self):
        """When a document exists but revision_id changed, the old doc should be deleted
        and the new doc returned in the unique list."""
        doc = Document(
            page_content="Updated content",
            metadata={"pageid": TEST_PAGE_ID, "revision_id": 2},
        )

        result = self.store.find_unique_documents([doc])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], doc)
        # The old document must have been removed from OpenSearch
        self.assertFalse(
            self.client.exists(index=OPENSEARCH_INDEX_DOCS, id=TEST_PAGE_ID),
            "Old document should have been deleted after revision change",
        )


if __name__ == "__main__":
    unittest.main()
