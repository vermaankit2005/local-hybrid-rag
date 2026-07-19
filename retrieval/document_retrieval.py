from config.constants import OPENSEARCH_INDEX_CHUNKS
from config.setup_opensearch import get_opensearch_client
from config.utils import get_embedding_model


class OpenSearchDocumentRetrieval:

    @staticmethod
    def hybrid_search(query, num_documents=10) -> list[str]:
        query_vector = get_embedding_model().embed_query(query)  # 👈 embed the QUERY too
        body = {
            "size": num_documents,
            "query": {
                "hybrid": {
                    "queries": [
                        {"multi_match": {"query": query, "fields": ["text", "metadata.title"]}},  # keyword arm
                        {"knn": {"embedding": {"vector": query_vector, "k": num_documents}}},  # vector arm
                    ]
                }
            },
        }
        resp = get_opensearch_client().search(
            index=OPENSEARCH_INDEX_CHUNKS, body=body,
            params={"search_pipeline": "hybrid-pipeline"},
        )

        page_content = []

        for hit in resp["hits"]["hits"]:
            page_content.append(hit["_source"]["text"])

        print(f"Retrieved {len(page_content)} documents from OpenSearch.")

        for i, content in enumerate(page_content):
            print("\n" + "*" * 25 + f" Document {i + 1} " + "*" * 25 + "\n")
            print(f"Document {i + 1} content: {content[:500]}...")  # Print first 500 characters of each document

        return page_content
