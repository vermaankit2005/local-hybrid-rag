from config.constants import OPENSEARCH_INDEX
from config.setup_opensearch import get_opensearch_client, get_embedding_model


class DocumentRetrieval:

    @staticmethod
    def hybrid_search(query, num_documents=3) -> list[str]:
        query_vector = get_embedding_model().embed_query(query)  # 👈 embed the QUERY too
        body = {
            "size": num_documents,
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"text": {"query": query}}},  # keyword arm
                        {"knn": {"embedding": {"vector": query_vector, "k": num_documents}}},  # vector arm
                    ]
                }
            },
        }
        resp = get_opensearch_client().search(
            index=OPENSEARCH_INDEX, body=body,
            params={"search_pipeline": "hybrid-pipeline"},
        )

        page_content = []

        for hit in resp["hits"]["hits"]:
            page_content.append(hit["_source"]["text"])


        return page_content


if __name__ == "__main__":
    DocumentRetrieval.hybrid_search(query="What is the history of Mickey Mouse?", num_documents=10)
