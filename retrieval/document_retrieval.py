from langchain_core.documents import Document

from config.constants import OPENSEARCH_INDEX_CHUNKS
from config.setup_opensearch import get_opensearch_client
from config.utils import get_embedding_model


class OpenSearchDocumentRetrieval:

    @staticmethod
    def hybrid_search(query, num_documents=10) -> list[Document]:
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

        doc_list = []

        for hit in resp["hits"]["hits"]:
            doc = Document(
                page_content=hit["_source"]["text"],
                metadata=hit["_source"]["metadata"]
            )
            doc_list.append(doc)

        print(f"Retrieved {len(doc_list)} documents from OpenSearch.")

        for i, doc in enumerate(doc_list):
            print("\n" + "*" * 25 + f" Document {i + 1} " + "*" * 25 + "\n")
            print(f"Document {i + 1} content: {doc.page_content[:300]}... \nmetadata: {doc.metadata}")  # Print first 300 characters of each document

        return doc_list

if __name__ == "__main__":
    # Example usage
    query = "Who is mickey mouse?"
    retrieved_docs = OpenSearchDocumentRetrieval.hybrid_search(query, num_documents=5)
    # print(f"Retrieved documents: {retrieved_docs}")
