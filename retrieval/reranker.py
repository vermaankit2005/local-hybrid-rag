import os

from dotenv import load_dotenv
from langchain_core.documents import Document

from retrieval.reranker_setup import OpenRouterReranker

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def rerank_documents(documents: list[Document], query: str) -> list[Document]:
    """
    Rerank the documents based on their relevance to the query.

    Args:
        documents (list[str]): A list of documents to be reranked.
        query (str): The query string used for reranking.

    Returns:
        list[str]: A list of reranked documents.
    """
    reranker = OpenRouterReranker(OPENROUTER_API_KEY)

    results = reranker.rerank(
        query,
        documents,
        top_n=5
    )

    docs = [documents[r["index"]] for r in results]
    print("\n".join(f"{i + 1}. {doc.page_content[:200]}" for i, doc in enumerate(docs)))
    return docs


if __name__ == "__main__":
    # Example usage
    documents = [
        Document(page_content="AI is transforming the world in many ways, from healthcare to finance.", metadata={"id": "1"}),
        Document(page_content="Recent research in AI focuses on improving natural language processing and computer vision.", metadata={"id": "2"}),
        Document(page_content="AI research is also exploring reinforcement learning and generative models.", metadata={"id": "3"}),
        Document(page_content="The field of AI is rapidly evolving, with new breakthroughs happening frequently.", metadata={"id": "4"}),
        Document(page_content="Recent research in AI focuses on improving natural language processing and computer vision.", metadata={"id": "5"}),
        Document(page_content="AI research is also exploring reinforcement learning and generative models.", metadata={"id": "6"}),
        Document(page_content="The field of AI is rapidly evolving, with new breakthroughs happening frequently.", metadata={"id": "7"}),
    ]
    query = "What research are happening in AI?"
    reranked_docs = rerank_documents(documents, query)
    print("Reranked Documents:", reranked_docs)
