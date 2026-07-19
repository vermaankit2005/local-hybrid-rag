import os

from dotenv import load_dotenv

from retrieval.reranker_setup import OpenRouterReranker

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def rerank_documents(documents: list[str], query: str) -> list[str]:
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
    )

    docs = [documents[r["index"]] for r in results]
    print("\n".join(f"{i + 1}. {doc[:200]}" for i, doc in enumerate(docs)))
    return docs


if __name__ == "__main__":
    # Example usage
    documents = [
        "Document 1: AI has made significant strides in natural language processing, enabling machines to understand and generate human language.",
        "Document 2: Recent advancements in AI include the development of more efficient neural network architectures and improved training techniques.",
        "Document 3: AI is being applied in various fields such as healthcare, finance, and autonomous vehicles, leading to innovative solutions.",
        "Document 4: Ethical considerations in AI development are becoming increasingly important, with a focus on fairness, transparency, and accountability.",
        "Document 5: The integration of AI with other technologies like IoT and blockchain is opening up new possibilities for smart systems.",
        "Document 6: AI research is exploring areas like reinforcement learning, generative models, and explainable AI to enhance capabilities and trustworthiness.",
        "Document 7: The use of AI in creative fields such as art, music, and literature is challenging traditional notions of creativity and authorship.",
        "Document 8: AI-powered tools are being developed to assist in education, providing personalized learning experiences and adaptive assessments.",
        "Document 9: The global AI market is expected to grow significantly in the coming years, driven by increased adoption across industries and advancements in technology.",
        "Document 10: Collaboration between humans and AI is leading to new forms of human-computer interaction, where AI systems augment human capabilities and decision-making."

    ]
    query = "What research are happening in AI?"
    reranked_docs = rerank_documents(documents, query)
    print("Reranked Documents:", reranked_docs)
