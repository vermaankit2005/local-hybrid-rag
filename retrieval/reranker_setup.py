import httpx
from langchain_core.documents import Document


class OpenRouterReranker:
    RERANK_METADATA_FIELDS = ("title", "source", "h1", "h2", "h3")

    def __init__(self, api_key: str):
        self.client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {api_key}"},
        )


    def rerank(self, query: str, docs: list[Document], top_n: int = 5):
        r = self.client.post(
            "/rerank",
            json={
                "model": "cohere/rerank-4-pro",
                "query": query,
                "documents": [d.page_content for d in docs],
                "top_n": min(top_n, len(docs)),
            },
        )
        r.raise_for_status()
        return r.json()["results"]
