import httpx

class OpenRouterReranker:
    def __init__(self, api_key: str):
        self.client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def rerank(self, query: str, docs: list[str], top_n: int = 5):
        r = self.client.post(
            "/rerank",
            json={
                "model": "cohere/rerank-4-pro",
                "query": query,
                "documents": docs,
                "top_n": top_n,
            },
        )
        r.raise_for_status()
        return r.json()["results"]