"""Azure AI Search keyword retrieval over the policy index.

Plain REST via httpx (already a transitive dependency) — no azure-search SDK needed
for a simple-query search. The index is populated by the Azure portal's
"connect to data" blob indexer, so documents carry `id`, `title` (blob name)
and `content` (full extracted text).
"""
import httpx
from pydantic import BaseModel


class SearchError(RuntimeError):
    pass


class SearchDocument(BaseModel):
    id: str
    title: str
    score: float
    content: str
    highlights: list[str]


class AzureSearchClient:
    def __init__(self, endpoint: str, index: str, api_key: str, *, api_version: str = "2024-07-01",
                 timeout: float = 20.0, transport: httpx.BaseTransport | None = None):
        if not (endpoint and index and api_key):
            raise ValueError("Azure AI Search endpoint, index and api key are all required")
        self.url = f"{endpoint.rstrip('/')}/indexes/{index}/docs/search"
        self.params = {"api-version": api_version}
        self.http = httpx.Client(headers={"api-key": api_key}, timeout=timeout, transport=transport)

    def search(self, query: str, *, top_k: int = 4) -> list[SearchDocument]:
        body = {
            "search": query,
            "top": top_k,
            "select": "id,title,content",
            "queryType": "simple",
            "searchMode": "any",
            "highlight": "content",
            "highlightPreTag": "",
            "highlightPostTag": "",
        }
        try:
            resp = self.http.post(self.url, params=self.params, json=body)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPStatusError as exc:
            raise SearchError(f"Azure AI Search returned {exc.response.status_code}: {exc.response.text[:300]}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"Azure AI Search request failed: {exc}") from exc
        return [_parse_hit(hit) for hit in payload.get("value", [])]

    def close(self) -> None:
        self.http.close()


def _parse_hit(hit: dict) -> SearchDocument:
    return SearchDocument(
        id=str(hit.get("id") or ""),
        title=str(hit.get("title") or "untitled"),
        score=float(hit.get("@search.score") or 0.0),
        content=(hit.get("content") or "").strip(),
        highlights=[h.strip() for h in (hit.get("@search.highlights") or {}).get("content", []) if h.strip()],
    )
