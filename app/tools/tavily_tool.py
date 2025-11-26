import os, requests
from langchain_core.tools import tool
from app.config import TAVILY_API_KEY

@tool
def tavily_search(query: str) -> str:
    """Search web with Tavily and return an answer string (may be brief)."""
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
            json={"query": query, "search_depth": "basic", "include_answer": True, "max_results": 5},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json().get("answer") or "No relevant online info found."
    except Exception as e:
        return f"[tavily_error] {e}"
