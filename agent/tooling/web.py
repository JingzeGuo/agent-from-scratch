"""Network tools: fetch a known URL and search the web through Tavily."""

import json
import os

import httpx

# Returns plain text. We don't parse HTML here — that's a separate concern.

_FETCH_TIMEOUT_SECONDS = 10.0
_MAX_FETCH_CHARS = 20_000  # truncate long pages


def fetch_url(url: str) -> str:
    """Fetch the raw content of a URL.

    Truncates response to 20,000 characters to avoid context overflow.
    """
    with httpx.Client(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    text = response.text
    if len(text) > _MAX_FETCH_CHARS:
        text = (
            text[:_MAX_FETCH_CHARS]
            + f"\n\n[... truncated, total {len(response.text)} chars ...]"
        )
    return text


# Tavily Search API: https://tavily.com
# - Free tier: 1000 queries/month, no credit card required
# - Designed for LLM/agent use cases (snippets are pre-optimized)
# - Uses POST + api_key in body (different from Brave's header-based auth)


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily Search API.

    Returns a JSON string with title/url/snippet for each result.
    Requires TAVILY_API_KEY in environment.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com"
        )

    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",  # basic | advanced (advanced costs more credits)
            },
        )
        response.raise_for_status()

    data = response.json()
    results = data.get("results", [])[:max_results]
    simplified = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),  # Tavily uses "content", not "description"
        }
        for r in results
    ]
    return json.dumps(simplified, ensure_ascii=False, indent=2)
