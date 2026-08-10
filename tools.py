def calculate_product(a: int, b: int) -> int:
    """Calculates the product of two integers."""
    return a * b


def search_web(query: str, max_results: int = 3) -> str:
    """
    Searches the web using DuckDuckGo and returns formatted results.

    Each result includes: title, URL, and a short snippet.
    Returns an error message string on failure so the LLM can handle it gracefully.
    """
    from ddgs import DDGS

    if not query or not query.strip():
        return "Error: search query cannot be empty."

    try:
        max_results = int(max_results)
        if max_results < 1:
            max_results = 1
        elif max_results > 10:
            max_results = 10
    except (TypeError, ValueError):
        max_results = 3

    try:
        results = list(DDGS().text(query.strip(), max_results=max_results))
    except Exception as e:
        return f"Error: search failed — {str(e)}"

    if not results:
        return "No results found for the given query."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        url = r.get("href", "No URL")
        snippet = r.get("body", "No snippet available.")
        formatted.append(f"[Result {i}]\nTitle: {title}\nURL: {url}\nSnippet: {snippet}")

    return "\n\n".join(formatted)
