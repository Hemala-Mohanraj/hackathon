from duckduckgo_search import DDGS

def search_web(query):

    results = DDGS().text(query, max_results=3)

    text = ""

    for r in results:
        text += f"Title: {r['title']}\n"
        text += f"Body: {r['body']}\n\n"

    return text