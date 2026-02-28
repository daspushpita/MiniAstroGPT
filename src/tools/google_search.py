import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class GoogleSearchError(Exception):
    """Raised when the Google search tool cannot return usable results."""


def google_search(query: str) -> str:
    """
    Fetches organic search snippets from Serper.dev.
    Returns formatted context text on success.
    """
    url = "https://google.serper.dev/search"
    api_key = os.getenv("SERPER_API_KEY")
    
    payload = {"q": query}
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for HTTP errors
        data = response.json()
        results = data.get('organic', [])
        
        # 1. Format the first 4 snippets into a string for the LLM
        context = []
        for item in results[:4]:
            title = item.get('title', 'No Title')
            snippet = item.get('snippet', 'No Snippet')
            context.append(f"Source: {title}\nContent: {snippet}")
            
        return "\n---\n".join(context) if context else "No results found."
    
    except requests.RequestException as e:
        raise GoogleSearchError(f"Google Search request failed: {str(e)}") from e
    
# TEST IT MANUALLY FIRST!
if __name__ == "__main__":
    try:
        print(google_search("What is a Neural Radiance Field in AI?"))
    except GoogleSearchError as e:
        print(f"Tool error: {e}")
    
