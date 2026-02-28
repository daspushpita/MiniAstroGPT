
def tool_call_schema() -> dict:
    GOOGLE_SEARCH_SCHEMA = {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "Search the web to define hard words for a glossary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The term to search for"},
                },
                "required": ["query"],
            }
        }
    }
    return GOOGLE_SEARCH_SCHEMA
