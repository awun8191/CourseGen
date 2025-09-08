import os

class GeminiApiKeys:
    """Provides a list of Gemini API keys."""

    def __init__(self, api_keys: list[str] = None):
        if api_keys is None:
            # Load keys from environment variables or a config file

            self.api_keys = [
                 
                 "AIzaSyDYi5YiD1jU2r5wdPHJZBaU81m0l1XZumE",
                "AIzaSyCP6igfyX0FTLiWxN0os50nvN748gn6YiA",
                 "AIzaSyCdgFdcdHjFNrII6HSwZ3jaI_TqB12K-UM", 
                 "AIzaSyCvSKg6UP3sdnSkjcVSE6EuzjPgKQVijbE", 
                 "AIzaSyCvSKg6UP3sdnSkjcVSE6EuzjPgKQVijbE",
                 ]
        else:
            self.api_keys = api_keys

    def get_keys(self) -> list[str]:
        """Returns the list of API keys."""
        return self.api_keys
