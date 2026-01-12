import requests
from requests.exceptions import Timeout, RequestException

def ollama_generate(
    prompt: str,
    model: str = "phi3:mini",   # lighter + faster by default
    temperature: float = 0.2,
    base_url: str = "http://localhost:11434",
    timeout: int = 300,         # increased timeout for local models
) -> str:
    """
    Calls local Ollama server to generate text.

    - Uses a lightweight local model by default
    - Prevents app freezing with timeout handling
    - Returns a safe message if Ollama is slow or unavailable
    """

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                },
            },
            timeout=timeout,
        )

        response.raise_for_status()
        data = response.json()

        # Defensive check
        if "response" not in data or not data["response"]:
            return "⚠️ Local AI did not return a response."

        return data["response"].strip()

    except Timeout:
        return (
            "⚠️ Local AI summarization timed out on this machine. "
            "The system will fall back to showing the most relevant document excerpts."
        )

    except RequestException as e:
        return (
            "⚠️ Unable to reach the local AI service (Ollama). "
            "Please ensure Ollama is running."
        )
