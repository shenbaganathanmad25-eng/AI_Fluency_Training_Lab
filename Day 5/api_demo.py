"""
Day 5, Part C:
Call the Groq REST API directly and measure what you get.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


# Groq API configuration
BASE = "https://api.groq.com/openai/v1"

API_KEY = os.getenv("GROQ_API_KEY")

MODEL = os.getenv("MODEL2", "qwen/qwen3.8-27b")

PROMPT = "In three sentences, explain what an AI agent is."


# Headers for Groq API
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def list_models():
    """GET /models - list models available through Groq."""

    response = requests.get(
        f"{BASE}/models",
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    models = response.json().get("data", [])

    print("\nModels available on Groq:")

    for model in models:
        print(f"   {model.get('id')}")


def generate_once(model=MODEL, prompt=PROMPT):
    """
    POST /chat/completions with stream=false.
    Measure total response time.
    """

    start = time.time()

    response = requests.post(
        f"{BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    elapsed = time.time() - start

    answer = data["choices"][0]["message"]["content"]

    usage = data.get("usage", {})

    tokens = usage.get("completion_tokens", 0)

    print("\n[Groq chat completion]")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Completion tokens: {tokens}")
    print(
        f"Speed: "
        f"{tokens / elapsed if elapsed else 0:.1f} tokens/s"
    )

    print("\nAnswer:")
    print(answer.strip()[:300])


def chat_streaming(model=MODEL, prompt=PROMPT):
    """
    POST /chat/completions with stream=true.
    Measure time to first token and total time.
    """

    start = time.time()

    first_token_at = None

    pieces = []

    response = requests.post(
        f"{BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "stream": True
        },
        stream=True,
        timeout=300
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        line = line.decode("utf-8")

        # Groq sends Server-Sent Events
        if line.startswith("data: "):
            line = line[6:]

        if line == "[DONE]":
            break

        chunk = json.loads(line)

        piece = (
            chunk
            .get("choices", [{}])[0]
            .get("delta", {})
            .get("content", "")
        )

        if piece and first_token_at is None:
            first_token_at = time.time() - start

        pieces.append(piece)

    total = time.time() - start

    text = "".join(pieces)

    print("\n[Groq streaming]")
    print(
        f"TTFT: "
        f"{first_token_at:.2f} seconds"
        if first_token_at is not None
        else "TTFT: Not received"
    )
    print(f"Total time: {total:.2f} seconds")
    print(f"Characters: {len(text)}")

    print("\nStreaming answer:")
    print(text.strip()[:300])


def openai_compatible(model=MODEL, prompt=PROMPT):
    """
    POST /chat/completions using the OpenAI-compatible Groq endpoint.
    """

    response = requests.post(
        f"{BASE}/chat/completions",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        },
        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    print("\n[Groq /chat/completions]")

    print(
        data["choices"][0]["message"]["content"]
        .strip()[:300]
    )


if __name__ == "__main__":

    print("======================================")
    print("Day 5 Part C - Groq REST API")
    print("======================================")

    print(f"\nUsing model: {MODEL}")

    # 1. List available Groq models
    list_models()

    # 2. Normal generation
    generate_once()

    # 3. Streaming generation
    chat_streaming()

    # 4. OpenAI-compatible request
    openai_compatible()

    print("\n======================================")
    print("Day 5 Part C completed")
    print("======================================")
