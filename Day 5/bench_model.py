"""
Day 5, Part D:
Compare two Groq models on the same prompts.
"""

import os
import time
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Groq OpenAI-compatible API
BASE = "https://api.groq.com/openai/v1"

API_KEY = os.getenv("GROQ_API_KEY")

# Compare two models from your .env
MODELS = [
    os.getenv("MODEL", "openai/gpt-oss-120b"),
    os.getenv("MODEL2", "qwen/qwen3.8-27b")
]

PROMPTS = [
    "Reply with exactly: OK",
    "In two sentences, what is an AI agent?",
    "A course costs Rs. 18,000 with a 15% scholarship. What is payable? Show the steps.",
]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def run(model, prompt):
    """Send one prompt to one Groq model and measure performance."""

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
        timeout=600
    )

    response.raise_for_status()

    data = response.json()

    elapsed = time.time() - start

    # Groq/OpenAI response usage
    usage = data.get("usage", {})

    tokens = usage.get("completion_tokens", 0)

    rate = tokens / elapsed if elapsed else 0

    answer = data["choices"][0]["message"]["content"].strip()

    return elapsed, tokens, rate, answer


if __name__ == "__main__":

    print("=" * 72)
    print("DAY 5 PART D - GROQ MODEL COMPARISON")
    print("=" * 72)

    for model in MODELS:

        print("\n" + "=" * 72)
        print("MODEL:", model)
        print("=" * 72)

        for prompt in PROMPTS:

            elapsed, tokens, rate, text = run(model, prompt)

            print(f"\n  prompt: {prompt[:60]}")

            print(
                f"  {elapsed:5.1f} s | "
                f"{tokens:4d} tokens | "
                f"{rate:5.1f} tok/s"
            )

            print(f"  answer: {text[:160]}")

        print()

    print("=" * 72)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 72)