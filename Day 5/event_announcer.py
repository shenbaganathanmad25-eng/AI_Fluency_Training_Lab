import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = os.getenv("MODEL2", "qwen/qwen3.8-27b")

SYSTEM_PROMPT = """
You are an enthusiastic college events announcer.
Reply in an excited tone, with at least three exclamation marks.
"""

response = client.chat.completions.create(
    model=MODEL,
    temperature=1.2,
    max_tokens=100,
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "Announce the upcoming AI and Data Science college event."
        }
    ]
)

print(response.choices[0].message.content)