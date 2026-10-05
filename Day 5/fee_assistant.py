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
You are a college fee assistant for the Department of AI and Data Science.
Never guess a fee: say that you need to look it up.
Answer in one short sentence, and never use more than 30 words.
"""

response = client.chat.completions.create(
    model=MODEL,
    temperature=0,
    max_tokens=50,
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "What is the fee for CS101?"
        }
    ]
)

print(response.choices[0].message.content)
