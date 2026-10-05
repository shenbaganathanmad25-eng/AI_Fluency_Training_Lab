import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

model = os.getenv("MODEL2", "qwen/qwen3.8-27b")

reply = client.chat.completions.create(
    model=model,
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": "You are a pirate. Answer in pirate speech."
        },
        {
            "role": "user",
            "content": "What is the fee for AI202?"
        }
    ]
)

print(reply.choices[0].message.content)
