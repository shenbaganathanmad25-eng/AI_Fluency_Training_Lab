"""Day 7: a LangChain chat model built from the SAME .env file as Day 1."""
import os
from dotenv import load_dotenv

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "ollama").strip().lower()

def get_model(temperature=0):
    """Return a LangChain chat model for whichever provider .env selects."""
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("MODEL", "qwen2.5:1.5b"), temperature=temperature)

    from langchain_openai import ChatOpenAI
    if PROVIDER == "groq":
        return ChatOpenAI(base_url="https://api.groq.com/openai/v1",
                          api_key=os.getenv("GROQ_API_KEY"),
                          model=os.getenv("MODEL", "openai/gpt-oss-120b"),
                          temperature=temperature)
    if PROVIDER == "huggingface":
        return ChatOpenAI(base_url="https://router.huggingface.co/v1",
                          api_key=os.getenv("HF_TOKEN"),
                          model=os.getenv("MODEL", "openai/gpt-oss-20b"),
                          temperature=temperature)
    raise SystemExit(f"Unknown PROVIDER '{PROVIDER}'. Use ollama, groq or huggingface.")