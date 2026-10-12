"""Session 8 helpers: chat model, embedding model and vector store, all chosen from .env."""
import hashlib
import math
import os
import re

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()
PROVIDER = os.getenv("PROVIDER", "ollama").strip().lower()
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama" if PROVIDER == "ollama" else "fastembed").strip().lower()
DB_DIR = "chroma_db"           # folder where Chroma saves the vectors on disk
COLLECTION = "college_helpdesk"


def get_model(temperature=0):
    """Return a LangChain chat model for whichever provider .env selects (same as Day 7)."""
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("MODEL", "qwen2.5:1.5b"), temperature=temperature)

    from langchain_openai import ChatOpenAI
    if PROVIDER == "groq":
        return ChatOpenAI(base_url="https://api.groq.com/openai/v1",
                          api_key=os.getenv("GROQ_API_KEY"),
                          model=os.getenv("MODEL", "openai/gpt-oss-20b"),
                          temperature=temperature)
    if PROVIDER == "huggingface":
        return ChatOpenAI(base_url="https://router.huggingface.co/v1",
                          api_key=os.getenv("HF_TOKEN"),
                          model=os.getenv("MODEL", "openai/gpt-oss-20b"),
                          temperature=temperature)
    raise SystemExit(f"Unknown PROVIDER '{PROVIDER}'. Use ollama, groq or huggingface.")


class HashEmbeddings(Embeddings):
    """Offline fallback: a keyword 'bag of words' turned into a 256-number vector.
    It needs no download, but it only matches WORDS, not MEANING. Use it only
    when neither Ollama nor FastEmbed is available."""
    STOP = set("a an the is are was to of in on for and or by at be it this that with what how do does i my can".split())

    def _vec(self, text):
        v = [0.0] * 256
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w in self.STOP:
                continue
            w = w[:-1] if w.endswith("s") and len(w) > 3 else w      # crude: fees -> fee
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embeddings():
    """Return the embedding model. All three options are open source and run on your laptop."""
    if EMBED_PROVIDER == "ollama":            # needs: ollama pull nomic-embed-text
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"))
    if EMBED_PROVIDER == "fastembed":         # downloads a ~70 MB ONNX model once
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    if EMBED_PROVIDER == "hash":
        return HashEmbeddings()
    raise SystemExit(f"Unknown EMBED_PROVIDER '{EMBED_PROVIDER}'. Use ollama, fastembed or hash.")


def get_vectorstore():
    """Open (or create) the Chroma collection saved in ./chroma_db."""
    from langchain_chroma import Chroma
    return Chroma(collection_name=COLLECTION,
                  embedding_function=get_embeddings(),
                  persist_directory=DB_DIR,
                  collection_metadata={"hnsw:space": "cosine"})   # score = cosine distance