"""Step 1: load the handbook, split it into chunks, embed the chunks, save them in Chroma."""
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lc_config import DB_DIR, EMBED_PROVIDER, get_vectorstore

# 1. LOAD: one Document per file, remembering which file it came from
docs = []
for path in sorted(Path("data").glob("*.md")):
    docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                         metadata={"source": path.name}))
print(f"Loaded {len(docs)} documents")

# 2. SPLIT: small overlapping chunks so each vector holds one idea
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks (chunk_size=300, overlap=50)")
print("Example chunk ->", repr(chunks[4].page_content[:120]), chunks[4].metadata)

# 3. EMBED + STORE: start fresh each run so chunks are not added twice
shutil.rmtree(DB_DIR, ignore_errors=True)
store = get_vectorstore()
ids = store.add_documents(chunks)
print(f"Stored {len(ids)} vectors in ./{DB_DIR} using EMBED_PROVIDER={EMBED_PROVIDER}")
print("Vector length:", len(store.embeddings.embed_query("test")))