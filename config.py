import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Where uploaded PDFs are temporarily stored before/after indexing
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)