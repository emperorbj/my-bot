"""
Core RAG (Retrieval-Augmented Generation) logic: PDF loading, chunking,
indexing into Qdrant (hybrid dense+sparse search), and question answering.
"""
import uuid
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue

import config

# --- Singletons (loaded once at import time, reused across requests) ---
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_EMBEDDING_MODEL)
client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


# ---------------------------------------------------------------------
# Loading & chunking
# ---------------------------------------------------------------------
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document


def chunk_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)


# ---------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------
def get_embedding_dimension() -> int:
    return len(embeddings.embed_query("dimension test"))


def ensure_document_id_index():
    """
    Create a payload index on metadata.document_id so we can filter by it
    in scroll/search/retrieval. Safe to call repeatedly — Qdrant treats
    creating an index that already exists as a no-op.
    """
    client.create_payload_index(
        collection_name=config.COLLECTION_NAME,
        field_name="metadata.document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def create_collection_if_missing():
    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if config.COLLECTION_NAME not in existing:
        vector_size = get_embedding_dimension()
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    # Always ensure the index exists, whether the collection is new or
    # pre-existing (e.g. created before this index was added).
    ensure_document_id_index()


def _vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=config.COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


# ---------------------------------------------------------------------
# Indexing a new document
# ---------------------------------------------------------------------
def index_pdf(file_path: str, filename: str) -> Tuple[str, int, int]:
    """
    Loads, chunks, and indexes a PDF. Returns (document_id, num_pages, num_chunks).
    """
    create_collection_if_missing()

    document_id = str(uuid.uuid4())

    documents = load_pdf(file_path)
    chunks = chunk_docs(documents)

    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["original_filename"] = filename

    vector_store = _vector_store()
    vector_store.add_documents(chunks)

    return document_id, len(documents), len(chunks)


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------
def get_retriever(document_id: str, k: int = 5):
    vector_store = _vector_store()
    return vector_store.as_retriever(
        search_kwargs={
            "k": k,
            "filter": Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        }
    )


def search(query: str, document_id: str) -> List[dict]:
    retriever = get_retriever(document_id)
    results = retriever.invoke(query)

    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page")
        clean_text = doc.page_content[:200].replace("\n", " ").strip()
        formatted.append({"source": source, "page": page, "snippet": clean_text})

    return formatted


def document_exists(document_id: str) -> bool:
    """Check whether any chunks are indexed for this document_id."""
    results, _ = client.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
        limit=1,
    )
    return len(results) > 0


def list_documents() -> List[dict]:
    """Return all distinct documents (id + filename) currently indexed in Qdrant."""
    seen = {}
    offset = None
    limit = 200

    while True:
        results, next_offset = client.scroll(
            collection_name=config.COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )
        for point in results:
            metadata = point.payload.get("metadata", {})
            doc_id = metadata.get("document_id")
            filename = metadata.get("original_filename")
            if doc_id and doc_id not in seen:
                seen[doc_id] = filename
        if next_offset is None:
            break
        offset = next_offset

    return [
        {"document_id": doc_id, "filename": filename}
        for doc_id, filename in sorted(seen.items())
    ]


# ---------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(document_id: str):
    llm = ChatGroq(model=config.LLM_MODEL, temperature=0)
    retriever = get_retriever(document_id)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I could not find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask_question(question: str, document_id: str) -> str:
    chain = create_rag_chain(document_id)
    return chain.invoke(question)