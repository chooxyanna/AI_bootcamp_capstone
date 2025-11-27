# app/rag/indexer.py
import os, shutil
from collections import defaultdict
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from ..config import DOC_DIR, INDEX_DIR

def _load_docs_safe(doc_dir: str):
    """Load docs; return [] if dir exists but is empty (avoid ValueError)."""
    if not os.path.isdir(doc_dir):
        os.makedirs(doc_dir, exist_ok=True)
        return []
    try:
        return SimpleDirectoryReader(doc_dir).load_data()
    except ValueError as e:
        # LlamaIndex raises ValueError("No files found in ...") on empty dirs
        if "No files found" in str(e):
            return []
        raise

def build_or_load_index(doc_dir: str = str(DOC_DIR),
                        index_dir: str = str(INDEX_DIR)) -> VectorStoreIndex:
    # Try load existing index
    if os.path.exists(index_dir):
        try:
            storage = StorageContext.from_defaults(persist_dir=index_dir)
            return load_index_from_storage(storage)
        except Exception:
            shutil.rmtree(index_dir, ignore_errors=True)  # rebuild cleanly

    # Fresh build
    documents = _load_docs_safe(doc_dir)

    if not documents:
        # Build a tiny placeholder index so the app can run before docs arrive
        placeholder = Document(text="(no RAG documents yet)", metadata={"file_name": "EMPTY"})
        index = VectorStoreIndex.from_documents([placeholder])
        index.storage_context.persist(persist_dir=index_dir)
        return index

    # Merge pages by file
    file_to_pages = defaultdict(list)
    for d in documents:
        fname = d.metadata.get("file_name") or d.metadata.get("filename") or "unknown_source"
        file_to_pages[fname].append(d.text)

    merged_docs = [
        Document(text="\n".join(pages), metadata={"file_name": fn})
        for fn, pages in file_to_pages.items()
    ]

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=32)
    nodes = splitter.get_nodes_from_documents(merged_docs)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=index_dir)
    return index

# --- singletons ---
_INDEX = build_or_load_index()
QUERY_ENGINE = _INDEX.as_query_engine(similarity_top_k=5)

def format_sources(resp):
    out = []
    for n in getattr(resp, "source_nodes", []) or []:
        out.append({
            "file": n.metadata.get("file_name", "unknown"),
            "score": getattr(n, "score", None),
            "preview": (n.text[:240] + "…") if n.text else ""
        })
    return out
