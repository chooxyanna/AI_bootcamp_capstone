# app/rag/indexer.py
import os, shutil, json, hashlib, time
from collections import defaultdict
from typing import Dict, Any, List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from ..config import RAG_DIR, INDEX_DIR

MANIFEST_NAME = "manifest.json"

def _walk_snapshot(doc_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Walk doc_dir and return {rel_path: {size, mtime}}.
    Only files (skip directories). Works for PDFs, txt, etc.
    """
    snap = {}
    for root, _, files in os.walk(doc_dir):
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except FileNotFoundError:
                continue
            rel = os.path.relpath(p, doc_dir)
            snap[rel] = {"size": st.st_size, "mtime": int(st.st_mtime)}
    return snap

def _load_manifest(index_dir: str) -> Dict[str, Any]:
    path = os.path.join(index_dir, MANIFEST_NAME)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_manifest(index_dir: str, snapshot: Dict[str, Any]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"snapshot": snapshot, "saved_at": int(time.time())}, f, ensure_ascii=False, indent=2)

def _has_changed(doc_dir: str, index_dir: str) -> bool:
    cur = _walk_snapshot(doc_dir)
    man = _load_manifest(index_dir)
    prev = man.get("snapshot", {})
    return cur != prev

def _build_index(doc_dir: str, index_dir: str) -> VectorStoreIndex:
    # Read documents (PDFs, txt, md, etc. supported by SimpleDirectoryReader)
    documents = SimpleDirectoryReader(doc_dir).load_data()

    # Merge pages per file
    file_to_pages = defaultdict(list)
    for d in documents:
        fname = d.metadata.get("file_name") or d.metadata.get("filename") or "unknown_source"
        file_to_pages[fname].append(d.text)

    merged_docs: List[Document] = [
        Document(text="\n".join(pages), metadata={"file_name": fn})
        for fn, pages in file_to_pages.items()
    ]

    # Chunk
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=32)
    nodes = splitter.get_nodes_from_documents(merged_docs) if merged_docs else []

    # Build + persist
    index = VectorStoreIndex(nodes)
    os.makedirs(index_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=index_dir)

    # Save manifest snapshot
    _save_manifest(index_dir, _walk_snapshot(doc_dir))
    return index

def build_or_load_index(doc_dir: str = str(RAG_DIR),
                        index_dir: str = str(INDEX_DIR)) -> VectorStoreIndex:
    """
    Load an existing index if present and up-to-date; otherwise rebuild.
    If loading fails, nuke the index dir and rebuild cleanly.
    """
    os.makedirs(doc_dir, exist_ok=True)

    if os.path.exists(index_dir):
        # If docs changed since last build, rebuild.
        if _has_changed(doc_dir, index_dir):
            shutil.rmtree(index_dir, ignore_errors=True)
            return _build_index(doc_dir, index_dir)
        # Try to load existing
        try:
            storage = StorageContext.from_defaults(persist_dir=index_dir)
            return load_index_from_storage(storage)
        except Exception:
            shutil.rmtree(index_dir, ignore_errors=True)
            return _build_index(doc_dir, index_dir)
    else:
        # No index yet -> build
        return _build_index(doc_dir, index_dir)

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
