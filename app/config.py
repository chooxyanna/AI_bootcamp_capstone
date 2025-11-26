# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# --- always load .env locally; harmless on Streamlit ---
load_dotenv()

# Try to read from Streamlit secrets when running on Streamlit Cloud
def _from_secrets(name: str, default: str | None = None) -> str | None:
    try:
        import streamlit as st  # present on Cloud
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return default

def get_secret(name: str, default: str | None = None) -> str | None:
    # priority: Streamlit secrets -> env var -> default
    v = _from_secrets(name)
    if v is not None:
        return v
    return os.getenv(name, default)

# ----- API KEYS & MODEL -----
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Provide it via Streamlit secrets or your local .env."
    )

# ----- PATHS -----
REPO_ROOT = Path(__file__).resolve().parents[1]

# Choose a writable base dir:
# - Streamlit Cloud: /mount/data
# - Local: <repo>/data
IS_STREAMLIT = "STREAMLIT_RUNTIME" in os.environ
DEFAULT_DATA_BASE = Path("/mount/data") if IS_STREAMLIT else (REPO_ROOT / "data")

# Allow override via secrets or env
DATA_BASE = Path(
    _from_secrets("DATA_BASE", os.getenv("DATA_BASE", str(DEFAULT_DATA_BASE)))
).resolve()

# RAG dirs live under the writable base (NOT inside the repo tree on Cloud)
RAG_DIR   = DATA_BASE / "rag"
DOC_DIR   = Path(get_secret("RAG_STORAGE_DIR", str(RAG_DIR / "rag_storage"))).resolve()
INDEX_DIR = Path(get_secret("INDEX_STORE_DIR",  str(RAG_DIR / "index_store"))).resolve()

# Where we save annotated images, etc.
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(DATA_BASE / "outputs"))).resolve()

# Ensure directories exist (writable on Cloud)
for p in (RAG_DIR, DOC_DIR, INDEX_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)
