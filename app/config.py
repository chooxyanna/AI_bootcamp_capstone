# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load .env locally; harmless on Streamlit Cloud
load_dotenv()

def _from_st_secrets(name: str):
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return None

def get_secret(name: str, default: str | None = None) -> str | None:
    # Priority: Streamlit secrets -> env var -> default
    v = _from_st_secrets(name)
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

def _pick_data_base() -> Path:
    """Choose a writable base dir. Prefer /mount/data on Streamlit Cloud."""
    candidates = []
    # 1) explicit override
    override = get_secret("DATA_BASE", os.getenv("DATA_BASE", "")).strip()
    if override:
        candidates.append(Path(override))
    # 2) Streamlit Cloud default
    candidates.append(Path("/mount/data"))
    # 3) local fallback (inside repo)
    candidates.append(REPO_ROOT / "data")

    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok")
            test.unlink()
            return p
        except Exception:
            continue
    # last resort
    return REPO_ROOT / "data"

DATA_BASE = _pick_data_base()

# RAG dirs live under the writable base
RAG_DIR   = DATA_BASE / "rag"
DOC_DIR   = Path(get_secret("RAG_DIR", str(RAG_DIR / "rag_storage"))).resolve()
INDEX_DIR = Path(get_secret("INDEX_DIR",  str(RAG_DIR / "index_store"))).resolve()

# Where we save annotated images, etc.
OUTPUT_DIR = Path(get_secret("OUTPUT_DIR", str(DATA_BASE / "outputs"))).resolve()

for p in (DOC_DIR, INDEX_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)
