# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv


# try to read from streamlit secrets when running on Streamlit Cloud
def _get_secret(name: str, default: str | None = None):
    try:
        import streamlit as st
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets[name]
    except Exception:
        # Load .env once here so everything downstream sees the env vars
        load_dotenv()

        # ----- API KEYS & MODEL -----
        OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
        TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY", "")
        OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # default if not set

        if not OPENAI_API_KEY:
            # Fail fast with a clear message; you can relax this if you prefer warnings
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env at repo root, e.g.\n"
                "OPENAI_API_KEY=sk-...\n"
                "TAVILY_API_KEY=tvly-...   # optional, only for tavily tool"
            )
    return os.getenv(name, default)



# ----- PATHS -----
# repo/
#   .env
#   requirements.txt
#   main.py
#   app/
#     ...
REPO_ROOT = Path(__file__).resolve().parents[1]

# RAG lives under app/rag/
RAG_DIR          = REPO_ROOT / "app" / "rag"
DOC_DIR          = Path(os.getenv("RAG_DIR", RAG_DIR / "rag_storage")).resolve()
INDEX_DIR        = Path(os.getenv("INDEX_DIR",  RAG_DIR / "index_store")).resolve()

# Outputs (keep inside repo by default)
OUTPUT_DIR       = Path(os.getenv("OUTPUT_DIR", REPO_ROOT / "outputs")).resolve()

# Ensure directories exist
for p in (DOC_DIR, INDEX_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)
