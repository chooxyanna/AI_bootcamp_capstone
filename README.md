# AI Bootcamp Capstone (RAG + Tools)
### About
Text-only LLM agent that **always** reads a local **RAG** context first, then decides whether to call tools:

> Flow: **RAG → LLM → tools (only if necessary) → Answer**

> Tools:
>- **OCR** (RapidOCR ONNX) for text extraction
>- **Object detection** (DETR) with **annotated image** saved to `outputs/`
>- **Web search** (Tavily)


### Project Structure
├─ .env                     # **Action:** add your API keys here (OPENAI_API_KEY, TAVILY_API_KEY)
├─ requirements.txt          # Python dependencies
├─ main.py                   # Entry point for the application
│
├─ app/
│  ├─ __init__.py
│  ├─ config.py              # Configuration and environment setup
│  ├─ utils.py               # Helper functions
│  ├─ graph.py               # Defines main workflow graph
│  │
│  ├─ rag/
│  │  ├─ __init__.py
│  │  ├─ indexer.py          # build_or_load_index(), QUERY_ENGINE, format_sources()
│  │
│  ├─ tools/
│  │  ├─ __init__.py
│  │  ├─ tavily_tool.py      # Tavily search tool
│  │  ├─ ocr_tool.py         # RapidOCR tool
│  │  └─ obj_detect_tool.py  # DETR tool (saves annotated image + optional data_url)
│
└─ data/
   ├─ output/                # Stores processed results or outputs
   └─ rag/
      ├─ rag_storage/        # **Action:** place your PDFs/TXT/Docs here
      └─ index_store/        # Auto-persisted vector index (cached)

# How to use

### Requirements
- Create an environment with **Python 3.11** (recommended & tested)
- pip install -r requirements.txt (install dependencies)

### Usage
Run **with streamlit** on local machine\
-streamlit run streamlit_app.py

Run **without streamlit** on local machine\
-python main.py --question QUESTION [--image IMAGE] **# --question is required**

### Keys
- Create .env file with\
    OPENAI_API_KEY=\
    TAVILY_API_KEY=
