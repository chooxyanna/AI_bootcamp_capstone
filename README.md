# AI Bootcamp Capstone (RAG + Tools)
### About
Text-only LLM agent that **always** reads a local **RAG** context first, then decides whether to call tools:

> Flow: **RAG → LLM → tools (only if necessary) → Answer**

> Tools:
>- **OCR** (RapidOCR ONNX) for text extraction
>- **Object detection** (DETR) with **annotated image** saved to `outputs/`
>- **Web search** (Tavily)


### Project Structure
├─ .env **# Action: put OPENAI_API_KEY, TAVILY_API_KEY here**\
├─ requirements.txt\
├─ main.py\
└─ app/\
├─ init.py\
├─ config.py\
├─ utils.py\
├─ graph.py\
├─ tools/\
│ ├─ init.py\
│ ├─ tavily_tool.py # tavily_search tool\
│ ├─ ocr_tool.py # RapidOCR tool\
│ └─ obj_detect_tool.py # DETR tool (saves annotated image + optional data_url)\
└─ rag/\
├─ init.py\
├─ indexer.py # build_or_load_index(), QUERY_ENGINE, format_sources()\
├─ rag_storage/ **# Action: put your PDFs/TXT/Docs here**\
└─ index_store/ # auto-persisted vector index (cached)\

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
