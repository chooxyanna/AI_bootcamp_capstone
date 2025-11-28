# AI Bootcamp Capstone (RAG + Tools)
### About
Text-only LLM agent that **always** reads a local **RAG** context first, then decides whether to call tools:

> Flow: **RAG → LLM → tools (only if necessary) → Answer**

> Tools:
>- **OCR** (RapidOCR ONNX) for text extraction
>- **Object detection** (DETR) with **annotated image** saved to `outputs/`
>- **Web search** (Tavily)


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

### Rag Storage
Expand the storage by adding files into data/rag/rag_storage
