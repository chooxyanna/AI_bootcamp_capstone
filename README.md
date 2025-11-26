# AI Bootcamp Capstone (RAG + Tools)

Text-only LLM agent that **always** reads a local **RAG** context first, then decides whether to call tools:

> Flow: **RAG → LLM → tools (only if necessary) → Answer**

> Tools:
- **OCR** (RapidOCR ONNX) for text extraction
- **Object detection** (DETR) with **annotated image** saved to `outputs/`
- **Web search** (Tavily)


## Project Structure
├─ .env **Action: put OPENAI_API_KEY, TAVILY_API_KEY here**
├─ requirements.txt # pins Python 3.11-compatible deps
├─ main.py # CLI entry point
└─ app/
├─ init.py
├─ config.py # env + paths (RAG dirs, OUTPUT dir, model name)
├─ utils.py # helpers (load_image_any, extract_user_text)
├─ graph.py # rag_node, agent_node, tool_node, build_graph()
├─ tools/
│ ├─ init.py
│ ├─ tavily_tool.py # tavily_search tool
│ ├─ ocr_tool.py # RapidOCR tool
│ └─ obj_detect_tool.py # DETR tool (saves annotated image + optional data_url)
└─ rag/
├─ init.py
├─ indexer.py # build_or_load_index(), QUERY_ENGINE, format_sources()
├─ rag_storage/ **action: put your PDFs/TXT/Docs here**
└─ index_store/ # auto-persisted vector index (cached)



## Requirements
- **Python 3.11** (recommended & tested)
- pip install -r requirements.txt (install dependencies)