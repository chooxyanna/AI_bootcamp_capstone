# streamlit_app.py
import os
import io
import json
import tempfile
from pathlib import Path
import streamlit as st
from app.config import DOC_DIR
from app.rag.indexer import build_or_load_index

import streamlit as st
from PIL import Image as PILImage

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage

# Your app code
from app.graph import build_graph
from app.config import OUTPUT_DIR

# --- Helpers -----------------------------------------------------------------

def ensure_output_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return Path(OUTPUT_DIR)

def _read_tool_payload(prefix: str, content: str):
    """Extract JSON payload from a tool string like: '[obj_detect ok] {...}'"""
    if not isinstance(content, str):
        return None
    if content.startswith(prefix):
        try:
            json_str = content[len(prefix):].strip()
            return json.loads(json_str)
        except Exception:
            return None
    return None

def find_objdetect_artifacts(messages: list[BaseMessage]):
    """
    Scan ToolMessages for obj_detect outputs. Return (annotated_path, data_url, dets).
    If multiple, return the most recent.
    """
    annotated_path = None
    data_url = None
    dets = None
    for m in messages[::-1]:
        if isinstance(m, ToolMessage):
            payload = _read_tool_payload("[obj_detect ok]", m.content)
            if payload and isinstance(payload, dict):
                annotated_path = payload.get("annotated_path") or annotated_path
                data_url = payload.get("data_url") or data_url
                dets = payload.get("detections") or dets
                if annotated_path or data_url:
                    break
    return annotated_path, data_url, dets

def save_uploaded_image(uploaded_file) -> str:
    """Save uploaded image to OUTPUT_DIR and return local path."""
    ensure_output_dir()
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(OUTPUT_DIR)) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def build_user_message(question: str, image_path: str | None):
    text = question.strip()
    if image_path:
        # Your graph expects the image hint in the text
        text += f"\nImage path: {image_path}"
    return HumanMessage(content=[{"type": "text", "text": text}])

# --- App UI ------------------------------------------------------------------

st.sidebar.header("Upload RAG docs")
files = st.sidebar.file_uploader(
    "Upload PDFs/TXTs/DOCX/MD", type=["pdf","txt","md","docx"], accept_multiple_files=True
)
if files:
    doc_dir = Path(DOC_DIR)
    for f in files:
        (doc_dir / f.name).write_bytes(f.read())
    st.sidebar.success(f"Uploaded {len(files)} file(s) to {doc_dir}")
    # Rebuild the index so the new docs are searchable
    build_or_load_index(doc_dir=str(doc_dir))

# Initialize graph + memory (once per session)
if "graph_bundle" not in st.session_state:
    graph, config, system_msg = build_graph()
    # Give each browser session its own thread_id so MemorySaver persists per user
    # Replace the default thread_id with a session-scoped one
    session_thread_id = f"tid-{os.urandom(8).hex()}"
    config = {"configurable": {"thread_id": session_thread_id}}
    st.session_state.graph_bundle = (graph, config, system_msg)
    st.session_state.history = []  # store (role, content)

graph, config, system_msg = st.session_state.graph_bundle

# Chat input
with st.form("qa"):
    question = st.text_area("Your question", placeholder="Ask anything… e.g., 'What is in the photo? Show detections.'", height=100)
    up = st.file_uploader("Optional image", type=["png", "jpg", "jpeg", "webp"])
    submitted = st.form_submit_button("Run")

# Run
if submitted and question.strip():
    img_path = None
    if up is not None:
        img_path = save_uploaded_image(up)

    user_msg = build_user_message(question, img_path)

    # Invoke the graph with memory (system message + user message)
    with st.spinner("Thinking…"):
        result = graph.invoke({"messages": [system_msg, user_msg]}, config)

    # Display final assistant reply
    final = result["messages"][-1]
    if isinstance(final, AIMessage):
        st.markdown("### Assistant")
        st.write(final.content)
        st.session_state.history.append(("assistant", final.content))

    # Try to pull annotated image/artifacts from ToolMessages
    ann_path, data_url, dets = find_objdetect_artifacts(result["messages"])
    if ann_path and os.path.exists(ann_path):
        st.markdown("### Annotated detection")
        st.image(PILImage.open(ann_path), use_column_width=True)
        with st.expander("Detections (JSON)"):
            st.json(dets or [])
    elif data_url:
        try:
            header, b64 = data_url.split(",", 1)
            st.markdown("### Annotated detection")
            st.image(io.BytesIO(bytearray(b64, "utf-8")), use_column_width=True)  # fallback approach
        except Exception:
            pass

# Show conversation history (current session)
if st.session_state.get("history"):
    st.markdown("---")
    st.markdown("### Session History")
    for role, content in st.session_state.history[-20:]:
        st.markdown(f"**{role.title()}:** {content}")
