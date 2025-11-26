from dotenv import load_dotenv

import requests, os, operator, io, base64, mimetypes, hashlib, json, cv2, torch, shutil
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import Annotated, Sequence
from langchain.callbacks.tracers.langchain import LangChainTracer
from langgraph.checkpoint.memory import MemorySaver

import numpy as np
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
from pathlib import Path
from IPython.display import Image as DisplayImage
from PIL import Image as PILImage
import pytesseract

from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext, load_index_from_storage
from collections import defaultdict
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader, Document, load_index_from_storage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.tools.tavily_research.base import TavilyToolSpec
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load variables from .env file
load_dotenv()

# Access them safely
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("OpenAI key loaded:", bool(openai_key))
print("Tavily key loaded:", bool(tavily_key))

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_key
    # temperature=0
)
# ------ HELPER FUNCTION ------
# universal image loader
def _load_image_any(image: str) -> Image:
    # if is local path
    if os.path.exists(image):
        return PILImage.open(image).convert("RGB")

    # if is data URL
    if image.startswith("data:image"):
        # keep only the part after the last comma (handles extra params before base64)
        encoded = image.split(",", 1)[1]
        # trim whitespace and fix missing padding
        encoded = "".join(encoded.split())
        pad = (-len(encoded)) % 4
        if pad:
            encoded += "=" * pad
        img_bytes = base64.b64decode(encoded, validate=False)
        bio = io.BytesIO(img_bytes)
        # verify & reopen for safety
        im = Image.open(bio)
        im.load()
        return im.convert("RGB")

    # if is online URL
    if image.startswith("http://") or image.startswith("https://"):
        import requests
        r = requests.get(image, timeout=15)
        r.raise_for_status()
        bio = io.BytesIO(r.content)
        im = Image.open(bio)
        im.load()
        return im.convert("RGB")

    raise ValueError("invalid image input (not a path, data URL, or http URL)")

# for internal_rag tool
def _format_sources(resp):
    srcs = []
    for n in getattr(resp, "source_nodes", []) or []:
        srcs.append({
            "file": n.metadata.get("file_name", "unknown"),
            "score": getattr(n, "score", None),
            "preview": (n.text[:240] + "…") if n.text else ""
        })
    return srcs

def _extract_user_text(messages: Sequence[BaseMessage]) -> str:
    """Pull the most recent user text block from the messages."""
    # Works with your style "HumanMessage(content=[{'type':'text','text':...}, ...])"
    for m in reversed(messages):
        if hasattr(m, "type") and m.type == "human":
            if isinstance(m.content, list):
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
            elif isinstance(m.content, str):
                return m.content
    return ""
