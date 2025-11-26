import os, io, base64
from typing import Sequence
from PIL import Image as PILImage
from langchain_core.messages import BaseMessage

def load_image_any(image: str) -> PILImage:
    if os.path.exists(image):
        return PILImage.open(image).convert("RGB")
    if image.startswith("data:image"):
        encoded = image.split(",", 1)[1]
        encoded = "".join(encoded.split())
        pad = (-len(encoded)) % 4
        if pad: encoded += "=" * pad
        img_bytes = base64.b64decode(encoded, validate=False)
        im = PILImage.open(io.BytesIO(img_bytes))
        im.load()
        return im.convert("RGB")
    if image.startswith(("http://", "https://")):
        import requests
        r = requests.get(image, timeout=15)
        r.raise_for_status()
        im = PILImage.open(io.BytesIO(r.content))
        im.load()
        return im.convert("RGB")
    raise ValueError("invalid image input (not a path, data URL, or http URL)")

def extract_user_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human":
            if isinstance(m.content, list):
                for part in m.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
            elif isinstance(m.content, str):
                return m.content
    return ""
