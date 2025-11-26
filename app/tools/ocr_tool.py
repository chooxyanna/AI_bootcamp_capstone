import json, hashlib, os
from PIL import Image
from langchain_core.tools import tool
from rapidocr_onnxruntime import RapidOCR
from app.utils import load_image_any

@tool
def ocr(image: str, hint: str = "") -> str:
    """
    Run OCR (RapidOCR) on a local path or data URL; return JSON payload string.
    """
    try:
        engine = RapidOCR(rec_model='onnx/ch_PP-OCRv3_rec.onnx')
        img = Image.open(image) if os.path.exists(image) else load_image_any(image)

        result, _ = engine(img)
        texts = [t for _, t, score in result if score > 0.5]
        text = "\n".join(texts).strip() or "[ocr: no text detected]"

        payload = {"source": "ocr", "hint": hint, "text": text}
        payload["sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"[ocr ok] {json.dumps(payload, ensure_ascii=False)}"
    except Exception as e:
        return f"[ocr error] {e}"
