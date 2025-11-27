# app/tools/obj_detect_tool.py
import os, json, base64, io
from typing import Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from langchain_core.tools import tool
from transformers import DetrImageProcessor, DetrForObjectDetection
from app.utils import load_image_any
from app.config import OUTPUT_DIR

def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], text: str):
    """Draw a small filled label box + text using PIL only."""
    # Use default bitmap font (no external deps)
    font = ImageFont.load_default()
    tw, th = draw.textbbox((0,0), text, font=font)[2:]
    x, y = xy
    pad = 3
    # background
    draw.rectangle([x, y - th - 2*pad, x + tw + 2*pad, y], fill=(0,255,0))
    # text
    draw.text((x + pad, y - th - pad), text, fill=(0,0,0), font=font)

@tool
def obj_detect(
    image: str,
    conf: float = 0.25,
    model_name: str = "facebook/detr-resnet-50",
    revision: str = "no_timm",
    return_data_url: bool = True,
) -> str:
    """
    DETR object detection (no OpenCV). Returns JSON with detections, a saved
    annotated image path, and (optionally) a data_url for UI that supports it.
    """
    try:
        # Load image (local path, data URL, or http URL)
        img_pil = Image.open(image).convert("RGB") if os.path.exists(image) else load_image_any(image)

        # Lazy-load & cache model/processor on the function
        if not hasattr(obj_detect, "_proc"):
            obj_detect._proc = DetrImageProcessor.from_pretrained(model_name, revision=revision)
        if not hasattr(obj_detect, "_model"):
            obj_detect._model = DetrForObjectDetection.from_pretrained(model_name, revision=revision).eval()

        proc  = obj_detect._proc
        model = obj_detect._model

        # Forward pass (DETR can take PIL directly)
        inputs = proc(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process to boxes
        target_sizes = torch.tensor([img_pil.size[::-1]])  # (H, W)
        results = proc.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=float(conf))[0]

        # Draw with PIL
        out = img_pil.copy()
        draw = ImageDraw.Draw(out)
        dets = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            s = float(score.item())
            lid = int(label.item())
            name = model.config.id2label.get(lid, str(lid))
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            dets.append({"cls_id": lid, "cls_name": name, "conf": s, "bbox_xyxy": [x1, y1, x2, y2]})

            # green rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
            _draw_label(draw, (x1, y1), f"{name} {s:.2f}")

        # Save annotated image
        base = os.path.splitext(os.path.basename(getattr(img_pil, "filename", "image.png")))[0] or "image"
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{base}.png")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out.save(out_path, format="PNG")

        payload = {
            "source": "obj_detect",
            "model": model_name,
            "conf": conf,
            "detections": dets,
            "annotated_path": out_path
        }

        if return_data_url:
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            payload["data_url"] = f"data:image/png;base64,{b64}"

        return f"[obj_detect ok] {json.dumps(payload)}"

    except Exception as e:
        return f"[obj_detect error] {e}"
