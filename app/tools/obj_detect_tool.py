import os, json, base64
import numpy as np
import cv2, torch
from PIL import Image
from langchain_core.tools import tool
from transformers import DetrImageProcessor, DetrForObjectDetection
from app.utils import load_image_any
from app.config import OUTPUT_DIR

@tool
def obj_detect(
    image: str,
    conf: float = 0.25,
    model_name: str = "facebook/detr-resnet-50",
    revision: str = "no_timm",
    return_data_url: bool = True,
) -> str:
    """
    DETR object detection. Returns JSON string with detections, the saved annotated path,
    and (optionally) a data_url for UI that supports it.
    """
    try:
        img_pil = Image.open(image) if os.path.exists(image) else load_image_any(image)

        if not hasattr(obj_detect, "_proc"):
            obj_detect._proc = DetrImageProcessor.from_pretrained(model_name, revision=revision)
        if not hasattr(obj_detect, "_model"):
            obj_detect._model = DetrForObjectDetection.from_pretrained(model_name, revision=revision).eval()

        proc  = obj_detect._proc
        model = obj_detect._model

        inputs = proc(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([img_pil.size[::-1]])  # (H, W)
        results = proc.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=float(conf))[0]

        out_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        dets = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            s = float(score.item())
            lid = int(label.item())
            name = model.config.id2label.get(lid, str(lid))
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            dets.append({"cls_id": lid, "cls_name": name, "conf": s, "bbox_xyxy": [x1, y1, x2, y2]})
            cv2.rectangle(out_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
            label_txt = f"{name} {s:.2f}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_top = max(0, y1 - th - 6)
            cv2.rectangle(out_bgr, (x1, y_top), (x1 + tw + 6, y1), (0,255,0), -1)
            cv2.putText(out_bgr, label_txt, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # save annotated image
        base = os.path.splitext(os.path.basename(getattr(img_pil, "filename", "image.png")))[0] or "image"
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{base}.png")
        cv2.imwrite(out_path, out_bgr)

        payload = {"source": "obj_detect", "model": model_name, "conf": conf, "detections": dets, "annotated_path": out_path}

        if return_data_url:
            ok, buf = cv2.imencode(".png", out_bgr)
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                payload["data_url"] = f"data:image/png;base64,{b64}"

        return f"[obj_detect ok] {json.dumps(payload)}"

    except Exception as e:
        return f"[obj_detect error] {e}"
