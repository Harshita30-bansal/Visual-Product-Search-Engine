"""utils/yolo_utils.py — YOLO detection and cropping."""

from PIL import Image

YOLO_CLASS_NAMES = {0: "Upper Body", 1: "Lower Body", 2: "Full Body"}


def load_yolo(weights_path: str):
    """Load YOLO model from weights file."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        print(f"[YOLO] Loaded from {weights_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO: {e}")


def run_yolo_crop(model, image, conf_threshold=0.3, padding=10):
    """Original single-crop — picks highest confidence box."""
    W, H = image.size
    try:
        results = model(image, conf=conf_threshold, verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes    = results[0].boxes
            best_idx = int(boxes.conf.argmax().item())
            conf     = float(boxes.conf[best_idx].item())
            if conf >= conf_threshold:
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                x1 = max(0, x1-padding); y1 = max(0, y1-padding)
                x2 = min(W, x2+padding); y2 = min(H, y2+padding)
                return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), True
    except Exception as e:
        print(f"[YOLO] error: {e}")
    return image.copy(), None, False


def detect_all_clothing(model, image, conf_threshold=0.1, padding=10):
    """
    Detect ALL clothing items grouped by class.

    KEY BEHAVIOUR:
    - Uses low confidence (0.1) so YOLO finds as many classes as possible
    - For any class YOLO misses, provides an intelligent fallback crop:
        Upper Body -> top 55% of image
        Lower Body -> bottom 55% of image
        Full Body  -> entire image
    - ALWAYS returns all 3 options so user can always choose

    Each entry has 'detected': True if YOLO found it, False if fallback.
    The UI can show a badge to tell user which ones YOLO actually detected.

    Returns ordered dict: Upper Body, Lower Body, Full Body
    """
    W, H = image.size
    found = {}

    try:
        results = model(image, conf=conf_threshold, verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes    = results[0].boxes
            xyxy_all = boxes.xyxy.cpu().numpy()
            cls_all  = boxes.cls.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()

            for i in range(len(xyxy_all)):
                cls_id   = int(cls_all[i])
                cls_name = YOLO_CLASS_NAMES.get(cls_id)
                if cls_name is None:
                    continue
                conf_val = float(conf_all[i])
                if cls_name in found and found[cls_name]["conf"] >= conf_val:
                    continue
                x1, y1, x2, y2 = xyxy_all[i].astype(int)
                x1 = max(0, x1-padding); y1 = max(0, y1-padding)
                x2 = min(W, x2+padding); y2 = min(H, y2+padding)
                found[cls_name] = {
                    "crop": image.crop((x1, y1, x2, y2)),
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf_val,
                    "class_id": cls_id,
                    "detected": True,
                }
    except Exception as e:
        print(f"[YOLO] detect_all_clothing error: {e}")

    # Always ensure all 3 options exist using intelligent fallbacks
    if "Upper Body" not in found:
        y2 = int(H * 0.55)
        found["Upper Body"] = {
            "crop": image.crop((0, 0, W, y2)),
            "bbox": (0, 0, W, y2),
            "conf": 0.0, "class_id": 0, "detected": False,
        }

    if "Lower Body" not in found:
        y1 = int(H * 0.45)
        found["Lower Body"] = {
            "crop": image.crop((0, y1, W, H)),
            "bbox": (0, y1, W, H),
            "conf": 0.0, "class_id": 1, "detected": False,
        }

    if "Full Body" not in found:
        found["Full Body"] = {
            "crop": image.copy(),
            "bbox": (0, 0, W, H),
            "conf": 0.0, "class_id": 2, "detected": False,
        }

    # Return in fixed order
    return {k: found[k] for k in ["Upper Body", "Lower Body", "Full Body"]}
