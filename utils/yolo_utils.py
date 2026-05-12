"""utils/yolo_utils.py — YOLO detection and cropping."""

from PIL import Image


# YOLO class ID → clothing type name
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


def run_yolo_crop(
    model,
    image: Image.Image,
    conf_threshold: float = 0.3,
    padding: int = 10,
):
    """
    Original single-crop function — kept for compatibility.
    Picks the highest-confidence detection regardless of class.

    Returns
    -------
    cropped  : PIL Image  — cropped region (full image if no detection)
    bbox     : tuple (x1, y1, x2, y2) or None
    success  : bool — True if YOLO found something above conf_threshold
    """
    W, H = image.size

    try:
        results = model(image, conf=conf_threshold, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes    = results[0].boxes
            best_idx = int(boxes.conf.argmax().item())
            conf     = float(boxes.conf[best_idx].item())

            if conf >= conf_threshold:
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(W, x2 + padding)
                y2 = min(H, y2 + padding)
                return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), True

    except Exception as e:
        print(f"[YOLO] Inference error: {e}")

    return image.copy(), None, False


def detect_all_clothing(
    model,
    image: Image.Image,
    conf_threshold: float = 0.25,
    padding: int = 10,
):
    """
    NEW — Detect ALL clothing items in the image grouped by class.

    Per class (Upper Body / Lower Body / Full Body), keeps the box
    with the highest confidence score.

    Returns
    -------
    detected : dict, e.g.:
        {
            "Upper Body": {
                "crop":     PIL Image,
                "bbox":     (x1, y1, x2, y2),
                "conf":     0.91,
                "class_id": 0,
            },
            "Lower Body": { ... },   # only present if detected
            "Full Body":  { ... },   # only present if detected
        }
    Returns empty dict if nothing detected.
    """
    W, H = image.size
    detected = {}

    try:
        results = model(image, conf=conf_threshold, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return detected

        boxes       = results[0].boxes
        xyxy_all    = boxes.xyxy.cpu().numpy()
        cls_all     = boxes.cls.cpu().numpy().astype(int)
        conf_all    = boxes.conf.cpu().numpy()

        for i in range(len(xyxy_all)):
            cls_id   = int(cls_all[i])
            cls_name = YOLO_CLASS_NAMES.get(cls_id, f"Class {cls_id}")
            conf_val = float(conf_all[i])

            # Keep only the highest-confidence box per class
            if cls_name in detected and detected[cls_name]["conf"] >= conf_val:
                continue

            x1, y1, x2, y2 = xyxy_all[i].astype(int)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(W, x2 + padding)
            y2 = min(H, y2 + padding)

            detected[cls_name] = {
                "crop":     image.crop((x1, y1, x2, y2)),
                "bbox":     (x1, y1, x2, y2),
                "conf":     conf_val,
                "class_id": cls_id,
            }

    except Exception as e:
        print(f"[YOLO] detect_all_clothing error: {e}")

    return detected
