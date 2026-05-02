"""utils/yolo_utils.py — YOLO detection and cropping."""

from PIL import Image


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
    Run YOLO on image and return the highest-confidence crop.

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

                # Apply padding, clamp to image bounds
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(W, x2 + padding)
                y2 = min(H, y2 + padding)

                return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2), True

    except Exception as e:
        print(f"[YOLO] Inference error: {e}")

    # Fallback — return full image
    return image.copy(), None, False
