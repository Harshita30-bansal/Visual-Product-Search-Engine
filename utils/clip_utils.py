"""utils/clip_utils.py — CLIP model loading and query encoding."""

import os
import numpy as np
import torch
from PIL import Image

CLIP_BASE = "openai/clip-vit-base-patch32"


def load_clip(finetuned_weights: str = None):
    """
    Load CLIP processor and model.
    First run downloads ~600MB (cached automatically after that).

    Returns: (model, processor)
    """
    from transformers import CLIPProcessor, CLIPModel

    print(f"[CLIP] Loading {CLIP_BASE} ...")
    processor = CLIPProcessor.from_pretrained(CLIP_BASE)
    model     = CLIPModel.from_pretrained(CLIP_BASE)
    print("[CLIP] Base model loaded ✓")

    # Apply fine-tuned weights
    if finetuned_weights and os.path.exists(finetuned_weights):
        print(f"[CLIP] Applying fine-tuned weights...")
        state = torch.load(finetuned_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        print("[CLIP] Fine-tuned weights applied ✓")
    elif finetuned_weights:
        print(f"[CLIP] WARNING: {finetuned_weights} not found — using vanilla CLIP")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print("[CLIP] Ready ✓")
    return model, processor


def embed_query(model, processor, image: Image.Image) -> np.ndarray:
    """
    Encode PIL image → L2-normalised CLIP embedding.
    Returns np.ndarray shape (1, 512), float32.
    """
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        # Use vision_model + visual_projection directly
        # (works across all transformers versions)
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled     = vision_out.pooler_output        # (1, hidden_dim)
        projected  = model.visual_projection(pooled) # (1, 512)

    projected = projected / projected.norm(dim=-1, keepdim=True)
    return projected.numpy().astype(np.float32)