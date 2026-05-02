"""
Run this BEFORE streamlit to verify all models load correctly.
Usage: python test_load.py
"""

import os
import sys
import time

print("=" * 50)
print("Visual Product Search — Load Test")
print("=" * 50)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

required = {
    "best.pt":                   os.path.join(MODELS_DIR, "best.pt"),
    "clip_finetuned_full.pt":    os.path.join(MODELS_DIR, "clip_finetuned_full.pt"),
    "faiss_index_C_alpha07.bin": os.path.join(MODELS_DIR, "faiss_index_C_alpha07.bin"),
    "gallery_metadata.csv":      os.path.join(MODELS_DIR, "gallery_metadata.csv"),
}

print("\n[1] Checking model files...")
all_ok = True
for name, path in required.items():
    exists = os.path.exists(path)
    size   = f"{os.path.getsize(path)/1e6:.1f} MB" if exists else "MISSING"
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {name}  ({size})")
    if not exists:
        all_ok = False

if not all_ok:
    print("\nERROR: Some files are missing.")
    sys.exit(1)

print("\n[2] Testing YOLO...")
t0 = time.time()
from ultralytics import YOLO
yolo = YOLO(required["best.pt"])
print(f"  YOLO loaded in {time.time()-t0:.1f}s ✓")

print("\n[3] Testing CLIP...")
t0 = time.time()
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print(f"  CLIP base loaded in {time.time()-t0:.1f}s ✓")

t0 = time.time()
state = torch.load(required["clip_finetuned_full.pt"], map_location="cpu", weights_only=False)
model.load_state_dict(state)
model.eval()
print(f"  Fine-tuned weights applied in {time.time()-t0:.1f}s ✓")

print("\n[4] Testing FAISS...")
t0 = time.time()
import faiss
index = faiss.read_index(required["faiss_index_C_alpha07.bin"])
print(f"  FAISS index loaded in {time.time()-t0:.1f}s — {index.ntotal:,} vectors ✓")

print("\n[5] Testing gallery metadata...")
import pandas as pd
df = pd.read_csv(required["gallery_metadata.csv"])
print(f"  Metadata loaded — {len(df):,} rows ✓")

print("\n[6] Test encoding a dummy image...")
from PIL import Image

dummy  = Image.new("RGB", (224, 224), color=(128, 64, 32))
inputs = processor(images=dummy, return_tensors="pt")

with torch.no_grad():
    # get_image_features returns a tensor directly (not a model output object)
    # Use vision_model + visual_projection to be safe across transformers versions
    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
    pooled     = vision_out.pooler_output          # (1, hidden_dim)
    projected  = model.visual_projection(pooled)   # (1, 512)

projected = projected / projected.norm(dim=-1, keepdim=True)
vec = projected.numpy()

print(f"  Encoded — shape={vec.shape}, norm={np.linalg.norm(vec[0]):.4f} ✓")

print("\n[7] Test FAISS search...")
dists, idxs = index.search(vec.astype(np.float32), 5)
print(f"  Search returned {len(idxs[0])} results ✓")
print(f"  Top scores: {[round(float(d),4) for d in dists[0]]}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✓")
print("Now run:  streamlit run app.py")
print("=" * 50)