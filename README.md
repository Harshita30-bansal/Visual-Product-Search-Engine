# Visual Product Search Engine — Demo App

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download model files from Kaggle

| File | Kaggle dataset | Put it at |
|------|---------------|-----------|
| `best.pt` | `harshitabansal307/yolo-bbox-crops-v1` | `models/best.pt` |
| `clip_finetuned_full.pt` | `likithareddy2508/clip-finetuned-indexes-c` | `models/clip_finetuned_full.pt` |
| `faiss_index_C_alpha07.bin` | `likithareddy2508/clip-finetuned-indexes-c` | `models/faiss_index_C_alpha07.bin` |
| `gallery_metadata.csv` | `likithareddy2508/clip-indexes-ab` | `models/gallery_metadata.csv` |

**Optional — for showing result images:**
- Download `data/bbox_crops/` folder from `yolo-bbox-crops-v1`
- Rename it to `gallery_crops` and place at `models/gallery_crops/`

### Step 3 — Run the app
```bash
streamlit run app.py
```

Open browser at: **http://localhost:8501**

---

## models/ folder structure
```
models/
├── best.pt                      ← YOLO weights
├── clip_finetuned_full.pt       ← Fine-tuned CLIP weights
├── faiss_index_C_alpha07.bin    ← FAISS search index
├── gallery_metadata.csv         ← Maps index positions to item_ids
└── gallery_crops/               ← (optional) gallery images for display
    ├── MEN/
    └── WOMEN/
```

---

## Project folder structure
```
vr_app/
├── app.py               ← Main Streamlit app
├── requirements.txt     ← Python packages
├── README.md
├── models/              ← Download model files here
└── utils/
    ├── yolo_utils.py    ← YOLO detection
    ├── clip_utils.py    ← CLIP encoding
    ├── faiss_utils.py   ← FAISS search
    └── display_utils.py ← Results display
```

---

## Ablation Results

| Config | Description | Recall@10 | mAP@10 |
|--------|-------------|-----------|--------|
| A | Frozen CLIP, image only (α=1.0) | 56.5% | 0.164 |
| B | Frozen CLIP + BLIP-2 captions (α=0.7) | 56.3% | 0.164 |
| **C** | **Fine-tuned CLIP + BLIP-2 captions (α=0.7)** | **91.4%** | **0.554** |

Seeds: [550, 537, 585, 35]
