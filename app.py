"""
Visual Product Search Engine — Streamlit Demo App
===================================================
Run with:   streamlit run app.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

# ── MUST BE FIRST STREAMLIT COMMAND ──────────────────────────────────────────
st.set_page_config(
    page_title="Visual Product Search Engine",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .title-text  { font-size:2.2rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem; }
    .sub-text    { font-size:1rem;   color:#666;      margin-bottom:1.5rem; }
    .step-header { background:#e8f4fd; border-left:4px solid #2196F3;
                   padding:0.5rem 1rem; border-radius:4px;
                   font-weight:600; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

from utils.yolo_utils     import load_yolo,       run_yolo_crop
from utils.clip_utils     import load_clip,       embed_query
from utils.faiss_utils    import load_index,      query_index
from utils.display_utils  import render_results
from utils.blip2_itm_utils import load_itm_model, rerank_with_itm

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")

YOLO_PT         = os.path.join(MODELS_DIR, "best.pt")
CLIP_PT         = os.path.join(MODELS_DIR, "clip_finetuned_full.pt")
FAISS_BIN       = os.path.join(MODELS_DIR, "faiss_index_C_alpha07.bin")
GALLERY_CSV     = os.path.join(MODELS_DIR, "gallery_metadata.csv")
GALLERY_IMG_DIR = os.path.join(MODELS_DIR, "gallery_crops")
CAPTIONS_JSON   = os.path.join(MODELS_DIR, "gallery_captions.json")


# ── Check required files ──────────────────────────────────────────────────────
def check_files():
    missing = []
    for label, path in [
        ("YOLO weights       → models/best.pt",                   YOLO_PT),
        ("CLIP weights       → models/clip_finetuned_full.pt",    CLIP_PT),
        ("FAISS index        → models/faiss_index_C_alpha07.bin", FAISS_BIN),
        ("Gallery metadata   → models/gallery_metadata.csv",      GALLERY_CSV),
        ("Gallery captions   → models/gallery_captions.json",     CAPTIONS_JSON),
    ]:
        if not os.path.exists(path):
            missing.append(label)
    return missing


# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model...")
def get_yolo():
    return load_yolo(YOLO_PT)

@st.cache_resource(show_spinner="Loading CLIP model...")
def get_clip():
    return load_clip(CLIP_PT)

@st.cache_resource(show_spinner="Loading FAISS index...")
def get_faiss():
    return load_index(FAISS_BIN)

@st.cache_data(show_spinner="Loading gallery metadata...")
def get_metadata():
    return pd.read_csv(GALLERY_CSV)

@st.cache_data(show_spinner="Loading gallery captions...")
def get_captions():
    with open(CAPTIONS_JSON, "r") as f:
        return json.load(f)

@st.cache_resource(show_spinner="Loading BLIP-2 ITM re-ranking model...")
def get_itm():
    return load_itm_model(cache_dir=MODELS_DIR)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_k      = st.slider("Top-K results",          min_value=1,  max_value=20, value=5)
    conf_thres = st.slider("YOLO confidence cutoff", min_value=0.1, max_value=0.9, value=0.3, step=0.05)

    st.divider()
    st.markdown("**Re-ranking**")
    use_reranking = st.checkbox("Enable BLIP-2 ITM re-ranking (Step 4)", value=True)
    st.caption("Re-ranks top-K results using image-text matching score")

    st.divider()
    st.markdown("**Model details**")
    st.markdown("- Backbone: `clip-vit-base-patch32`")
    st.markdown("- Fine-tuning: 10 epochs, InfoNCE loss")
    st.markdown("- Index: Config C (α = 0.7)")
    st.markdown("- Gallery: 12,612 images")
    st.markdown("- Re-ranking: BLIP ITM")

    st.divider()
    st.markdown("**Ablation Results**")
    abl = pd.DataFrame({
        "Config":    ["A — image only", "B — α=0.7", "C — α=0.7 ★"],
        "Recall@10": ["56.5%",          "56.3%",     "91.4%"],
        "mAP@10":    ["0.164",          "0.164",     "0.554"],
    })
    st.dataframe(abl, hide_index=True, use_container_width=True)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="title-text">👗 Visual Product Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a clothing image → YOLO crops it → CLIP retrieves → BLIP-2 re-ranks</div>', unsafe_allow_html=True)

# ── Setup check ───────────────────────────────────────────────────────────────
missing = check_files()
if missing:
    st.error("**Missing files. Download from Kaggle and place in `models/` folder:**")
    for m in missing:
        st.error(f"  ✗  {m}")
    with st.expander("Setup instructions"):
        st.markdown("""
        | File | Kaggle dataset |
        |------|---------------|
        | `best.pt` | `harshitabansal307/yolo-bbox-crops-v1` |
        | `clip_finetuned_full.pt` | `likithareddy2508/clip-finetuned-indexes-c` |
        | `faiss_index_C_alpha07.bin` | `likithareddy2508/clip-finetuned-indexes-c` |
        | `gallery_metadata.csv` | `likithareddy2508/clip-indexes-ab` |
        | `gallery_captions.json` | `harshitabansal307/blipcaptionsoutput` |
        """)
    st.stop()

# ── Load all models ───────────────────────────────────────────────────────────
yolo_model            = get_yolo()
clip_model, clip_proc = get_clip()
faiss_index           = get_faiss()
gallery_meta          = get_metadata()
gallery_captions      = get_captions()
itm_model, itm_proc   = get_itm()

st.success("✅ All models loaded and ready!")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("stage",        "upload"),
    ("orig_img",     None),
    ("cropped_img",  None),
    ("crop_bbox",    None),
    ("yolo_ok",      False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Upload
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.stage == "upload":
    st.markdown('<div class="step-header">Step 1 — Upload a clothing image</div>', unsafe_allow_html=True)

    col_upload, col_tips = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded:
            from PIL import Image
            img = Image.open(uploaded).convert("RGB")
            st.session_state.orig_img = img
            st.image(img, caption="Uploaded image", use_container_width=True)

            if st.button("🔍  Detect & crop product", type="primary", use_container_width=True):
                with st.spinner("Step 1 — Running YOLO detection..."):
                    cropped, bbox, success = run_yolo_crop(
                        yolo_model, img, conf_threshold=conf_thres
                    )
                st.session_state.cropped_img = cropped
                st.session_state.crop_bbox   = bbox
                st.session_state.yolo_ok     = success
                st.session_state.stage       = "confirm"
                st.rerun()

    with col_tips:
        st.markdown("**Tips for best results:**")
        st.markdown("- Single clothing item clearly visible")
        st.markdown("- Front-facing, minimal background clutter")
        st.markdown("- Works with shop images or real-world photos")
        st.markdown("")
        st.markdown("**Pipeline (from PDF):**")
        st.markdown("1. **YOLO** — detect & crop clothing item")
        st.markdown("2. **CLIP** — encode crop to 512-dim vector")
        st.markdown("3. **FAISS** — find top-K similar gallery items")
        st.markdown("4. **BLIP-2 ITM** — re-rank by image-text matching")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Confirm crop
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "confirm":
    st.markdown('<div class="step-header">Step 2 — Confirm the crop</div>', unsafe_allow_html=True)

    col_orig, col_crop = st.columns([1, 1])

    with col_orig:
        st.markdown("**Original image**")
        st.image(st.session_state.orig_img, use_container_width=True)

    with col_crop:
        st.markdown("**YOLO crop**")
        if st.session_state.yolo_ok:
            st.success("✓ YOLO detected clothing item")
        else:
            st.warning("⚠ Low confidence — using full image as fallback")
        st.image(st.session_state.cropped_img, use_container_width=True)
        if st.session_state.crop_bbox:
            x1, y1, x2, y2 = st.session_state.crop_bbox
            st.caption(f"Crop region: ({x1}, {y1}) → ({x2}, {y2})")

    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("✅  Confirm — Search!", type="primary", use_container_width=True):
            st.session_state.stage = "results"
            st.rerun()

    with c2:
        if st.button("🔄  Use full image instead", use_container_width=True):
            st.session_state.cropped_img = st.session_state.orig_img
            st.session_state.crop_bbox   = None
            st.session_state.yolo_ok     = False
            st.session_state.stage       = "results"
            st.rerun()

    with c3:
        if st.button("← Upload a different image", use_container_width=True):
            st.session_state.stage       = "upload"
            st.session_state.orig_img    = None
            st.session_state.cropped_img = None
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Results (with BLIP-2 ITM re-ranking)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "results":
    st.markdown('<div class="step-header">Step 3 & 4 — CLIP Retrieval + BLIP-2 ITM Re-ranking</div>', unsafe_allow_html=True)

    with st.spinner("Step 2 — Encoding query with CLIP..."):
        t0        = time.time()
        query_vec = embed_query(clip_model, clip_proc, st.session_state.cropped_img)
        t_clip    = time.time() - t0

    # Retrieve more than top_k so re-ranking has candidates to work with
    retrieve_n = max(top_k * 3, 15)

    with st.spinner("Step 3 — Searching FAISS index..."):
        t0 = time.time()
        distances, indices = query_index(faiss_index, query_vec, top_k=retrieve_n)
        t_faiss = time.time() - t0

    # Step 4 — BLIP-2 ITM re-ranking
    if use_reranking:
        with st.spinner(f"Step 4 — BLIP-2 ITM re-ranking top {retrieve_n} candidates..."):
            t0 = time.time()
            final_indices, itm_scores, clip_scores = rerank_with_itm(
                itm_model        = itm_model,
                itm_processor    = itm_proc,
                query_image      = st.session_state.cropped_img,
                candidate_indices  = indices[0],
                candidate_distances= distances[0],
                gallery_meta     = gallery_meta,
                gallery_captions = gallery_captions,
            )
            t_itm = time.time() - t0
            # Take only top_k after re-ranking
            final_indices = final_indices[:top_k]
            final_scores  = itm_scores[:top_k]
            clip_scores   = clip_scores[:top_k]
        reranked = True
    else:
        final_indices = indices[0][:top_k]
        final_scores  = distances[0][:top_k]
        clip_scores   = distances[0][:top_k]
        t_itm         = 0
        reranked      = False

    elapsed = t_clip + t_faiss + t_itm

    # ── Display ───────────────────────────────────────────────────────────────
    col_q, col_stats = st.columns([1, 3])

    with col_q:
        st.markdown("**Query crop**")
        st.image(st.session_state.cropped_img, width=160)

    with col_stats:
        st.markdown("**Pipeline summary**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("CLIP encode", f"{t_clip:.2f}s")
        s2.metric("FAISS search", f"{t_faiss:.2f}s")
        s3.metric("ITM re-rank", f"{t_itm:.2f}s" if reranked else "skipped")
        s4.metric("Total time", f"{elapsed:.2f}s")

        if reranked:
            st.success(f"✓ BLIP-2 ITM re-ranking applied on top {retrieve_n} candidates")
        else:
            st.info("ℹ BLIP-2 ITM re-ranking disabled (toggle in sidebar)")

    st.divider()

    score_label = "ITM score" if reranked else "CLIP score"
    st.markdown(f"**Top {top_k} results** (ranked by {score_label}):")

    render_results(
        indices     = final_indices,
        distances   = final_scores,
        meta_df     = gallery_meta,
        gallery_dir = GALLERY_IMG_DIR,
        top_k       = top_k,
    )

    # Show comparison — CLIP order vs ITM order
    if reranked:
        with st.expander("📊 See how re-ranking changed the order"):
            comp_data = []
            for rank, (fi, its, cs) in enumerate(zip(final_indices, final_scores, clip_scores)):
                if fi < len(gallery_meta):
                    row = gallery_meta.iloc[int(fi)]
                    comp_data.append({
                        "Rank after ITM": rank + 1,
                        "Item ID":        row.get("item_id", "—"),
                        "ITM score":      round(float(its), 4),
                        "CLIP score":     round(float(cs),  4),
                    })
            if comp_data:
                st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

    st.divider()
    if st.button("← New search", type="primary"):
        st.session_state.stage       = "upload"
        st.session_state.orig_img    = None
        st.session_state.cropped_img = None
        st.rerun()