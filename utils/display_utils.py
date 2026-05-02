"""utils/display_utils.py — Show retrieval results grid in Streamlit."""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def _find_image(image_name: str, gallery_dir: str):
    """
    Try to find the gallery crop on disk.
    image_name looks like: img/WOMEN/Dresses/id_00000002/02_1_front.jpg
    """
    if not gallery_dir or not os.path.isdir(gallery_dir):
        return None

    # Strip leading img/
    rel = image_name[4:] if image_name.startswith("img/") else image_name

    candidates = [
        os.path.join(gallery_dir, rel),
        os.path.join(gallery_dir, image_name),
        os.path.join(gallery_dir, os.path.basename(image_name)),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # Also try crop_path directly from metadata if column exists
    return None


def render_results(
    indices:     np.ndarray,
    distances:   np.ndarray,
    meta_df:     pd.DataFrame,
    gallery_dir: str,
    top_k:       int = 5,
):
    """
    Render retrieval results as a grid (max 5 columns per row).

    Parameters
    ----------
    indices     : 1D array of FAISS hit positions
    distances   : 1D array of cosine similarity scores
    meta_df     : gallery_metadata.csv as DataFrame
    gallery_dir : local folder containing gallery crop images
    top_k       : number of results to show
    """
    cols_per_row = min(top_k, 5)
    result_idx   = 0

    while result_idx < top_k and result_idx < len(indices):
        cols = st.columns(cols_per_row)

        for col in cols:
            if result_idx >= top_k or result_idx >= len(indices):
                break

            faiss_pos = int(indices[result_idx])
            score     = float(distances[result_idx])

            # Look up metadata
            if faiss_pos < len(meta_df):
                row        = meta_df.iloc[faiss_pos]
                item_id    = row.get("item_id",    "—")
                image_name = row.get("image_name", "")
                crop_path  = str(row.get("crop_path", ""))
            else:
                item_id = image_name = crop_path = ""

            # Score colour
            score_color = (
                "#28a745" if score >= 0.85 else
                "#fd7e14" if score >= 0.70 else
                "#6c757d"
            )

            with col:
                # Try to load image
                img_loaded = False

                # 1) Try crop_path from metadata
                if crop_path and os.path.exists(crop_path):
                    try:
                        st.image(Image.open(crop_path).convert("RGB"), use_container_width=True)
                        img_loaded = True
                    except Exception:
                        pass

                # 2) Try gallery_dir lookup by image_name
                if not img_loaded and image_name:
                    found = _find_image(image_name, gallery_dir)
                    if found:
                        try:
                            st.image(Image.open(found).convert("RGB"), use_container_width=True)
                            img_loaded = True
                        except Exception:
                            pass

                # 3) Placeholder
                if not img_loaded:
                    st.markdown(
                        "<div style='background:#f0f0f0;height:160px;"
                        "display:flex;align-items:center;justify-content:center;"
                        "border-radius:8px;color:#999;font-size:0.8rem;'>"
                        "Image not found</div>",
                        unsafe_allow_html=True,
                    )

                # Caption
                st.markdown(
                    f"<div style='text-align:center;padding:4px 0'>"
                    f"<b>Rank {result_idx + 1}</b><br>"
                    f"<span style='color:{score_color};font-weight:600'>"
                    f"Score: {score:.4f}</span><br>"
                    f"<span style='color:#555;font-size:0.75rem'>{item_id}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            result_idx += 1
