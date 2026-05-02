"""
utils/blip2_itm_utils.py — BLIP-2 ITM Re-ranking (Step 4 of Online Query Pipeline)

Uses lightweight BLIP-ITM model (~450MB) for fast CPU inference.
Only runs on top-K results (5-15 images), so it's very fast.
"""

import os
import torch
import numpy as np
from PIL import Image


ITM_MODEL_NAME = "Salesforce/blip-itm-base-coco"


def load_itm_model(cache_dir: str):
    """
    Load BLIP ITM model for re-ranking.
    Saves to cache_dir so second run is instant.

    Returns: (model, processor)
    """
    from transformers import BlipProcessor, BlipForImageTextRetrieval

    itm_cache = os.path.join(cache_dir, "blip_itm_cache")
    os.makedirs(itm_cache, exist_ok=True)

    config_file = os.path.join(itm_cache, "config.json")

    if os.path.exists(config_file):
        print(f"[ITM] Loading from local cache: {itm_cache}")
        processor = BlipProcessor.from_pretrained(itm_cache, local_files_only=True)
        model     = BlipForImageTextRetrieval.from_pretrained(itm_cache, local_files_only=True)
        print("[ITM] Loaded from cache ✓")
    else:
        print(f"[ITM] First run — downloading {ITM_MODEL_NAME} (~450MB)...")
        processor = BlipProcessor.from_pretrained(ITM_MODEL_NAME)
        model     = BlipForImageTextRetrieval.from_pretrained(ITM_MODEL_NAME)
        print("[ITM] Saving to local cache...")
        processor.save_pretrained(itm_cache)
        model.save_pretrained(itm_cache)
        print(f"[ITM] Saved to {itm_cache} ✓")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print("[ITM] Ready ✓")
    return model, processor


def rerank_with_itm(
    itm_model,
    itm_processor,
    query_image: Image.Image,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    gallery_meta,
    gallery_captions: dict,
):
    """
    Re-rank CLIP top-K results using BLIP ITM scores.

    For each candidate:
    - Get its caption from gallery_captions
    - Compute ITM score: does query_image match this caption?
    - Re-sort by ITM score (descending)

    Parameters
    ----------
    query_image        : PIL Image — the YOLO-cropped query
    candidate_indices  : 1D array of FAISS hit positions
    candidate_distances: 1D array of CLIP cosine similarity scores
    gallery_meta       : gallery_metadata DataFrame
    gallery_captions   : dict {image_name: caption}

    Returns
    -------
    reranked_indices   : np.ndarray — re-sorted index positions
    reranked_scores    : np.ndarray — ITM scores (0-1)
    reranked_clips     : np.ndarray — original CLIP scores (for display)
    """
    itm_scores = []

    for i, faiss_pos in enumerate(candidate_indices):
        if faiss_pos >= len(gallery_meta):
            itm_scores.append(0.0)
            continue

        # Get caption for this gallery item
        row         = gallery_meta.iloc[int(faiss_pos)]
        image_name  = row.get("image_name", "")
        caption     = gallery_captions.get(image_name, "a clothing item")

        try:
            # Compute ITM score
            inputs = itm_processor(
                images=query_image,
                text=caption,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs    = itm_model(**inputs, use_itm_head=True)
                itm_logits = outputs.itm_score          # shape (1, 2)
                # Softmax → probability that image MATCHES text
                probs      = torch.softmax(itm_logits, dim=1)
                match_prob = float(probs[0][1].item())  # index 1 = match

        except Exception as e:
            print(f"[ITM] Error on item {i}: {e}")
            match_prob = float(candidate_distances[i])  # fallback to CLIP score

        itm_scores.append(match_prob)

    # Re-rank by ITM score (highest first)
    itm_scores  = np.array(itm_scores)
    sorted_order = np.argsort(itm_scores)[::-1]

    reranked_indices  = candidate_indices[sorted_order]
    reranked_scores   = itm_scores[sorted_order]
    reranked_clips    = candidate_distances[sorted_order]

    return reranked_indices, reranked_scores, reranked_clips