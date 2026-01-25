# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, Tuple

def fuse_and_select_top1(
    retrieval_item: Dict[str, Any],
    judge_scores_by_path: Dict[str, Dict[str, float]],
    weights: Tuple[float, float, float] = (0.20, 0.35, 0.45),
) -> Dict[str, Any]:
    """
    Select best evidence image with: S = similarity + (λr*r + λt*t + λa*a)

    retrieval_item requires:
      - id
      - retrieved_images: [{"path":..., "similarity":...}, ...]
    judge_scores_by_path:
      {path: {"r":..., "t":..., "a":...}}
    """
    lam_r, lam_t, lam_a = weights
    best_total = None
    best_row = None

    for rec in retrieval_item.get("retrieved_images", []):
        img_path = rec["path"]
        sim = float(rec.get("similarity", 0.0))

        js = judge_scores_by_path.get(img_path, {"r": 0.0, "t": 0.0, "a": 0.0})
        r = float(js.get("r", 0.0))
        t = float(js.get("t", 0.0))
        a = float(js.get("a", 0.0))

        s2 = lam_r * r + lam_t * t + lam_a * a
        total = sim + s2

        if best_total is None or total > best_total:
            best_total = total
            best_row = {
                "id": str(retrieval_item.get("id")),
                "final_image": img_path,
                "similarity": round(sim, 6),
                "s2": round(s2, 6),
                "sum": round(total, 6),
                "judge": {"r": round(r, 6), "t": round(t, 6), "a": round(a, 6)},
            }

    if best_row is None:
        raise ValueError(f"No retrieved images for id={retrieval_item.get('id')}")
    return best_row
