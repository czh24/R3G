# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from typing import Dict

from retrieve import ImageSimilarityRetrieval
from reasoning_generator import LLaVAReasoningPlanner
from image_score import LLaVAJudgeEvaluator
from image_choose import fuse_and_select_top1
from image_answer import LLaVAAnswerGenerator


def main():
    ap = argparse.ArgumentParser()

    # data & retrieval
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--metadata_pkl", required=True)
    ap.add_argument("--query_image_root", required=True)
    ap.add_argument("--encoder_type", choices=["eva", "uniir"], required=True)
    ap.add_argument("--eva_model_path", default="")
    ap.add_argument("--local_retriever_module_dir", default="")
    ap.add_argument("--uniir_checkpoint_path", default="")
    ap.add_argument("--clip_model_id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--num_samples", type=int, default=0)

    # llava
    ap.add_argument("--llava_model", required=True)
    ap.add_argument("--output_jsonl", required=True)

    # fusion weights
    ap.add_argument("--lambda_r", type=float, default=0.20)
    ap.add_argument("--lambda_t", type=float, default=0.35)
    ap.add_argument("--lambda_a", type=float, default=0.45)

    args = ap.parse_args()

    retrieval = ImageSimilarityRetrieval(
        faiss_index_path=args.faiss_index,
        metadata_path=args.metadata_pkl,
        parquet_dir=args.parquet_dir,
        query_image_root=args.query_image_root,
        encoder_type=args.encoder_type,
        device="cuda",
        eva_model_path=args.eva_model_path,
        local_retriever_module_dir=args.local_retriever_module_dir,
        uniir_checkpoint_path=args.uniir_checkpoint_path,
        clip_model_id=args.clip_model_id,
    )
    data = retrieval.load_parquet_data()
    if args.num_samples:
        data = data[: args.num_samples]

    planner = LLaVAReasoningPlanner(model_path=args.llava_model)
    judge = LLaVAJudgeEvaluator(model_path=args.llava_model)
    answerer = LLaVAAnswerGenerator(model_path=args.llava_model)

    weights = (args.lambda_r, args.lambda_t, args.lambda_a)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for item in data:
            rec = retrieval.retrieve_for_item(item, top_k=args.top_k)

            qid = rec["id"]
            q = rec["question"]
            choices = rec["choices"]
            gold = rec.get("answer", "")
            scenario = rec.get("scenario", "")
            aspect = rec.get("aspect", "")
            qimg = rec["query_image_path"]

            retrieved = rec["retrieval_result"]["retrieved_images"]

            reasoning_steps = planner.plan(question=q, query_image_path=qimg)

            judge_scores_by_path: Dict[str, Dict[str, float]] = {}
            for cand in retrieved:
                cpath = cand["path"]
                try:
                    js = judge.score_pair(
                        question=q,
                        query_image_path=qimg,
                        candidate_image_path=cpath,
                        reasoning_steps=reasoning_steps,
                    )
                    judge_scores_by_path[cpath] = js.to_dict()
                except Exception:
                    judge_scores_by_path[cpath] = {"r": 0.0, "t": 0.0, "a": 0.0}

            evidence_path = None
            if retrieved:
                retrieval_item_for_choose = {
                    "id": qid,
                    "retrieved_images": [{"path": c["path"], "similarity": float(c.get("similarity", 0.0))} for c in retrieved],
                }
                selected = fuse_and_select_top1(retrieval_item_for_choose, judge_scores_by_path, weights=weights)
                evidence_path = selected["final_image"]

            try:
                pred = answerer.answer(
                    question=q,
                    choices=choices,
                    query_image_path=qimg,
                    evidence_image_path=evidence_path,
                    reasoning_steps=reasoning_steps,
                )
            except Exception:
                pred = ""

            out = {
                "id": qid,
                "pred": pred,
                "gold": gold,
                "scenario": scenario,
                "aspect": aspect,
                "query_image": qimg,
                "selected_image": evidence_path or "",
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
