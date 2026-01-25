# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import pickle
import glob
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import faiss
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

SIMILARITY_SKIP_THRESHOLD = 0.90
SEARCH_CANDIDATE_BUFFER = 50


class LocalCLIPVisionModel(nn.Module):
    """Fallback tiny vision model (kept for compatibility)."""
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.vision_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, pixel_values):
        return self.vision_model(pixel_values)


class LocalUniIRRetriever:
    """
    UniIR-style retriever using CLIP backbone + optional checkpoint weights.
    All paths must be provided by caller (no hard-coded paths).
    """
    def __init__(self, device: str, checkpoint_path: str = "", clip_model_id: str = "openai/clip-vit-large-patch14"):
        self.device = device
        self.checkpoint_path = checkpoint_path

        try:
            self.model = CLIPModel.from_pretrained(clip_model_id)
            self.processor = CLIPProcessor.from_pretrained(clip_model_id)

            if checkpoint_path:
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"UniIR checkpoint not found: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                if isinstance(state_dict, dict):
                    self.model.load_state_dict(state_dict, strict=False)
        except Exception:
            # fallback
            self.model = LocalCLIPVisionModel(embed_dim=768)
            self.processor = None
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, image_paths: List[str]) -> np.ndarray:
        feats = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                if self.processor is not None:
                    inputs = self.processor(images=[img], return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    f = self.model.get_image_features(**inputs)
                else:
                    x = self.transform(img).unsqueeze(0).to(self.device)
                    f = self.model(x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().numpy())
            except Exception:
                feats.append(np.zeros((1, 768), dtype=np.float32))
        return np.vstack(feats) if feats else np.array([], dtype=np.float32)


class LocalEVACLIPRetriever:
    """
    EVA-CLIP wrapper.
    Requires:
      - eva_model_path: EVA-CLIP model directory
      - local_retriever_module_dir: directory that contains local_retriever.py with EVACLIPImageEncoder
    """
    def __init__(self, device: str, eva_model_path: str, local_retriever_module_dir: str):
        import sys
        if local_retriever_module_dir and local_retriever_module_dir not in sys.path:
            sys.path.insert(0, local_retriever_module_dir)
        from local_retriever import EVACLIPImageEncoder  # type: ignore

        self.encoder = EVACLIPImageEncoder(model_path=eva_model_path, device=device)

    @torch.no_grad()
    def get_image_features(self, image_paths: List[str]) -> np.ndarray:
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        return self.encoder.encode_images(imgs)


class ImageSimilarityRetrieval:
    """
    Refactored retrieval system:
    - no intermediate file writes
    - configurable paths passed by args (use run.sh)
    - returns in-memory dicts compatible with downstream stages
    """
    def __init__(
        self,
        faiss_index_path: str,
        metadata_path: str,
        parquet_dir: str,
        query_image_root: str,
        encoder_type: str,
        device: str = "cuda",
        eva_model_path: str = "",
        local_retriever_module_dir: str = "",
        uniir_checkpoint_path: str = "",
        clip_model_id: str = "openai/clip-vit-large-patch14",
    ):
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.parquet_dir = parquet_dir
        self.query_image_root = query_image_root
        self.device = device

        enc = encoder_type.lower().strip()
        if enc in ("eva", "evaclip", "eva-clip"):
            if not eva_model_path:
                raise ValueError("eva_model_path is required for encoder_type=eva")
            if not local_retriever_module_dir:
                raise ValueError("local_retriever_module_dir is required for encoder_type=eva")
            self.encoder = LocalEVACLIPRetriever(device=device, eva_model_path=eva_model_path, local_retriever_module_dir=local_retriever_module_dir)
        elif enc in ("uniir", "clip"):
            self.encoder = LocalUniIRRetriever(device=device, checkpoint_path=uniir_checkpoint_path, clip_model_id=clip_model_id)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.faiss_index = faiss.read_index(self.faiss_index_path)
        self.vec_dim = int(self.faiss_index.d)

        with open(self.metadata_path, "rb") as f:
            self.image_metadata = pickle.load(f)
        if not isinstance(self.image_metadata, list):
            raise ValueError("metadata must be a list of dicts")

        self.parquet_data: List[Dict[str, Any]] = []

    def load_parquet_data(self) -> List[Dict[str, Any]]:
        all_data: List[Dict[str, Any]] = []
        parquet_files = glob.glob(os.path.join(self.parquet_dir, "*.parquet"))
        for fp in parquet_files:
            df = pd.read_parquet(fp)
            for _, row in df.iterrows():
                all_data.append({
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "A": row.get("A", ""),
                    "B": row.get("B", ""),
                    "C": row.get("C", ""),
                    "D": row.get("D", ""),
                    "answer": row.get("answer", ""),
                    "aspect": row.get("aspect", ""),
                    "scenario": row.get("scenario", ""),
                    "image_path": row.get("image", ""),
                    "gt_images": row.get("gt_images", []),
                })
        self.parquet_data = all_data
        return all_data

    def _resolve_query_image_path(self, image_field: str) -> str:
        if image_field and os.path.isabs(image_field) and os.path.exists(image_field):
            return image_field
        if self.query_image_root:
            p = os.path.join(self.query_image_root, image_field)
            if os.path.exists(p):
                return p
        return image_field

    def _extract_image_features(self, image_path: str) -> np.ndarray:
        feats = self.encoder.get_image_features([image_path])
        return feats[0] if len(feats) > 0 else np.zeros(self.vec_dim, dtype=np.float32)

    def _search_similar_images(self, query_features: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        norm = np.linalg.norm(query_features, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        query_features = query_features / norm
        sims, idxs = self.faiss_index.search(query_features.astype(np.float32), top_k)
        out = []
        for i in range(len(idxs[0])):
            if idxs[0][i] != -1:
                out.append((int(idxs[0][i]), float(sims[0][i])))
        return out

    def retrieve_similar_images(self, query_image_path: str, top_k: int = 5) -> Dict[str, Any]:
        qf = self._extract_image_features(query_image_path)
        search_k = top_k + SEARCH_CANDIDATE_BUFFER
        results = self._search_similar_images(qf, search_k)

        filtered = [(idx, sim) for idx, sim in results if sim <= SIMILARITY_SKIP_THRESHOLD]
        if len(filtered) < top_k:
            search_k = max(search_k * 2, top_k + SEARCH_CANDIDATE_BUFFER)
            results = self._search_similar_images(qf, search_k)
            filtered = [(idx, sim) for idx, sim in results if sim <= SIMILARITY_SKIP_THRESHOLD]

        final = filtered[:top_k]

        retrieved_images = []
        for idx, sim in final:
            if idx < len(self.image_metadata):
                info = self.image_metadata[idx]
                retrieved_images.append({
                    "path": info.get("path", ""),
                    "filename": info.get("filename", ""),
                    "similarity": float(sim),
                    "category": info.get("category", "unknown"),
                })

        return {"query_image": query_image_path, "retrieved_images": retrieved_images, "total_found": len(retrieved_images)}

    def retrieve_for_item(self, item: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        qimg = self._resolve_query_image_path(item.get("image_path", ""))
        retrieval = {"query_image": qimg, "retrieved_images": [], "total_found": 0}
        if qimg and os.path.exists(qimg):
            retrieval = self.retrieve_similar_images(qimg, top_k=top_k)

        return {
            "id": str(item.get("id", "")),
            "question": item.get("question", ""),
            "choices": {"A": item.get("A",""), "B": item.get("B",""), "C": item.get("C",""), "D": item.get("D","")},
            "answer": item.get("answer", ""),
            "aspect": item.get("aspect", ""),
            "scenario": item.get("scenario", ""),
            "query_image_path": qimg,
            "gt_images": item.get("gt_images", []),
            "retrieval_result": retrieval,
        }
