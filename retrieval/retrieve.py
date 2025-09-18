import os
import sys
import json
import pickle
import glob
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import faiss
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

SIMILARITY_SKIP_THRESHOLD = 0.90
SEARCH_CANDIDATE_BUFFER = 50


def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_list(item) for item in obj)
    else:
        return obj


class LocalCLIPVisionModel(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.vision_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, pixel_values):
        return self.vision_model(pixel_values)


class LocalUniIRRetriever:
    def __init__(self, device="cuda", checkpoint_path=None, base_model_name="openai/clip-vit-large-patch14"):
        self.device = device
        self.processor = None
        self.transform = None
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.model = CLIPModel.from_pretrained(base_model_name)
                self.processor = CLIPProcessor.from_pretrained(base_model_name)
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError("Checkpoint not found or not provided.")
        except Exception:
            self.model = LocalCLIPVisionModel(embed_dim=768)
            self.processor = None
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, image_paths: List[str]) -> np.ndarray:
        features = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                if self.processor is not None:
                    inputs = self.processor(images=[image], return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy())
                else:
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    image_features = self.model(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy())
            except Exception:
                features.append(np.zeros((1, 768), dtype=np.float32))
        return np.vstack(features) if features else np.array([])


class LocalEVACLIPRetriever:
    def __init__(self, device="cuda", model_path=None, extra_module_path=None, fallback_base_model="openai/clip-vit-large-patch14"):
        self.device = device
        self.encoder = None
        self.model = None
        self.processor = None
        try:
            if extra_module_path and os.path.isdir(extra_module_path):
                if extra_module_path not in sys.path:
                    sys.path.append(extra_module_path)
            from local_retriever import EVACLIPImageEncoder
            self.encoder = EVACLIPImageEncoder(model_path=model_path, device=self.device)
        except Exception:
            try:
                self.model = CLIPModel.from_pretrained(fallback_base_model)
                self.processor = CLIPProcessor.from_pretrained(fallback_base_model)
                self.model.to(self.device).eval()
            except Exception:
                self.model = LocalCLIPVisionModel(embed_dim=768)
                self.processor = None
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.model.to(self.device).eval()

    @torch.no_grad()
    def get_image_features(self, image_paths: List[str]) -> np.ndarray:
        if self.encoder is not None:
            images = []
            for p in image_paths:
                img = Image.open(p).convert("RGB")
                images.append(img)
            feats = self.encoder.encode_images(images)
            return feats
        features = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                if self.processor is not None:
                    inputs = self.processor(images=[image], return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_features = self.model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy())
                else:
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    image_features = self.model(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy())
            except Exception:
                features.append(np.zeros((1, 768), dtype=np.float32))
        return np.vstack(features) if features else np.array([])


class ImageSimilarityRetrieval:
    def __init__(self, faiss_index_path: str, metadata_path: str, parquet_dir: str, device: str = "cuda", encoder: str = "eva", eva_model_path: str = None, eva_extra_module_path: str = None, uniir_ckpt_path: str = None):
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.parquet_dir = parquet_dir
        self.device = device
        self.encoder_type = encoder
        self.eva_model_path = eva_model_path
        self.eva_extra_module_path = eva_extra_module_path
        self.uniir_ckpt_path = uniir_ckpt_path
        self.encoder = None
        self.faiss_index = None
        self.image_metadata = None
        self.parquet_data = None
        self.vec_dim = None
        self._load_components()

    def _load_components(self):
        if self.encoder_type == "eva":
            self.encoder = LocalEVACLIPRetriever(device=self.device, model_path=self.eva_model_path, extra_module_path=self.eva_extra_module_path)
        elif self.encoder_type == "uniir":
            self.encoder = LocalUniIRRetriever(device=self.device, checkpoint_path=self.uniir_ckpt_path)
        else:
            self.encoder = LocalEVACLIPRetriever(device=self.device)
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        self.vec_dim = int(self.faiss_index.d)
        with open(self.metadata_path, "rb") as f:
            self.image_metadata = pickle.load(f)
        self.parquet_data = self._load_parquet_data()

    def _load_parquet_data(self) -> List[Dict[str, Any]]:
        all_data = []
        parquet_files = glob.glob(os.path.join(self.parquet_dir, "*.parquet"))
        for file_path in tqdm(parquet_files, desc="Loading parquet", ncols=80):
            try:
                df = pd.read_parquet(file_path)
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
                        "gt_images": row.get("gt_images", [])
                    })
            except Exception:
                continue
        return all_data

    def _extract_image_features(self, image_path: str) -> np.ndarray:
        try:
            features = self.encoder.get_image_features([image_path])
            if features is None or len(features) == 0:
                return np.zeros(self.vec_dim, dtype=np.float32)
            feat = features[0]
            if feat.ndim > 1:
                feat = feat.squeeze()
            if feat.shape[-1] != self.vec_dim:
                feat = feat.astype(np.float32)
                if feat.ndim == 1 and feat.shape[0] != self.vec_dim:
                    z = np.zeros(self.vec_dim, dtype=np.float32)
                    n = min(self.vec_dim, feat.shape[0])
                    z[:n] = feat[:n]
                    feat = z
            return feat.astype(np.float32)
        except Exception:
            return np.zeros(self.vec_dim, dtype=np.float32)

    def _search_similar_images(self, query_features: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        try:
            if query_features.ndim == 1:
                query_features = query_features.reshape(1, -1)
            query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
            similarities, indices = self.faiss_index.search(query_features.astype(np.float32), top_k)
            results = []
            for i in range(len(indices[0])):
                if indices[0][i] != -1:
                    results.append((int(indices[0][i]), float(similarities[0][i])))
            return results
        except Exception:
            return []

    def retrieve_similar_images(self, query_image_path: str, top_k: int = 5, similarity_skip_threshold: float = SIMILARITY_SKIP_THRESHOLD, search_candidate_buffer: int = SEARCH_CANDIDATE_BUFFER) -> Dict[str, Any]:
        try:
            query_features = self._extract_image_features(query_image_path)
            search_k = top_k + search_candidate_buffer
            similar_results = self._search_similar_images(query_features, search_k)
            filtered_results = [(idx, sim) for idx, sim in similar_results if sim <= similarity_skip_threshold]
            if len(filtered_results) < top_k:
                search_k = max(search_k * 2, top_k + search_candidate_buffer)
                similar_results = self._search_similar_images(query_features, search_k)
                filtered_results = [(idx, sim) for idx, sim in similar_results if sim <= similarity_skip_threshold]
            final_results = filtered_results[:top_k]
            retrieved_images = []
            for idx, similarity in final_results:
                if idx < len(self.image_metadata):
                    img_info = self.image_metadata[idx]
                    retrieved_images.append({
                        "image_path": img_info.get("path", ""),
                        "filename": img_info.get("filename", ""),
                        "similarity": similarity,
                        "category": img_info.get("category", "unknown")
                    })
            return {
                "query_image": query_image_path,
                "retrieved_images": retrieved_images,
                "total_found": len(retrieved_images)
            }
        except Exception as e:
            return {
                "query_image": query_image_path,
                "retrieved_images": [],
                "total_found": 0,
                "error": str(e)
            }

    def process_all_parquet_data(self, output_file: str, top_k: int = 5, similarity_skip_threshold: float = SIMILARITY_SKIP_THRESHOLD, search_candidate_buffer: int = SEARCH_CANDIDATE_BUFFER, save_every_n: int = 50, save_every_seconds: int = 30) -> Dict[str, Any]:
        print("Starting batch similarity retrieval")
        print(f"Total items: {len(self.parquet_data):,}")
        results = {
            "total_processed": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "results": []
        }
        start_time = time.time()
        last_save_time = start_time
        main_pbar = tqdm(
            total=len(self.parquet_data),
            desc="Retrieval",
            ncols=100,
            unit="it",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        for i, data_item in enumerate(self.parquet_data):
            try:
                current_time = time.time()
                elapsed_time = current_time - start_time
                avg_time_per_item = elapsed_time / (i + 1) if i > 0 else 0
                main_pbar.set_postfix({
                    "ok": results["successful_retrievals"],
                    "fail": results["failed_retrievals"],
                    "avg": f"{avg_time_per_item:.2f}s/it"
                })
                image_filename = data_item.get("image_path", "")
                if image_filename:
                    full_image_path = image_filename if os.path.exists(image_filename) else image_filename
                    if os.path.exists(full_image_path):
                        retrieval_result = self.retrieve_similar_images(full_image_path, top_k=top_k, similarity_skip_threshold=similarity_skip_threshold, search_candidate_buffer=search_candidate_buffer)
                        result_item = {
                            "id": data_item.get("id", f"item_{i}"),
                            "question": data_item.get("question", ""),
                            "choices": {
                                "A": data_item.get("A", ""),
                                "B": data_item.get("B", ""),
                                "C": data_item.get("C", ""),
                                "D": data_item.get("D", "")
                            },
                            "answer": data_item.get("answer", ""),
                            "aspect": data_item.get("aspect", ""),
                            "scenario": data_item.get("scenario", ""),
                            "query_image_path": full_image_path,
                            "gt_images": data_item.get("gt_images", []),
                            "retrieval_result": retrieval_result
                        }
                        results["results"].append(result_item)
                        results["successful_retrievals"] += 1
                    else:
                        main_pbar.write(f"Missing image: {full_image_path}")
                        results["failed_retrievals"] += 1
                else:
                    main_pbar.write(f"Missing image path at item {i}")
                    results["failed_retrievals"] += 1
                results["total_processed"] += 1
                if ((i + 1) % save_every_n == 0) or (current_time - last_save_time > save_every_seconds):
                    temp_file = output_file.replace(".json", "_temp.json")
                    try:
                        serializable_temp_results = convert_numpy_to_list(results)
                        with open(temp_file, "w", encoding="utf-8") as f:
                            json.dump(serializable_temp_results, f, ensure_ascii=False, indent=2)
                        last_save_time = current_time
                        main_pbar.write(f"Saved interim results at {i + 1}")
                    except Exception as save_e:
                        main_pbar.write(f"Failed to save interim results: {save_e}")
                main_pbar.update(1)
            except Exception as e:
                main_pbar.write(f"Failed processing item {i}: {e}")
                results["failed_retrievals"] += 1
                results["total_processed"] += 1
                main_pbar.update(1)
                continue
        main_pbar.close()
        print(f"Saving final results to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        serialized_results = convert_numpy_to_list(results)
        if "results" in serialized_results and serialized_results["results"]:
            serialized_results["results"].sort(key=lambda x: x.get("id", ""))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serialized_results, f, ensure_ascii=False, indent=2)
        temp_file = output_file.replace(".json", "_temp.json")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        total_time = time.time() - start_time
        avg_time = total_time / results["total_processed"] if results["total_processed"] > 0 else 0
        print("Done")
        print(f"Processed: {results['total_processed']:,}")
        print(f"Success: {results['successful_retrievals']:,}")
        print(f"Failed: {results['failed_retrievals']:,}")
        print(f"Success rate: {results['successful_retrievals']/results['total_processed']*100:.2f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time: {avg_time:.2f}s/it")
        return results

    def evaluate_retrieval_performance(self, results_file: str) -> Dict[str, float]:
        print("Evaluating retrieval performance")
        if not os.path.exists(results_file):
            print(f"Results file not found: {results_file}")
            return {}
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        total_queries = len(results.get("results", []))
        if total_queries == 0:
            print("No results found")
            return {}
        successful_retrievals = 0
        total_similarities = []
        category_performance = {}
        for result in tqdm(results["results"], desc="Analyzing", ncols=80):
            retrieval_result = result.get("retrieval_result", {})
            retrieved_images = retrieval_result.get("retrieved_images", [])
            if retrieved_images:
                successful_retrievals += 1
                for img in retrieved_images:
                    total_similarities.append(img.get("similarity", 0.0))
                scenario = result.get("scenario", "unknown")
                if scenario not in category_performance:
                    category_performance[scenario] = {"total": 0, "successful": 0}
                category_performance[scenario]["total"] += 1
                category_performance[scenario]["successful"] += 1
            else:
                scenario = result.get("scenario", "unknown")
                if scenario not in category_performance:
                    category_performance[scenario] = {"total": 0, "successful": 0}
                category_performance[scenario]["total"] += 1
        performance_metrics = {
            "total_queries": total_queries,
            "successful_retrievals": successful_retrievals,
            "success_rate": successful_retrievals / total_queries if total_queries > 0 else 0.0,
            "average_similarity": float(np.mean(total_similarities)) if total_similarities else 0.0,
            "median_similarity": float(np.median(total_similarities)) if total_similarities else 0.0,
            "min_similarity": float(np.min(total_similarities)) if total_similarities else 0.0,
            "max_similarity": float(np.max(total_similarities)) if total_similarities else 0.0
        }
        per_category = {}
        for category, stats in category_performance.items():
            total = stats["total"]
            succ = stats["successful"]
            per_category[category] = {
                "total": total,
                "successful": succ,
                "success_rate": (succ / total) if total > 0 else 0.0
            }
        out = {"overall": performance_metrics, "by_category": per_category}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return performance_metrics


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(prog="retrieve", add_help=True)
    parser.add_argument("--mode", type=str, default="process", choices=["process", "evaluate", "single"])
    parser.add_argument("--faiss-index", type=str, required=False)
    parser.add_argument("--metadata", type=str, required=False)
    parser.add_argument("--parquet-dir", type=str, required=False)
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder", type=str, default="eva", choices=["eva", "uniir", "auto"])
    parser.add_argument("--eva-model-path", type=str, default=None)
    parser.add_argument("--eva-extra-module-path", type=str, default=None)
    parser.add_argument("--uniir-ckpt-path", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--similarity-skip-threshold", type=float, default=SIMILARITY_SKIP_THRESHOLD)
    parser.add_argument("--search-candidate-buffer", type=int, default=SEARCH_CANDIDATE_BUFFER)
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--query-image", type=str, default=None)
    return parser.parse_args()


def _require_paths(args, require_parquet: bool, require_output: bool):
    missing = []
    if not args.faiss_index:
        missing.append("--faiss-index")
    if not args.metadata:
        missing.append("--metadata")
    if require_parquet and not args.parquet_dir:
        missing.append("--parquet-dir")
    if require_output and not args.output:
        missing.append("--output")
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")


def main():
    args = _parse_args()
    if args.mode == "single":
        _require_paths(args, require_parquet=False, require_output=False)
        if not args.query_image:
            raise SystemExit("Missing required argument: --query-image")
        retriever = ImageSimilarityRetrieval(
            faiss_index_path=args.faiss_index,
            metadata_path=args.metadata,
            parquet_dir=args.parquet_dir or ".",
            device=args.device,
            encoder=args.encoder,
            eva_model_path=args.eva_model_path,
            eva_extra_module_path=args.eva_extra_module_path,
            uniir_ckpt_path=args.uniir_ckpt_path
        )
        res = retriever.retrieve_similar_images(
            args.query_image,
            top_k=args.top_k,
            similarity_skip_threshold=args.similarity_skip_threshold,
            search_candidate_buffer=args.search_candidate_buffer
        )
        print(json.dumps(convert_numpy_to_list(res), indent=2, ensure_ascii=False))
        return
    if args.mode == "process":
        _require_paths(args, require_parquet=True, require_output=True)
        retriever = ImageSimilarityRetrieval(
            faiss_index_path=args.faiss_index,
            metadata_path=args.metadata,
            parquet_dir=args.parquet_dir,
            device=args.device,
            encoder=args.encoder,
            eva_model_path=args.eva_model_path,
            eva_extra_module_path=args.eva_extra_module_path,
            uniir_ckpt_path=args.uniir_ckpt_path
        )
        retriever.process_all_parquet_data(
            output_file=args.output,
            top_k=args.top_k,
            similarity_skip_threshold=args.similarity_skip_threshold,
            search_candidate_buffer=args.search_candidate_buffer
        )
        return
    if args.mode == "evaluate":
        results_file = args.results_file or args.output
        if not results_file:
            raise SystemExit("Missing required argument: --results-file or --output")
        retriever = ImageSimilarityRetrieval(
            faiss_index_path=args.faiss_index,
            metadata_path=args.metadata,
            parquet_dir=args.parquet_dir or ".",
            device=args.device,
            encoder=args.encoder,
            eva_model_path=args.eva_model_path,
            eva_extra_module_path=args.eva_extra_module_path,
            uniir_ckpt_path=args.uniir_ckpt_path
        )
        retriever.evaluate_retrieval_performance(results_file=results_file)
        return


if __name__ == "__main__":
    main()