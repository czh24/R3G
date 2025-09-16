import os
import json
import pickle
from PIL import Image
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm

try:
    from local_retriever import EVACLIPImageEncoder
except Exception:
    EVACLIPImageEncoder = None


def build_universal_mrag_index(model_path, image_dir, output_dir, batch_size=16, device=None):
    if EVACLIPImageEncoder is None:
        raise RuntimeError("EVACLIPImageEncoder not found. Ensure 'local_retriever' module is importable on PYTHONPATH.")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading EVA-CLIP model: {model_path}")
    print(f"Using device: {device}")
    try:
        encoder = EVACLIPImageEncoder(model_path, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load EVA-CLIP model: {e}")
    all_images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images.append(
                {
                    "filename": filename,
                    "path": os.path.join(image_dir, filename),
                    "category": filename.split("_")[0] if "_" in filename else "unknown",
                }
            )
    print(f"Found {len(all_images)} images")
    if not all_images:
        raise RuntimeError("No images found to index.")
    def encode_images_batch(image_list):
        return encoder.encode_images(image_list)
    print("Encoding images...")
    image_vectors = []
    valid_images = []
    for i in tqdm(range(0, len(all_images), batch_size)):
        batch_info = all_images[i : i + batch_size]
        batch_images = []
        batch_valid = []
        for img_info in batch_info:
            try:
                image = Image.open(img_info["path"]).convert("RGB")
                batch_images.append(image)
                batch_valid.append(img_info)
            except Exception:
                continue
        if batch_images:
            try:
                vectors = encode_images_batch(batch_images)
                image_vectors.extend(vectors)
                valid_images.extend(batch_valid)
            except Exception:
                continue
    if not image_vectors or not valid_images:
        raise RuntimeError("No valid images were encoded.")
    print("Building FAISS index...")
    image_vectors = np.array(image_vectors).astype("float32")
    faiss.normalize_L2(image_vectors)
    dimension = image_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(image_vectors)
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(model_path)) or "model"
    slug = model_name.lower().replace(" ", "_")
    faiss_path = os.path.join(output_dir, f"images_{slug}.faiss")
    metadata_path = os.path.join(output_dir, f"metadata_{slug}.pkl")
    stats_path = os.path.join(output_dir, f"index_stats_{slug}.json")
    faiss.write_index(index, faiss_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(valid_images, f)
    stats = {
        "total_images": len(valid_images),
        "vector_dimension": int(dimension),
        "categories": list(sorted({img["category"] for img in valid_images})),
        "index_type": "IndexFlatIP",
        "model_used": model_name,
        "model_path": model_path,
        "device_used": device,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Index built successfully.")
    print(f"FAISS index: {faiss_path}")
    print(f"Image metadata: {metadata_path}")
    print(f"Stats: {stats_path}")
    print(f"Total {len(valid_images)} images, vector dimension: {dimension}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index for an image corpus using a local EVA-CLIP encoder.")
    parser.add_argument("--model-path", required=True, type=str, help="Path to the local EVA-CLIP model directory.")
    parser.add_argument("--image-dir", required=True, type=str, help="Directory containing images to index.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save the FAISS index and metadata.")
    parser.add_argument("--batch-size", default=16, type=int, help="Batch size for image encoding.")
    parser.add_argument("--device", default=None, type=str, choices=["cuda", "cpu"], help="Device to use. Defaults to CUDA if available.")
    args = parser.parse_args()
    build_universal_mrag_index(
        model_path=args.model_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )