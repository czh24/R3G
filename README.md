<div align="center">
  <h1>R<sup>3</sup>G‑VQA (Minimal Release)</h1>
  <p><strong>Retrieve → Re‑rank → Reason</strong> (This repository includes: flow diagram, index-building script, stage‑1 retrieval script, and stage‑2 judging prompt)</p>
</div>

<p align="center">
  <img src="assets/overview.png" alt="R3G overview diagram" width="920"/>
</p>

## Quick Start (Step‑by‑Step)

> Objective: Create the environment → Download data and weights → Run Script 1 to build the index → Run Script 2 to retrieve and obtain Stage‑1 scores.

### 1) Create environment `R3G` (Python 3.10)
```bash
conda create -n R3G python=3.10 -y
conda activate R3G
pip install -r requirements.txt
```

(Optional) Install PyTorch (choose the wheel matching your GPU/CUDA version):
```bash
# Example command; please visit https://pytorch.org for the correct command for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2) Download the dataset (MRAG‑Bench)
- HF dataset page: <https://huggingface.co/datasets/uclanlp/MRAG-Bench>
- Place the extracted data at a path of your choice, e.g.: `/MRAG/dataset/`
- Must include at least:
  - `mrag_bench_image_corpus/` (image corpus)

### 3) Download EVA‑CLIP weights
- HF weights page: <https://huggingface.co/QuanSun/EVA-CLIP>

### 4) Build the index (Script 1: `retrieval/build_universal_mrag_index.py`)
This script: (1) extracts image‑corpus features with EVA‑CLIP; (2) builds a FAISS index. Example command (check `--help` in your script for exact argument names):
```bash
python retrieval/build_universal_mrag_index.py
```
Outputs (under `--out_dir`):
- `features.npy`, `ids.npy`, `faiss.index`

### 5) Stage‑1 retrieval and scoring (Script 2: `retrieval/retrieve.py`)
This script: (1) loads the index; (2) retrieves Top‑P candidates from the image corpus for each query image; (3) outputs Stage‑1 scores and the candidate list. Example command:
```bash
python retrieval/retrieve.py
```
Output: `val_topP.jsonl` (for each sample, the Top‑P candidates and the Stage‑1 score `s1(i)`).

---

## Stage‑2 Scoring (MLLM‑as‑Judge) Prompt
- File: `stage_2_prompt.txt` (provided, plain text)
- Purpose: For each candidate image, output three sub‑scores **(r, t, a)** and a fused score **s2(i)**; recommended weights: `λ_r=0.20, λ_t=0.35, λ_a=0.45`
- Usage: In your Stage‑2 implementation, directly read this text as the model prompt.

---

## Roadmap
- ✅ Flow diagram (overview.png)
- ✅ EVA‑CLIP + FAISS index building 
- ✅ Stage‑1 retrieval and scoring 
- ✅ Stage‑2 Prompt 
- ⏳ Stage‑2 scoring implementation (MLLM‑as‑Judge) — coming soon
- ⏳ R* (Reasoning‑Before‑Evidence) — coming soon
- ⏳ Answer generation and end‑to‑end evaluation — coming soon

---

## License and Citation
- License: To be added (will be provided later via a `LICENSE` file).
- If this repository/paper is useful to you, please cite it. BibTeX will be added after the camera‑ready version is released.
