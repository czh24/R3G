PARQUET_DIR="PATH_TO_PARQUET_DIR"
FAISS_INDEX="PATH_TO_FAISS_INDEX"
METADATA_PKL="PATH_TO_METADATA_PKL"
QUERY_IMAGE_ROOT="PATH_TO_QUERY_IMAGE_ROOT"

ENCODER_TYPE="eva"  # eva | uniir
EVA_MODEL_PATH="PATH_TO_EVA_MODEL"
UNIIR_CKPT="PATH_TO_UNIIR_CKPT"

LLAVA_MODEL="PATH_TO_LLAVA_MODEL"
OUTPUT_JSONL="PATH_TO_OUTPUT_JSONL"

python pipeline.py \
  --parquet_dir "$PARQUET_DIR" \
  --faiss_index "$FAISS_INDEX" \
  --metadata_pkl "$METADATA_PKL" \
  --query_image_root "$QUERY_IMAGE_ROOT" \
  --encoder_type "$ENCODER_TYPE" \
  --eva_model_path "$EVA_MODEL_PATH" \
  --local_retriever_module_dir "$LOCAL_RETRIEVER_DIR" \
  --uniir_checkpoint_path "$UNIIR_CKPT" \
  --llava_model "$LLAVA_MODEL" \
  --top_k 5 \
  --output_jsonl "$OUTPUT_JSONL"
