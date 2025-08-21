#!/usr/bin/env bash
set -euo pipefail

mkdir -p /models/mbti-distilbert-model
cd /models/mbti-distilbert-model

if [ ! -f model.safetensors ]; then
  echo "Downloading model weights..."
  # Replace URLs with your Hugging Face (or other) links
  # Add: -H "Authorization: Bearer $HF_TOKEN" for private models
  curl -L -o model.safetensors \
    "https://huggingface.co/wbmartin00/mbti-distilbert/resolve/main/model.safetensors"
fi

# Ensure tokenizer/config files exist (these live in repo too, but harmless to (re)download)
[ -f config.json ] || curl -L -o config.json \
  "https://huggingface.co/wbmartin00/mbti-distilbert/resolve/main/config.json"
[ -f tokenizer_config.json ] || curl -L -o tokenizer_config.json \
  "https://huggingface.co/wbmartin00/mbti-distilbert/resolve/main/tokenizer_config.json"
[ -f special_tokens_map.json ] || curl -L -o special_tokens_map.json \
  "https://huggingface.co/wbmartin00/mbti-distilbert/resolve/main/special_tokens_map.json"
[ -f vocab.txt ] || curl -L -o vocab.txt \
  "https://huggingface.co/wbmartin00/mbti-distilbert/resolve/main/vocab.txt"
