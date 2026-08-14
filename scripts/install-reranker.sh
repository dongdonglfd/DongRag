#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cache_root="${XDG_CACHE_HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)/.cache}"
rerank_dir="${MINIRAG_RERANK_DIR:-$cache_root/minirag-reranker}"
llama_version="b10293"
model_name="bge-reranker-v2-m3-Q4_K_M.gguf"
binary="$rerank_dir/bin/llama-$llama_version/llama-server"
model="$rerank_dir/models/$model_name"

mkdir -p "$rerank_dir/bin" "$rerank_dir/models"

if [[ ! -x "$binary" ]]; then
  temp_dir=$(mktemp -d /tmp/minirag-reranker-install.XXXXXX)
  trap 'rm -rf "$temp_dir"' EXIT
  curl -fL --retry 3 -o "$temp_dir/llama.tar.gz" "https://github.com/ggml-org/llama.cpp/releases/download/$llama_version/llama-$llama_version-bin-ubuntu-x64.tar.gz"
  mkdir -p "$temp_dir/llama"
  tar -xzf "$temp_dir/llama.tar.gz" -C "$temp_dir/llama"
  cp -a "$temp_dir/llama/." "$rerank_dir/bin/"
fi

if [[ ! -f "$model" ]]; then
  model_base_url="${RERANK_MODEL_BASE_URL:-https://hf-mirror.com/gpustack/bge-reranker-v2-m3-GGUF/resolve/main}"
  curl --http1.1 -fL --retry 5 --retry-all-errors -C - -o "$model.part" "$model_base_url/$model_name"
  mv "$model.part" "$model"
fi

echo "reranker installed: $rerank_dir"
