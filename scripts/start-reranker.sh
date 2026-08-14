#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cache_root="${XDG_CACHE_HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)/.cache}"
rerank_dir="${MINIRAG_RERANK_DIR:-$cache_root/minirag-reranker}"
binary="$rerank_dir/bin/llama-b10293/llama-server"
model="$rerank_dir/models/bge-reranker-v2-m3-Q4_K_M.gguf"
pid_file="$rerank_dir/reranker.pid"
log_file="$rerank_dir/reranker.log"

if curl -fsS --max-time 2 http://127.0.0.1:8081/health >/dev/null 2>&1; then
  echo "reranker is ready"
  exit 0
fi

if [[ ! -x "$binary" || ! -f "$model" ]]; then
  bash scripts/install-reranker.sh
fi

if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
  kill "$(<"$pid_file")" || true
  for _ in $(seq 1 20); do
    kill -0 "$(<"$pid_file")" 2>/dev/null || break
    sleep 1
  done
fi

nohup "$binary" \
  --model "$model" \
  --host 127.0.0.1 \
  --port 8081 \
  --embedding \
  --pooling rank \
  --ctx-size 2048 \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --parallel 1 \
  --threads "${RERANK_THREADS:-4}" \
  >"$log_file" 2>&1 &
echo $! >"$pid_file"

for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 http://127.0.0.1:8081/health >/dev/null 2>&1; then
    echo "reranker is ready"
    exit 0
  fi
  if ! kill -0 "$(<"$pid_file")" 2>/dev/null; then
    tail -40 "$log_file"
    exit 1
  fi
  sleep 1
done

echo "reranker did not become ready; see $log_file"
exit 1
