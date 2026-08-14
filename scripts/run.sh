#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "已创建 .env，请填写 BAILIAN_API_KEY 后重新运行。"
  exit 1
fi

set -a
source .env
set +a

if [[ -z "${BAILIAN_API_KEY:-}" ]]; then
  echo "请先在 .env 中填写 BAILIAN_API_KEY。"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "未检测到 Docker，请先运行 bash scripts/install-docker.sh。"
  exit 1
fi

docker compose up -d postgres
if [[ "${RERANK_ENABLED:-true}" == "true" ]]; then
  bash scripts/start-reranker.sh
fi
exec go run ./cmd/minirag
