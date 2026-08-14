#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "Docker and Docker Compose are already installed."
  exit 0
fi

if ! sudo -v; then
  echo "Unable to obtain sudo permission; Docker was not installed."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

echo "Docker installed. Log out and back in if docker still needs sudo."
