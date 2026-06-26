#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot deploy / update for 10-K RAG Studio on an Ubuntu/Debian server.
# Idempotent: safe to re-run to pull the latest code and restart.
#
# Usage (on the server, as a sudo-capable user):
#   REPO=https://github.com/Epiphany-Leon/rag-10k-chatbot.git \
#   DOMAIN=your.server.ip  bash deploy.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="${REPO:-https://github.com/Epiphany-Leon/rag-10k-chatbot.git}"
APP_DIR="${APP_DIR:-$HOME/rag-10k-chatbot}"
DOMAIN="${DOMAIN:-_}"
RUN_USER="${RUN_USER:-$(whoami)}"

echo "==> Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git nginx

echo "==> Fetching code into $APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> Python venv + dependencies"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip -q
./.venv/bin/pip install -r requirements.txt -q

if [ ! -f .env ]; then
  cp .env.example .env
  echo "!! Created .env from template — edit it and add your API keys, then re-run."
fi

echo "==> Pre-building the vector index"
set +e
./.venv/bin/python scripts/build_index.py || echo "(index build skipped/failed — check API keys in .env)"
set -e

echo "==> systemd service"
sudo cp deploy/ragstudio.service /etc/systemd/system/ragstudio.service
sudo sed -i "s|__USER__|$RUN_USER|g;  s|__APP_DIR__|$APP_DIR|g" /etc/systemd/system/ragstudio.service
sudo systemctl daemon-reload
sudo systemctl enable ragstudio
sudo systemctl restart ragstudio

echo "==> nginx reverse proxy"
sudo cp deploy/nginx.conf /etc/nginx/sites-available/ragstudio
sudo sed -i "s|__DOMAIN_OR_IP__|$DOMAIN|g" /etc/nginx/sites-available/ragstudio
sudo ln -sf /etc/nginx/sites-available/ragstudio /etc/nginx/sites-enabled/ragstudio
sudo nginx -t && sudo systemctl reload nginx

echo "==> Done. Visit  http://$DOMAIN/   (status: systemctl status ragstudio)"
