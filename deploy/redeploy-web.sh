#!/usr/bin/env bash
# Rebuild the Next.js static frontend locally and push it to the server.
# The server has no Node, so we build the static export here and rsync it; the
# FastAPI process (ragstudio-api) serves web/out/ + /api on port 8600.
#
# Usage (from the repo root):  bash deploy/redeploy-web.sh
set -euo pipefail

HOST="${HOST:-hermes}"
APP_DIR="${APP_DIR:-/srv/rag-10k-chatbot}"

echo "==> Building static export (web/out)"
( cd web && BUILD_EXPORT=1 npm run build )

echo "==> Syncing web/out -> $HOST:$APP_DIR/web/out"
rsync -az --delete web/out/ "$HOST:$APP_DIR/web/out/"

echo "==> Syncing backend code (server/ + ragstudio/)"
ssh "$HOST" "cd $APP_DIR && git fetch origin main --depth 1 -q && git reset --hard origin/main -q"

echo "==> Restarting the API service"
ssh "$HOST" "sudo systemctl restart ragstudio-api && sleep 5 && \
  curl -s -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:8600/api/health"

echo "==> Done — https://rag.gaolihong.com"
