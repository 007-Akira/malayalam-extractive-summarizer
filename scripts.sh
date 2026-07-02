#!/usr/bin/env bash

#SCRIPT TO RUN THE WEHBSITE
#./scripts.sh dev

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_DIR="$ROOT_DIR/backend"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

usage() {
  cat <<'EOF'
Usage: ./scripts.sh <command>

Commands:
  install    Install frontend and backend dependencies
  frontend   Start the React/Vite frontend on http://127.0.0.1:5173
  backend    Start the FastAPI backend on http://127.0.0.1:8000
  dev        Start backend and frontend together
  build      Build the frontend
EOF
}

install_deps() {
  "$PYTHON_BIN" -m pip install -r "$BACKEND_DIR/requirements.txt"
  npm install --prefix "$FRONTEND_DIR"
}

start_frontend() {
  npm run dev --prefix "$FRONTEND_DIR" -- --host 127.0.0.1
}

start_backend() {
  cd "$BACKEND_DIR"
  "$PYTHON_BIN" -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
}

start_dev() {
  cd "$ROOT_DIR"
  "$0" backend &
  backend_pid=$!

  "$0" frontend &
  frontend_pid=$!

  trap 'kill "$backend_pid" "$frontend_pid" 2>/dev/null || true' INT TERM EXIT
  wait
}

build_frontend() {
  npm run build --prefix "$FRONTEND_DIR"
}

case "${1:-}" in
  install)
    install_deps
    ;;
  frontend)
    start_frontend
    ;;
  backend)
    start_backend
    ;;
  dev)
    start_dev
    ;;
  build)
    build_frontend
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 1
    ;;
esac


