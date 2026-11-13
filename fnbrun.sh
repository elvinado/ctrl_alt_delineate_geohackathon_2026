#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDS=()

cleanup() {
    echo ""
    echo "Stopping frontend and backend..."

    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    wait 2>/dev/null || true
    echo "All services stopped."
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"

echo "Starting backend..."
bash backend.sh &
PIDS+=($!)

echo "Starting frontend..."
bash frontend.sh &
PIDS+=($!)

echo ""
echo "Frontend and backend are running."
echo "Press Ctrl+C to stop both."

wait

