#!/usr/bin/env bash

set -e

# Move to the directory containing this script
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Start the FastAPI server
exec uvicorn app.main:app --host 0.0.0.0 --port 9090 --reload
