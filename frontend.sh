#!/usr/bin/env bash

set -e

cd frontend

npm ci
npm run build
npm run dev
