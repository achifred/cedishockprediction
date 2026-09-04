#!/usr/bin/env bash

set -Eeuo pipefail

echo "================================="
echo "Running models pipeline"
echo "================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_xgb.sh"
"$SCRIPT_DIR/run_lstm.sh"
"$SCRIPT_DIR/run_ensemble.sh"

echo "==================================="
echo "Running models pipeline successful"
echo "==================================="