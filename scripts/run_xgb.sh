#!/usr/bin/env bash

set -Eeuo pipefail

echo "================================="
echo "Running XGBOOST pipeline"
echo "================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"
"$SCRIPT_DIR/run_train_xgb.sh"
"$SCRIPT_DIR/run_evaluate_xgb.sh"

echo "==================================="
echo "Running XGBOOST pipeline successful"
echo "==================================="