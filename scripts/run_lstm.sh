#!/usr/bin/env bash

set -Eeuo pipefail

echo "================================="
echo "Running LSTM pipeline"
echo "================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/run_train_lstm.sh"
"$SCRIPT_DIR/run_evaluate_lstm.sh"

echo "==================================="
echo "Running LSTM pipeline successful"
echo "==================================="