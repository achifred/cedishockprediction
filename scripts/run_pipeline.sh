#!/usr/bin/env bash

set -Eeuo pipefail

echo "================================="
echo "starting pipeline"
echo "================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#"$SCRIPT_DIR/ingest_data.sh"
"$SCRIPT_DIR/preprocessing_data.sh"
"$SCRIPT_DIR/feature_engineer.sh"
"$SCRIPT_DIR/run_models.sh"
"$SCRIPT_DIR/run_data_summary.sh"
"$SCRIPT_DIR/run_metric_graph.sh"
"$SCRIPT_DIR/run_feat_impt.sh"

echo "==================================="
echo "pipeline successful"
echo "==================================="