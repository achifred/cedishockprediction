#!/usr/bin/env bash

set -Eeuo pipefail

echo "================================="
echo "starting pipeline"
echo "================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#"$SCRIPT_DIR/ingest_data.sh"
"$SCRIPT_DIR/preprocessing_data.sh"
"$SCRIPT_DIR/feature_engineer.sh"

echo "==================================="
echo "pipeline successful"
echo "==================================="