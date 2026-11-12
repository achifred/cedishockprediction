#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

echo "================================="
echo "running feature importance "
echo "================================="
python -m src.data.feature_importance

echo "================================="
echo "feature importance  successful"
echo "================================="