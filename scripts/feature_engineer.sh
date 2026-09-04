#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

echo "================================="
echo "starting feature engineering"
echo "================================="
python -m src.features.run_feature_engineering

echo "================================="
echo "feature engineering successful"
echo "================================="