#!/usr/bin/env bash

set -Eeuo pipefail

source "$(dirname "0")/common.sh"

echo "========================="
echo "Evaluating xgboost model"
echo "========================="

python -m src.models.xgboost_model.run_evaluate_xgboost


echo "========================="
echo "Evaluating xgboost model done"
echo "========================="
