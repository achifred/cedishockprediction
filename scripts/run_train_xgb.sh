#!/usr/bin/env bash

set -Eeuo pipefail
source "$(dirname "0")/common.sh"

echo "========================="
echo "Training xgboost model"
echo "========================="

python -m src.models.xgboost_model.run_train_xgboost


echo "========================="
echo "Training xgboost model done"
echo "========================="
