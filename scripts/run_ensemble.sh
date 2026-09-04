#!/usr/bin/env bash

source "$(dirname "0")/common.sh"

echo "========================="
echo "Running Ensemble model"
echo "========================="

python -m src.models.ensemble.run_ensemble


echo "========================="
echo "Running Ensemble model done"
echo "========================="
