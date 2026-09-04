#!/usr/bin/env bash

set -Eeuo pipefail

source "$(dirname "0")/common.sh"

echo "========================="
echo "Evaluating LSTM model"
echo "========================="

python -m src.models.lstm_model.run_evaluate_lstm


echo "========================="
echo "Evaluating LSTM model done"
echo "========================="
