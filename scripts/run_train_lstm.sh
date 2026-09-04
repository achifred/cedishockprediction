#!/usr/bin/env bash

set -Eeuo pipefail
source "$(dirname "0")/common.sh"

echo "========================="
echo "Training LSTM model"
echo "========================="

python -m src.models.lstm_model.run_train_lstm


echo "========================="
echo "Training LSTM model done"
echo "========================="
