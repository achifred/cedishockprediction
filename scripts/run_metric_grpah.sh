#!/usr/bin/env bash

source "$(dirname "0")/common.sh"

echo "========================="
echo "Generating metric graphs"
echo "========================="

python -m src.data.metric_graphs


echo "========================="
echo "Generating metric graphs done"
echo "========================="
