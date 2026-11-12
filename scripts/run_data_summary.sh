#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

echo "================================="
echo "running summary data"
echo "================================="
python -m src.data.dataset_summary

echo "================================="
echo "summary data successful"
echo "================================="