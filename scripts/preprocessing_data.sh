#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

echo "================================="
echo "starting preprocessing data"
echo "================================="
python -m src.data.preprocessing_data

echo "================================="
echo "preprocessing successful"
echo "================================="