#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"

echo "================================="
echo "starting ingesting data"
echo "================================="
python -m src.main

echo "================================="
echo "ingesting successful"
echo "================================="