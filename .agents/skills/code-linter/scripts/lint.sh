#!/bin/bash
# Reusable capability script for linting core Python source files using Ruff

# Check if --fix argument is passed
FIX_ARG=""
if [ "$1" == "--fix" ]; then
  FIX_ARG="--fix"
fi

echo "Running project code linter..."
uv run --with ruff ruff check api/ retrieval_core.py fashion_ds_processing.py config.py setup_check.py tests/ $FIX_ARG
