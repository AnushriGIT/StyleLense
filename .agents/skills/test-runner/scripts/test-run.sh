#!/bin/bash
# Reusable capability script for running test suites
echo "Running project test runner..."
if [ -f "./.venv/bin/pytest" ]; then
  ./.venv/bin/pytest tests/
elif [ -f "./venv/bin/pytest" ]; then
  ./venv/bin/pytest tests/
else
  pytest tests/
fi
