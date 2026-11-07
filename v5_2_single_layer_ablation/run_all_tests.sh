#!/usr/bin/env bash
# Runs every verification test. No model download, no GPU required.
set -e
echo "=== Unit + logic tests ==="
python3 eval_awareness/tests/test_pipeline.py
echo
echo "=== End-to-end orchestration test ==="
python3 eval_awareness/tests/test_orchestration.py
echo
echo "ALL TESTS PASSED"
