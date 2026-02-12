#!/bin/bash
# Run all MiniCPM-o tests from PR #1454
# Usage: ./run_minicpmo_tests.sh

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  MiniCPM-o PR #1454 Test Suite"
echo "  Model: h0witended/tiny-random-MiniCPM-o-2_6"
echo "============================================"
echo ""

# Seq2seq tests (3 tests)
echo "[1/3] Seq2seq tests..."
python -m pytest tests/openvino/test_seq2seq.py -k "minicpmo" -v --tb=short 2>&1 | tail -5
echo ""

# Quantization: compressed weights 
echo "[2/3] Quantization: compressed weights test..."
python -m pytest tests/openvino/test_quantization.py -k "test_ovmodel_load_with_compressed_weights_17" -v --tb=short 2>&1 | tail -5
echo ""

# Quantization: uncompressed weights 
echo "[3/3] Quantization: uncompressed weights test..."
python -m pytest tests/openvino/test_quantization.py -k "test_ovmodel_load_with_uncompressed_weights_17" -v --tb=short 2>&1 | tail -5
echo ""

echo "  All MiniCPM-o tests complete"

