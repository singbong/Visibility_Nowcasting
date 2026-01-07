#!/bin/bash

# GPU 0번에서 CTGAN 샘플 생성 스크립트들을 순차적으로 실행
# 실행 디렉토리: /workspace/visibility_prediction/Analysis_code/make_oversample_data

export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Starting CTGAN sample generation on GPU 0"
echo "=========================================="
echo ""

# 7000 샘플 생성
echo "=== Processing 7000 samples ==="
echo "Running only_ctgan/ctgan_sample_7000_1.py..."
python only_ctgan/ctgan_sample_7000_1.py
echo ""

echo "Running only_ctgan/ctgan_sample_7000_2.py..."
python only_ctgan/ctgan_sample_7000_2.py
echo ""

echo "Running only_ctgan/ctgan_sample_7000_3.py..."
python only_ctgan/ctgan_sample_7000_3.py
echo ""

# 10000 샘플 생성
echo "=== Processing 10000 samples ==="
echo "Running only_ctgan/ctgan_sample_10000_1.py..."
python only_ctgan/ctgan_sample_10000_1.py
echo ""

echo "Running only_ctgan/ctgan_sample_10000_2.py..."
python only_ctgan/ctgan_sample_10000_2.py
echo ""

echo "Running only_ctgan/ctgan_sample_10000_3.py..."
python only_ctgan/ctgan_sample_10000_3.py
echo ""

# 20000 샘플 생성
echo "=== Processing 20000 samples ==="
echo "Running only_ctgan/ctgan_sample_20000_1.py..."
python only_ctgan/ctgan_sample_20000_1.py
echo ""

echo "Running only_ctgan/ctgan_sample_20000_2.py..."
python only_ctgan/ctgan_sample_20000_2.py
echo ""

echo "Running only_ctgan/ctgan_sample_20000_3.py..."
python only_ctgan/ctgan_sample_20000_3.py
echo ""

echo "=========================================="
echo "All CTGAN sample generation completed!"
echo "=========================================="

