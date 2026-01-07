#!/bin/bash

# GPU 1번에서 SMOTENC+CTGAN 샘플 생성 스크립트들을 순차적으로 실행
# 실행 디렉토리: /workspace/visibility_prediction/Analysis_code/make_oversample_data

export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Starting SMOTENC+CTGAN sample generation on GPU 1"
echo "=========================================="
echo ""

# 7000 샘플 생성
echo "=== Processing 7000 samples ==="
echo "Running smotenc_ctgan/smotenc_ctgan_sample_7000_1.py..."
python smotenc_ctgan/smotenc_ctgan_sample_7000_1.py
echo ""

echo "Running smotenc_ctgan/smotenc_ctgan_sample_7000_2.py..."
python smotenc_ctgan/smotenc_ctgan_sample_7000_2.py
echo ""

echo "Running smotenc_ctgan/smotenc_ctgan_sample_7000_3.py..."
python smotenc_ctgan/smotenc_ctgan_sample_7000_3.py
echo ""

# 10000 샘플 생성
echo "=== Processing 10000 samples ==="
echo "Running smotenc_ctgan/smotenc_ctgan_sample_10000_1.py..."
python smotenc_ctgan/smotenc_ctgan_sample_10000_1.py
echo ""

echo "Running smotenc_ctgan/smotenc_ctgan_sample_10000_2.py..."
python smotenc_ctgan/smotenc_ctgan_sample_10000_2.py
echo ""

echo "Running smotenc_ctgan/smotenc_ctgan_sample_10000_3.py..."
python smotenc_ctgan/smotenc_ctgan_sample_10000_3.py
echo ""

# 20000 샘플 생성
echo "=== Processing 20000 samples ==="
echo "Running smotenc_ctgan/smotenc_ctgan_sample_20000_1.py..."
python smotenc_ctgan/smotenc_ctgan_sample_20000_1.py
echo ""

echo "Running smotenc_ctgan/smotenc_ctgan_sample_20000_2.py..."
python smotenc_ctgan/smotenc_ctgan_sample_20000_2.py
echo ""

echo "Running smotenc_ctgan/smotenc_ctgan_sample_20000_3.py..."
python smotenc_ctgan/smotenc_ctgan_sample_20000_3.py
echo ""

echo "=========================================="
echo "All SMOTENC+CTGAN sample generation completed!"
echo "=========================================="

