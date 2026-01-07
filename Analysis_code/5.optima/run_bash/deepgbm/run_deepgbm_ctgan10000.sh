#!/bin/bash

# 스크립트 디렉토리로 이동 (상위 디렉토리인 5.optima로 이동)
cd "$(dirname "$0")/../.."

# 시작 시간 기록
START_TIME=$(date +%s)
echo "=========================================="
echo "DeepGBM CTGAN10000 파일 실행 시작"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: 0번 (CUDA_VISIBLE_DEVICES=0)"
echo "=========================================="
echo ""

# 실행할 파일 목록
FILES=(
    "deepgbm_ctgan10000_busan.py"
    "deepgbm_ctgan10000_daegu.py"
    "deepgbm_ctgan10000_daejeon.py"
    "deepgbm_ctgan10000_gwangju.py"
    "deepgbm_ctgan10000_incheon.py"
    "deepgbm_ctgan10000_seoul.py"
)

# 에러 발생 시 중단 여부 (set -e를 사용하면 에러 발생 시 즉시 중단)
set -e

# 각 파일 실행
SUCCESS_COUNT=0
FAIL_COUNT=0

for file in "${FILES[@]}"; do
    filepath="deepgbm_ctgan10000/$file"
    if [ ! -f "$filepath" ]; then
        echo "⚠️  경고: $filepath 파일을 찾을 수 없습니다. 건너뜁니다."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    echo "----------------------------------------"
    echo "실행 중: $filepath"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
    
    FILE_START=$(date +%s)
    
    # Python 스크립트 실행 (GPU 0번 설정)
    if CUDA_VISIBLE_DEVICES=0 python3 -u "$filepath"; then
        FILE_END=$(date +%s)
        FILE_DURATION=$((FILE_END - FILE_START))
        echo ""
        echo "✓ 완료: $filepath (소요 시간: ${FILE_DURATION}초)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FILE_END=$(date +%s)
        FILE_DURATION=$((FILE_END - FILE_START))
        echo ""
        echo "✗ 실패: $filepath (소요 시간: ${FILE_DURATION}초)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "에러 발생으로 인해 스크립트를 중단합니다."
        exit 1
    fi
    echo ""
done

# 종료 시간 기록
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo "=========================================="
echo "DeepGBM CTGAN10000 파일 실행 완료"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "총 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"
echo "성공: ${SUCCESS_COUNT}개"
echo "실패: ${FAIL_COUNT}개"
echo "=========================================="

