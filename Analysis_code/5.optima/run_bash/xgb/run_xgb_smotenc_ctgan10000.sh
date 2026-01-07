#!/bin/bash

# 스크립트 디렉토리로 이동 (상위 디렉토리인 5.optima로 이동)
cd "$(dirname "$0")/../.."

# 시작 시간 기록
START_TIME=$(date +%s)
echo "=========================================="
echo "XGB SMOTENC CTGAN20000 파일 병렬 실행 시작"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: 0번 (CUDA_VISIBLE_DEVICES=0)"
echo "모든 파일을 병렬로 실행합니다."
echo "=========================================="
echo ""

# 실행할 파일 목록
FILES=(
    "XGB_smotenc_ctgan10000_busan.py"
    "XGB_smotenc_ctgan10000_daegu.py"
    "XGB_smotenc_ctgan10000_daejeon.py"
    "XGB_smotenc_ctgan10000_gwangju.py"
    "XGB_smotenc_ctgan10000_incheon.py"
    "XGB_smotenc_ctgan10000_seoul.py"
)

# 로그 디렉토리 생성
LOG_DIR="run_bash/xgb/logs"
mkdir -p "$LOG_DIR"

# 각 파일을 병렬로 실행
declare -a PIDS=()
declare -a FILE_PATHS=()
SUCCESS_COUNT=0
FAIL_COUNT=0

for file in "${FILES[@]}"; do
    filepath="xgb_smotenc_ctgan10000/$file"
    if [ ! -f "$filepath" ]; then
        echo "⚠️  경고: $filepath 파일을 찾을 수 없습니다. 건너뜁니다."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # 로그 파일 경로
    logfile="${LOG_DIR}/$(basename "$filepath" .py).log"
    
    echo "----------------------------------------"
    echo "백그라운드 실행 시작: $filepath"
    echo "로그 파일: $logfile"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
    
    # Python 스크립트를 백그라운드로 실행 (GPU 0번 설정)
    CUDA_VISIBLE_DEVICES=0 python3 -u "$filepath" > "$logfile" 2>&1 &
    PID=$!
    PIDS+=($PID)
    FILE_PATHS+=("$filepath")
    
    echo "PID: $PID"
    echo ""
done

# 모든 프로세스가 완료될 때까지 대기
echo "=========================================="
echo "모든 작업이 백그라운드에서 실행 중입니다."
echo "총 ${#PIDS[@]}개의 프로세스가 실행 중입니다."
echo "완료될 때까지 대기 중..."
echo "=========================================="
echo ""

# 각 프로세스의 완료 상태 확인
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    FILEPATH=${FILE_PATHS[$i]}
    LOGFILE="${LOG_DIR}/$(basename "$FILEPATH" .py).log"
    
    # 프로세스가 완료될 때까지 대기
    if wait $PID; then
        echo "✓ 완료: $FILEPATH (PID: $PID)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        echo "✗ 실패: $FILEPATH (PID: $PID, Exit Code: $EXIT_CODE)"
        echo "   로그 확인: $LOGFILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# 종료 시간 기록
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "=========================================="
echo "XGB SMOTENC CTGAN20000 파일 병렬 실행 완료"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "총 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"
echo "성공: ${SUCCESS_COUNT}개"
echo "실패: ${FAIL_COUNT}개"
echo "로그 디렉토리: $LOG_DIR"
echo "=========================================="

# 실패한 작업이 있으면 종료 코드 1 반환
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi


