import optuna
import numpy as np
import random
import pandas as pd
import joblib
import os
import torch
from utils import *
# Python 및 Numpy 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)


# 1. Study 생성 시 'maximize'로 설정
study = optuna.create_study(
    direction="maximize",  # CSI 점수가 높을수록 좋으므로 maximize
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10) # 초반 10에폭은 지켜보고 이후 가지치기
)
# Trial 완료 시 상세 정보 출력하는 callback 함수
def print_trial_callback(study, trial):
    """각 trial 완료 시 best value를 포함한 상세 정보 출력"""
    print(f"\n{'='*80}")
    print(f"Trial {trial.number} 완료")
    print(f"  Value (CSI): {trial.value:.6f}" if trial.value is not None else f"  Value: {trial.value}")
    print(f"  Parameters: {trial.params}")
    print(f"  Best Value (CSI): {study.best_value:.6f}" if study.best_value is not None else f"  Best Value: {study.best_value}")
    print(f"  Best Trial: {study.best_trial.number}")
    print(f"  Best Parameters: {study.best_params}")
    print(f"{'='*80}\n")



# 2. 최적화 실행
study.optimize(
    lambda trial: objective(trial, model_choose="resnet_like", region="busan"), 
    n_trials=100
,
    callbacks=[print_trial_callback]
)

# 3. 결과 확인 및 요약
print(f"\n최적화 완료.")
print(f"Best CSI Score: {study.best_value:.4f}")
print(f"Best Hyperparameters: {study.best_params}")

try:
    # 모든 trial의 CSI 점수 추출
    csi_scores = [trial.value for trial in study.trials if trial.value is not None]
    
    if len(csi_scores) > 0:
        print(f"\n최적화 과정 요약:")
        print(f"  - 총 시도 횟수: {len(study.trials)}")
        print(f"  - 성공한 시도: {len(csi_scores)}")
        print(f"  - 최초 CSI: {csi_scores[0]:.4f}")
        print(f"  - 최종 CSI: {csi_scores[-1]:.4f}")
        print(f"  - 최고 CSI: {max(csi_scores):.4f}")
        print(f"  - 최저 CSI: {min(csi_scores):.4f}")
        print(f"  - 평균 CSI: {np.mean(csi_scores):.4f}")
    
    # Study 객체 저장
    # 파일 위치 기반으로 base 디렉토리 경로 설정
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_file_dir))  # 5.optima 디렉토리
    os.makedirs(os.path.join(base_dir, "optimization_history"), exist_ok=True)
    study_path = os.path.join(base_dir, "optimization_history/resnet_like_pure_busan_trials.pkl")
    joblib.dump(study, study_path)
    print(f"\n최적화 Study 객체가 {study_path}에 저장되었습니다.")
    
    # 최적화된 하이퍼파라미터로 최종 모델 학습 및 저장
    print("\n" + "="*50)
    print("최적화된 하이퍼파라미터로 최종 모델 학습 시작")
    print("="*50)
    
    best_params = study.best_params
    model_paths = train_final_model(
        best_params=best_params,
        model_choose="resnet_like",
        region="busan",
        data_sample='pure',
        target='multi',
        n_folds=3,
        random_state=seed
    )
    
    print(f"\n최종 모델 학습 및 저장 완료!")
    print(f"저장된 모델 경로:")
    for path in model_paths:
        print(f"  - {path}")
        
except Exception as e:
    print(f"\n⚠️  최적화 결과 분석 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()

# 정상 종료
import sys
sys.exit(0)

