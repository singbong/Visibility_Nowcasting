"""모델 로딩 및 메트릭 함수."""

import os
import sys
from typing import Any, List, Tuple

import numpy as np
from joblib import load
from sklearn.metrics import confusion_matrix

# ==========================================
# 상수 정의
# ==========================================
TEST_YEAR = 2021
DROP_COLUMNS = ['multi_class', 'year']
FEATURE_COLUMNS = [
    'temp_C', 'precip_mm', 'wind_speed', 'wind_dir', 'hm',
    'vap_pressure', 'dewpoint_C', 'loc_pressure', 'sea_pressure',
    'solarRad', 'snow_cm', 'cloudcover', 'lm_cloudcover', 'low_cloudbase',
    'groundtemp', 'O3', 'NO2', 'PM10', 'PM25', 'year',
    'month', 'hour', 'ground_temp - temp_C', 'hour_sin', 'hour_cos',
    'month_sin', 'month_cos', 'multi_class'
]

# 모델 타입 상수
TREE_MODELS = ['xgb', 'lgb']
DL_MODELS = ['ft_transformer', 'resnet_like', 'deepgbm']


# ==========================================
# 경로 설정
# ==========================================
def _setup_model_path() -> str:
    """모델 클래스 import를 위한 경로를 설정하고 반환합니다.
    
    Returns:
        models 디렉토리의 절대 경로
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.abspath(os.path.join(current_file_dir, '../../models'))
    except NameError:
        # 노트북 환경에서 __file__이 없는 경우
        cwd = os.getcwd()
        if 'Analysis_code' in cwd:
            analysis_code_dir = cwd[:cwd.index('Analysis_code') + len('Analysis_code')]
            models_path = os.path.join(analysis_code_dir, 'models')
        else:
            models_path = '/workspace/visibility_prediction/Analysis_code/models'
    
    if models_path not in sys.path:
        sys.path.insert(0, models_path)
    
    return models_path


# ==========================================
# 메트릭 함수 (pickle 호환성)
# ==========================================
def calculate_csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """CSI(Critical Success Index) 점수를 계산합니다.
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        
    Returns:
        CSI 점수 (0~1 사이의 값)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 혼동 행렬에서 H(Hit), F(False alarm), M(Miss) 추출
    H = cm[0, 0] + cm[1, 1]
    F = cm[1, 0] + cm[2, 0] + cm[0, 1] + cm[2, 1]
    M = cm[0, 2] + cm[1, 2]
    
    # CSI 계산
    csi = H / (H + F + M + 1e-10)
    return csi


def eval_metric_csi(y_true: np.ndarray, pred_prob: np.ndarray) -> float:
    """XGBoost용 CSI 메트릭 함수.
    
    Args:
        y_true: 실제 레이블
        pred_prob: 예측 확률
        
    Returns:
        CSI 점수의 음수값
    """
    pred = np.argmax(pred_prob, axis=1)
    csi = calculate_csi(y_true, pred)
    return -1 * csi


def csi_metric(y_true: np.ndarray, pred_prob: np.ndarray) -> Tuple[str, float, bool]:
    """LightGBM용 CSI 메트릭 함수.
    
    Args:
        y_true: 실제 레이블
        pred_prob: 예측 확률 (shape: [n_samples, n_classes])
        
    Returns:
        ('CSI', score, higher_better) 튜플
    """
    y_pred_binary = np.argmax(pred_prob, axis=1)
    score = calculate_csi(y_true, y_pred_binary)
    return 'CSI', score, True


# ==========================================
# Pickle 호환성 함수 등록
# ==========================================
def _register_pickle_functions(model_name: str) -> None:
    """모델 로드를 위해 __main__ 모듈에 필요한 함수를 등록합니다.
    
    모델이 저장될 때 __main__ 모듈에서 정의된 함수를 참조했을 수 있으므로,
    pickle이 모델을 로드할 때 함수를 찾을 수 있도록 등록합니다.
    
    Args:
        model_name: 모델 이름 ('xgb' 또는 'lgb')
    """
    if '__main__' not in sys.modules:
        return
    
    main_module = sys.modules['__main__']
    main_module.calculate_csi = calculate_csi
    
    if model_name == 'xgb':
        main_module.eval_metric_csi = eval_metric_csi
    elif model_name == 'lgb':
        main_module.csi_metric = csi_metric


# ==========================================
# 모델 로드 함수
# ==========================================
def load_model(model_name: str, region: str, data_sample: str) -> List[Any]:
    """모델을 로드하는 함수.
    
    Args:
        model_name: 모델 이름 ('xgb', 'lgb', 'ft_transformer', 'resnet_like', 'deepgbm')
        region: 지역명 (예: 'seoul', 'busan')
        data_sample: 데이터 샘플 타입 (예: 'pure', 'smote', 'ctgan10000')
        
    Returns:
        로드된 모델 리스트 (각 fold별 모델)
        
    Raises:
        FileNotFoundError: 모델 파일을 찾을 수 없는 경우
        ValueError: 모델 이름이 유효하지 않은 경우
    """
    if model_name not in TREE_MODELS + DL_MODELS:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Must be one of {TREE_MODELS + DL_MODELS}"
        )
    
    # Tree-based 모델의 경우 pickle 호환성 함수 등록
    if model_name in TREE_MODELS:
        _register_pickle_functions(model_name)
    
    # Deep learning 모델의 경우 모델 클래스 import
    if model_name in DL_MODELS:
        _setup_model_path()
    
    model_path = f"../save_model/{model_name}_optima/{model_name}_{data_sample}_{region}.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {model_path}"
        )
    
    try:
        models = load(model_path)
    except Exception as e:
        raise FileNotFoundError(f"모델 로드 실패 ({model_path}): {e}")
    
    return models


def load_scaler(model_name: str, region: str, data_sample: str) -> List[Any]:
    """각 fold의 scaler를 로드하는 함수.
    
    Args:
        model_name: 모델 이름
        region: 지역명
        data_sample: 데이터 샘플 타입
        
    Returns:
        로드된 scaler 리스트 (각 fold별 scaler)
        
    Raises:
        FileNotFoundError: Scaler 파일을 찾을 수 없는 경우
    """
    if data_sample == 'pure':
        scaler_filename = f'{model_name}_pure_{region}_scaler.pkl'
    else:
        scaler_filename = f'{model_name}_{data_sample}_{region}_scaler.pkl'
    
    scaler_path = f'../save_model/{model_name}_optima/scaler/{scaler_filename}'
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler 파일을 찾을 수 없습니다: {scaler_path}"
        )
    
    try:
        scalers = load(scaler_path)
    except Exception as e:
        raise FileNotFoundError(f"Scaler 로드 실패 ({scaler_path}): {e}")
    
    return scalers
