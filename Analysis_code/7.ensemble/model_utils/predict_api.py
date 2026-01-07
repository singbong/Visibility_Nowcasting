"""예측 API 함수."""

from typing import List, Tuple

import numpy as np

from .dl_utils import _predict_dl_model, _prepare_test_data_dl, _prepare_val_data_dl
from .model_utils import DL_MODELS, TREE_MODELS, load_model, load_scaler
from .tree_utils import _predict_tree_model, _prepare_test_data_tree, _prepare_val_data_tree


def predict_test_proba(
    model_name: str,
    region: str,
    data_sample: str,
    device: str = 'cpu',
    n_folds: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Test 데이터에 대한 예측 확률과 target 값을 반환하는 함수.
    
    Args:
        model_name: 모델 이름 ('xgb', 'lgb', 'ft_transformer', 'resnet_like', 'deepgbm')
        region: 지역명 (예: 'seoul', 'busan')
        data_sample: 데이터 샘플 타입 (예: 'pure', 'smote', 'ctgan10000')
        device: 연산 장치 ('cpu' 또는 'cuda', 기본값: 'cpu')
        n_folds: Fold 개수 (기본값: 3)
    
    Returns:
        tuple: (probs, y_test)
            - probs: shape (n_folds, n_samples, n_classes) - 각 fold별 예측 확률
            - y_test: shape (n_samples,) - test 데이터의 target 값 (multi_class)
        
    Raises:
        ValueError: 모델 이름이 유효하지 않거나 fold 개수가 일치하지 않는 경우
        FileNotFoundError: 모델 또는 scaler 파일을 찾을 수 없는 경우
    """
    # 모델 타입 확인
    if model_name not in TREE_MODELS + DL_MODELS:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Must be one of {TREE_MODELS + DL_MODELS}"
        )
    
    # 모델 로드
    models = load_model(model_name, region, data_sample)
    
    if len(models) != n_folds:
        raise ValueError(
            f"로드된 모델 개수({len(models)})가 "
            f"n_folds({n_folds})와 일치하지 않습니다."
        )
    
    # Tree-based 모델 처리
    if model_name in TREE_MODELS:
        X_test, y_test = _prepare_test_data_tree(region)
        all_probs = []
        
        for fold_idx in range(n_folds):
            prob = _predict_tree_model(models[fold_idx], X_test)
            all_probs.append(prob)
        
        return np.array(all_probs), y_test
    
    # Deep learning 모델 처리
    else:
        # Scaler 로드
        scalers = load_scaler(model_name, region, data_sample)
        
        if len(scalers) != n_folds:
            raise ValueError(
                f"로드된 scaler 개수({len(scalers)})가 "
                f"n_folds({n_folds})와 일치하지 않습니다."
            )
        
        all_probs = []
        y_test = None
        
        for fold_idx in range(n_folds):
            # Test 데이터 준비
            X_test_num, X_test_cat, _, _, y_test = _prepare_test_data_dl(
                region, data_sample, fold_idx, scalers[fold_idx], model_name
            )
            
            # 예측
            prob = _predict_dl_model(
                models[fold_idx], X_test_num, X_test_cat, model_name, device
            )
            all_probs.append(prob)
        
        return np.array(all_probs), y_test


def predict_val_proba(
    model_name: str,
    region: str,
    data_sample: str,
    device: str = 'cpu',
    n_folds: int = 3
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Validation 데이터에 대한 예측 확률과 target 값을 반환하는 함수.
    
    Args:
        model_name: 모델 이름 ('xgb', 'lgb', 'ft_transformer', 'resnet_like', 'deepgbm')
        region: 지역명 (예: 'seoul', 'busan')
        data_sample: 데이터 샘플 타입 (예: 'pure', 'smote', 'ctgan10000')
        device: 연산 장치 ('cpu' 또는 'cuda', 기본값: 'cpu')
        n_folds: Fold 개수 (기본값: 3)
    
    Returns:
        tuple: (all_probs, y_pred, y_true)
            - all_probs: list of arrays - 각 fold별 예측 확률 리스트 [prob_fold0, prob_fold1, prob_fold2]
              각 배열의 shape은 (n_samples_fold, n_classes)이며, fold별로 샘플 수가 다를 수 있음
            - y_pred: list of arrays - 각 fold별 예측 클래스 리스트 [y_pred_fold0, y_pred_fold1, y_pred_fold2]
              각 배열의 shape은 (n_samples_fold,)
            - y_true: list of arrays - 각 fold별 실제값 리스트 [y_true_fold0, y_true_fold1, y_true_fold2]
              각 배열의 shape은 (n_samples_fold,)
        
    Raises:
        ValueError: 모델 이름이 유효하지 않거나 fold 개수가 일치하지 않는 경우
        FileNotFoundError: 모델 또는 scaler 파일을 찾을 수 없는 경우
    """
    # 모델 타입 확인
    if model_name not in TREE_MODELS + DL_MODELS:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Must be one of {TREE_MODELS + DL_MODELS}"
        )
    
    # 모델 로드
    models = load_model(model_name, region, data_sample)
    
    if len(models) != n_folds:
        raise ValueError(
            f"로드된 모델 개수({len(models)})가 "
            f"n_folds({n_folds})와 일치하지 않습니다."
        )
    
    # Tree-based 모델 처리
    if model_name in TREE_MODELS:
        all_probs = []
        y_pred_list = []
        y_true_list = []
        
        for fold_idx in range(n_folds):
            X_val, y_true = _prepare_val_data_tree(region, fold_idx)
            prob = _predict_tree_model(models[fold_idx], X_val)
            y_pred = np.argmax(prob, axis=1)
            
            all_probs.append(prob)
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
        
        # 각 fold별 validation 데이터 샘플 수가 다르므로 리스트로 반환
        return all_probs, y_pred_list, y_true_list
    
    # Deep learning 모델 처리
    else:
        # Scaler 로드
        scalers = load_scaler(model_name, region, data_sample)
        
        if len(scalers) != n_folds:
            raise ValueError(
                f"로드된 scaler 개수({len(scalers)})가 "
                f"n_folds({n_folds})와 일치하지 않습니다."
            )
        
        all_probs = []
        y_pred_list = []
        y_true_list = []
        
        for fold_idx in range(n_folds):
            # Validation 데이터 준비
            X_val_num, X_val_cat, _, _, y_true = _prepare_val_data_dl(
                region, data_sample, fold_idx, scalers[fold_idx], model_name
            )
            
            # 예측
            prob = _predict_dl_model(
                models[fold_idx], X_val_num, X_val_cat, model_name, device
            )
            y_pred = np.argmax(prob, axis=1)
            
            all_probs.append(prob)
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
        
        # 각 fold별 validation 데이터 샘플 수가 다르므로 리스트로 반환
        return all_probs, y_pred_list, y_true_list
