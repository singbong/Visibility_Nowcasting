"""Tree 기반 모델용 헬퍼 함수."""

from typing import Any, Tuple

import numpy as np
import pandas as pd

from .model_utils import DROP_COLUMNS, TEST_YEAR
from .preprocessing import preprocessing


def _prepare_test_data_tree(region: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Tree-based 모델용 test 데이터를 준비합니다.
    
    Args:
        region: 지역명
        
    Returns:
        tuple: (X_test, y_test)
            - X_test: 전처리된 test 데이터 (설명변수만)
            - y_test: test 데이터의 target 값 (multi_class)
    """
    test_data = pd.read_csv(f"../../data/data_for_modeling/{region}_test.csv", index_col=0)
    test_data = preprocessing(test_data)
    test_data = test_data.loc[test_data['year'].isin([TEST_YEAR]), :]
    X_test = test_data.drop(columns=DROP_COLUMNS)
    y_test = test_data['multi_class'].values
    return X_test, y_test


def _prepare_val_data_tree(region: str, fold_idx: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Tree-based 모델용 validation 데이터를 준비합니다.
    
    Args:
        region: 지역명
        fold_idx: Fold 인덱스 (0, 1, 2) - fold 0: 2020, fold 1: 2019, fold 2: 2018
        
    Returns:
        tuple: (X_val, y_val)
            - X_val: 전처리된 validation 데이터 (설명변수만)
            - y_val: validation 데이터의 target 값 (multi_class)
    """
    train_data = pd.read_csv(f"../../data/data_for_modeling/{region}_train.csv", index_col=0)
    train_data = preprocessing(train_data)
    val_year = TEST_YEAR - (fold_idx + 1)  # fold 0: 2020, fold 1: 2019, fold 2: 2018
    val_data = train_data.loc[train_data['year'].isin([val_year]), :]
    X_val = val_data.drop(columns=DROP_COLUMNS)
    y_val = val_data['multi_class'].values
    return X_val, y_val


def _predict_tree_model(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    """Tree-based 모델로 예측을 수행합니다.
    
    Args:
        model: 학습된 모델 (XGBoost 또는 LightGBM)
        X_test: Test 데이터
        
    Returns:
        예측 확률 (n_samples, n_classes)
    """
    return model.predict_proba(X_test)
