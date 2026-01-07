"""SHAP 분석을 위한 전처리 유틸리티."""

import sys
import warnings
from contextlib import contextmanager
from io import StringIO
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch

from .config import SHAPConfig


class PreprocessingUtils:
    """전처리 유틸리티 클래스."""
    
    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 추가."""
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['ground_temp - temp_C'] = df['groundtemp'] - df['temp_C']
        return df
    
    @staticmethod
    def preprocessing(df: pd.DataFrame, config: SHAPConfig) -> pd.DataFrame:
        """데이터 전처리."""
        df = df[df.columns].copy()
        df['year'] = df['year'].astype('int')
        df['month'] = df['month'].astype('int')
        df['hour'] = df['hour'].astype('int')
        df = PreprocessingUtils.add_derived_features(df).copy()
        df['multi_class'] = df['multi_class'].astype('int')
        df.loc[df['wind_dir'] == config.CALM_WIND_DIR, 'wind_dir'] = "0"
        df['wind_dir'] = df['wind_dir'].astype('int')
        df = df[config.FEATURE_COLUMNS].copy()
        return df


class ShapValueExtractor:
    """SHAP 값 추출 및 변환 유틸리티 클래스."""
    
    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """다양한 타입을 numpy array로 변환."""
        if isinstance(value, torch.Tensor):
            return value.cpu().numpy()
        elif isinstance(value, list):
            return np.array(value)
        elif isinstance(value, np.ndarray):
            return value
        else:
            return np.array(value)
    
    @staticmethod
    def _combine_num_cat_shap(shap_num: Any, shap_cat: Any) -> np.ndarray:
        """Numerical과 categorical SHAP 값을 결합."""
        shap_num = ShapValueExtractor._to_numpy(shap_num)
        shap_cat = ShapValueExtractor._to_numpy(shap_cat)
        return np.concatenate([shap_num, shap_cat], axis=-1)
    
    @staticmethod
    def _extract_from_list_of_tuples(shap_list: List[Tuple]) -> np.ndarray:
        """튜플 리스트에서 SHAP 값을 추출."""
        combined = []
        for shap_item in shap_list:
            if isinstance(shap_item, tuple):
                combined.append(ShapValueExtractor._combine_num_cat_shap(shap_item[0], shap_item[1]))
            else:
                combined.append(ShapValueExtractor._to_numpy(shap_item))
        return np.array(combined)
    
    @staticmethod
    def _extract_from_list(shap_list: List[Any]) -> np.ndarray:
        """리스트에서 SHAP 값을 추출."""
        if len(shap_list) == 0:
            return np.array([])
        
        if isinstance(shap_list[0], tuple):
            return ShapValueExtractor._extract_from_list_of_tuples(shap_list)
        
        if len(shap_list) == 2:
            return ShapValueExtractor._combine_num_cat_shap(shap_list[0], shap_list[1])
        
        try:
            return np.array([ShapValueExtractor._to_numpy(item) for item in shap_list])
        except (ValueError, TypeError):
            converted = [ShapValueExtractor._to_numpy(item) for item in shap_list]
            if len(converted) == 2:
                return np.concatenate(converted, axis=-1)
            else:
                return np.array(converted)
    
    @staticmethod
    def extract(shap_values: Any) -> np.ndarray:
        """SHAP 값을 추출하여 numpy array로 변환 (fog = class 0 + class 1)."""
        # 튜플인 경우
        if isinstance(shap_values, tuple):
            return ShapValueExtractor._combine_num_cat_shap(shap_values[0], shap_values[1])
        
        # 리스트인 경우: class 0과 class 1 합산 (fog)
        if isinstance(shap_values, list):
            if len(shap_values) >= 2:
                # class 0과 class 1 추출
                shap_class0 = shap_values[0]
                shap_class1 = shap_values[1]
                
                # 각각을 numpy array로 변환
                if isinstance(shap_class0, tuple):
                    shap_class0 = ShapValueExtractor._combine_num_cat_shap(shap_class0[0], shap_class0[1])
                elif isinstance(shap_class0, list):
                    shap_class0 = ShapValueExtractor._extract_from_list(shap_class0)
                else:
                    shap_class0 = ShapValueExtractor._to_numpy(shap_class0)
                
                if isinstance(shap_class1, tuple):
                    shap_class1 = ShapValueExtractor._combine_num_cat_shap(shap_class1[0], shap_class1[1])
                elif isinstance(shap_class1, list):
                    shap_class1 = ShapValueExtractor._extract_from_list(shap_class1)
                else:
                    shap_class1 = ShapValueExtractor._to_numpy(shap_class1)
                
                # fog = class 0 + class 1
                return shap_class0 + shap_class1
            else:
                # 리스트에 2개 미만인 경우 첫 번째 요소 사용
                target_shap = shap_values[0]
                
                if isinstance(target_shap, tuple):
                    return ShapValueExtractor._combine_num_cat_shap(target_shap[0], target_shap[1])
                
                if isinstance(target_shap, list):
                    return ShapValueExtractor._extract_from_list(target_shap)
                
                return ShapValueExtractor._to_numpy(target_shap)
        
        # numpy array인 경우: class 0과 class 1 합산 (fog)
        if isinstance(shap_values, np.ndarray):
            if len(shap_values.shape) == 3:
                # shape: (n_samples, n_classes, n_features)
                # fog = class 0 + class 1
                return shap_values[:, 0, :] + shap_values[:, 1, :]
            return shap_values
        
        # 기타 타입
        return ShapValueExtractor._to_numpy(shap_values)


@contextmanager
def suppress_shap_output():
    """SHAP DeepExplainer의 출력을 억제하는 컨텍스트 매니저."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    finally:
        sys.stdout = old_stdout


class ModelWrapper(torch.nn.Module):
    """PyTorch 모델을 SHAP DeepExplainer에 맞게 래핑하는 클래스."""
    
    def __init__(self, base_model, numerical_cols):
        super().__init__()
        self.base_model = base_model
        self.num_features = len(numerical_cols)
    
    def forward(self, *inputs):
        """SHAP이 리스트로 전달하는 경우 처리"""
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            x_num, x_cat = inputs[0]
        elif len(inputs) == 2:
            x_num, x_cat = inputs
        else:
            x = inputs[0]
            x_num = x[:, :self.num_features]
            x_cat = x[:, self.num_features:].long()
            return self.base_model(x_num, x_cat)
        
        if x_cat.dtype != torch.long:
            x_cat = x_cat.long()
        
        return self.base_model(x_num, x_cat)
