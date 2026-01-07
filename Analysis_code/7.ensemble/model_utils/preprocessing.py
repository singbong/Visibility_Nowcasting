"""데이터 전처리 함수."""

import numpy as np
import pandas as pd

from .model_utils import FEATURE_COLUMNS


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """제거했던 파생 변수들을 복구합니다.
    
    Args:
        df: 원본 데이터프레임
        
    Returns:
        파생 변수가 추가된 데이터프레임
    """
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['ground_temp - temp_C'] = df['groundtemp'] - df['temp_C']
    return df


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 전처리 함수.
    
    Args:
        df: 원본 데이터프레임
        
    Returns:
        전처리된 데이터프레임
    """
    df = df[df.columns].copy()
    df['year'] = df['year'].astype('int')
    df['month'] = df['month'].astype('int')
    df['hour'] = df['hour'].astype('int')
    df = add_derived_features(df).copy()
    df['multi_class'] = df['multi_class'].astype('int')
    df.loc[df['wind_dir'] == '정온', 'wind_dir'] = "0"
    df['wind_dir'] = df['wind_dir'].astype('int')
    df = df[FEATURE_COLUMNS].copy()
    return df
