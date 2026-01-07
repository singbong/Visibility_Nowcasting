"""SHAP 분석을 위한 데이터 로딩."""

from typing import Any, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from .config import SHAPConfig
from .preprocessors import PreprocessingUtils


class DataLoader:
    """데이터 로딩 및 전처리를 담당하는 클래스."""
    
    def __init__(self, config: SHAPConfig):
        self.config = config
    
    def load_raw_data(
        self,
        region: str,
        data_sample: str,
        fold_idx: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """원본 데이터 로드 (전처리만 적용)."""
        if data_sample == 'pure':
            dat_path = f"../../data/data_for_modeling/{region}_train.csv"
            train_data = PreprocessingUtils.preprocessing(
                pd.read_csv(dat_path, index_col=0), self.config
            )
            train_data = train_data.loc[
                ~train_data['year'].isin([self.config.TEST_YEAR - (fold_idx + 1)]), :
            ]
        else:
            train_data = pd.read_csv(
                f"../../data/data_oversampled/{data_sample}/{data_sample}_{fold_idx+1}_{region}.csv"
            )
            train_data = PreprocessingUtils.preprocessing(train_data, self.config)
        
        val_data = pd.read_csv(f"../../data/data_for_modeling/{region}_train.csv", index_col=0)
        val_data = PreprocessingUtils.preprocessing(val_data, self.config)
        val_data = val_data.loc[
            val_data['year'].isin([self.config.TEST_YEAR - (fold_idx + 1)]), :
        ]
        
        test_data = pd.read_csv(f"../../data/data_for_modeling/{region}_test.csv", index_col=0)
        test_data = PreprocessingUtils.preprocessing(test_data, self.config)
        test_data = test_data.loc[test_data['year'].isin([self.config.TEST_YEAR]), :]
        
        # 컬럼 정렬
        common_columns = train_data.columns.to_list()
        train_data = train_data[common_columns]
        val_data = val_data[common_columns]
        test_data = test_data[common_columns]
        
        return train_data, val_data, test_data
    
    def prepare_for_tree_model(
        self,
        region: str,
        data_sample: str,
        fold_idx: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Tree 모델용 데이터 준비."""
        train_data = pd.read_csv(
            f"../../data/data_oversampled/{data_sample}/{data_sample}_{fold_idx+1}_{region}.csv"
        )
        val_data = pd.read_csv(f"../../data/data_for_modeling/{region}_train.csv", index_col=0)
        test_data = pd.read_csv(f"../../data/data_for_modeling/{region}_test.csv", index_col=0)
        
        train_data = PreprocessingUtils.preprocessing(train_data, self.config).copy()
        val_data = PreprocessingUtils.preprocessing(val_data, self.config).copy()
        test_data = PreprocessingUtils.preprocessing(test_data, self.config).copy()
        
        train_data = train_data.loc[
            train_data['year'].isin(self.config.FOLD_YEARS[fold_idx]), :
        ]
        val_data = val_data.loc[
            ~val_data['year'].isin(self.config.FOLD_YEARS[fold_idx]), :
        ]
        test_data = test_data.loc[test_data['year'].isin([self.config.TEST_YEAR]), :]
        
        return train_data, val_data, test_data
    
    def prepare_for_dl_model(
        self,
        region: str,
        data_sample: str,
        fold_idx: int,
        scaler: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index, pd.Index,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deep Learning 모델용 데이터 준비 (scaler 적용, 텐서 변환)."""
        train_data, val_data, test_data = self.load_raw_data(region, data_sample, fold_idx)
        
        X_train = train_data.drop(columns=self.config.DROP_COLUMNS)
        X_val = val_data.drop(columns=self.config.DROP_COLUMNS)
        X_test = test_data.drop(columns=self.config.DROP_COLUMNS)
        
        # 범주형 & 연속형 변수 분리
        categorical_cols = X_train.select_dtypes(include=['object', 'category', 'int64']).columns
        numerical_cols = X_train.select_dtypes(include=['float64']).columns
        
        # Label Encoding
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X_train[col])
            label_encoders[col] = le
        
        for col in categorical_cols:
            X_train[col] = label_encoders[col].transform(X_train[col])
            X_val[col] = label_encoders[col].transform(X_val[col])
            X_test[col] = label_encoders[col].transform(X_test[col])
        
        # Scaler 변환
        X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
        X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # 텐서 변환
        X_train_num = torch.tensor(X_train[numerical_cols].values, dtype=torch.float32)
        X_train_cat = torch.tensor(X_train[categorical_cols].values, dtype=torch.long)
        X_val_num = torch.tensor(X_val[numerical_cols].values, dtype=torch.float32)
        X_val_cat = torch.tensor(X_val[categorical_cols].values, dtype=torch.long)
        X_test_num = torch.tensor(X_test[numerical_cols].values, dtype=torch.float32)
        X_test_cat = torch.tensor(X_test[categorical_cols].values, dtype=torch.long)
        
        return (X_train, X_val, X_test, categorical_cols, numerical_cols,
                X_train_num, X_train_cat, X_val_num, X_val_cat, X_test_num, X_test_cat)
