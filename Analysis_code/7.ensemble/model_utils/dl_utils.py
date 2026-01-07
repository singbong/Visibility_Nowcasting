"""Deep Learning 모델용 헬퍼 함수."""

from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .model_utils import DROP_COLUMNS, TEST_YEAR, DL_MODELS
from .preprocessing import preprocessing


def _prepare_test_data_dl(
    region: str,
    data_sample: str,
    fold_idx: int,
    scaler: Any,
    model_name: str = None
) -> Tuple[torch.Tensor, torch.Tensor, pd.Index, pd.Index, np.ndarray]:
    """Deep learning 모델용 test 데이터를 준비합니다.
    
    Args:
        region: 지역명
        data_sample: 데이터 샘플 타입
        fold_idx: Fold 인덱스 (0, 1, 2)
        scaler: 해당 fold의 scaler (QuantileTransformer)
        
    Returns:
        tuple: (X_test_num, X_test_cat, categorical_cols, numerical_cols, y_test)
            - X_test_num: Numerical features tensor
            - X_test_cat: Categorical features tensor
            - categorical_cols: 범주형 변수 컬럼명
            - numerical_cols: 연속형 변수 컬럼명
            - y_test: test 데이터의 target 값 (multi_class)
    """
    # Train 데이터 로드 (Label Encoder 학습용)
    if data_sample == 'pure':
        train_path = f"../../data/data_for_modeling/{region}_train.csv"
        train_data = preprocessing(pd.read_csv(train_path, index_col=0))
        train_data = train_data.loc[
            ~train_data['year'].isin([TEST_YEAR - (fold_idx + 1)]), :
        ]
    else:
        train_data = pd.read_csv(
            f"../../data/data_oversampled/{data_sample}/"
            f"{data_sample}_{fold_idx+1}_{region}.csv"
        )
        train_data = preprocessing(train_data)
    
    # Test 데이터 로드
    test_data = pd.read_csv(f"../../data/data_for_modeling/{region}_test.csv", index_col=0)
    test_data = preprocessing(test_data)
    test_data = test_data.loc[test_data['year'].isin([TEST_YEAR]), :]
    
    # 컬럼 정렬 (일관성 유지)
    common_columns = train_data.columns.to_list()
    train_data = train_data[common_columns]
    test_data = test_data[common_columns]
    
    # 설명변수 분리
    X_train = train_data.drop(columns=DROP_COLUMNS)
    X_test = test_data.drop(columns=DROP_COLUMNS)
    
    # 범주형 & 연속형 변수 분리
    original_categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'int64']
    ).columns
    numerical_cols = X_train.select_dtypes(include=['float64']).columns
    
    # resnet_like 모델이고 pure 데이터 샘플일 때 One-Hot Encoding 사용
    use_onehot = (model_name == 'resnet_like' and data_sample == 'pure')
    
    if use_onehot:
        # 범주형 변수 One-Hot Encoding (train 데이터 기준으로 학습)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
        ohe.fit(X_train[original_categorical_cols])
        
        # One-Hot 인코딩 적용
        X_train_cat_encoded = pd.DataFrame(
            ohe.transform(X_train[original_categorical_cols]),
            columns=ohe.get_feature_names_out(original_categorical_cols),
            index=X_train.index
        )
        X_test_cat_encoded = pd.DataFrame(
            ohe.transform(X_test[original_categorical_cols]),
            columns=ohe.get_feature_names_out(original_categorical_cols),
            index=X_test.index
        )
        
        # 원래 범주형 컬럼 제거 후 One-Hot 인코딩된 컬럼 추가
        X_train = X_train.drop(columns=original_categorical_cols)
        X_test = X_test.drop(columns=original_categorical_cols)
        
        X_train = pd.concat([X_train, X_train_cat_encoded], axis=1)
        X_test = pd.concat([X_test, X_test_cat_encoded], axis=1)
        
        # 연속형 변수 Scaler 변환
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # 모든 features를 numerical로 합쳐서 텐서 변환
        X_test_all = torch.tensor(
            X_test.values, dtype=torch.float32
        )
        X_test_num = X_test_all
        X_test_cat = torch.zeros((X_test_all.shape[0], 0), dtype=torch.long)  # 빈 텐서
        
        categorical_cols = pd.Index([])  # 빈 인덱스
    else:
        # 범주형 변수 Label Encoding (train 데이터 기준으로 학습)
        label_encoders = {}
        for col in original_categorical_cols:
            le = LabelEncoder()
            le.fit(X_train[col])
            label_encoders[col] = le
        
        # 변환 적용
        for col in original_categorical_cols:
            X_test[col] = label_encoders[col].transform(X_test[col])
        
        # 연속형 변수 Scaler 변환
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # 텐서 변환
        X_test_num = torch.tensor(
            X_test[numerical_cols].values, dtype=torch.float32
        )
        X_test_cat = torch.tensor(
            X_test[original_categorical_cols].values, dtype=torch.long
        )
        
        categorical_cols = original_categorical_cols
    
    # Target 값 추출
    y_test = test_data['multi_class'].values
    
    return X_test_num, X_test_cat, categorical_cols, numerical_cols, y_test


def _prepare_val_data_dl(
    region: str,
    data_sample: str,
    fold_idx: int,
    scaler: Any,
    model_name: str = None
) -> Tuple[torch.Tensor, torch.Tensor, pd.Index, pd.Index, np.ndarray]:
    """Deep learning 모델용 validation 데이터를 준비합니다.
    
    Args:
        region: 지역명
        data_sample: 데이터 샘플 타입
        fold_idx: Fold 인덱스 (0, 1, 2) - fold 0: 2020, fold 1: 2019, fold 2: 2018
        scaler: 해당 fold의 scaler (QuantileTransformer)
        
    Returns:
        tuple: (X_val_num, X_val_cat, categorical_cols, numerical_cols, y_val)
            - X_val_num: Numerical features tensor
            - X_val_cat: Categorical features tensor
            - categorical_cols: 범주형 변수 컬럼명
            - numerical_cols: 연속형 변수 컬럼명
            - y_val: validation 데이터의 target 값 (multi_class)
    """
    val_year = TEST_YEAR - (fold_idx + 1)  # fold 0: 2020, fold 1: 2019, fold 2: 2018
    
    # Train 데이터 로드 (Label Encoder 학습용)
    if data_sample == 'pure':
        train_path = f"../../data/data_for_modeling/{region}_train.csv"
        train_data = preprocessing(pd.read_csv(train_path, index_col=0))
        train_data = train_data.loc[
            ~train_data['year'].isin([val_year]), :
        ]
    else:
        train_data = pd.read_csv(
            f"../../data/data_oversampled/{data_sample}/"
            f"{data_sample}_{fold_idx+1}_{region}.csv"
        )
        train_data = preprocessing(train_data)
    
    # Validation 데이터 로드
    val_data = pd.read_csv(f"../../data/data_for_modeling/{region}_train.csv", index_col=0)
    val_data = preprocessing(val_data)
    val_data = val_data.loc[val_data['year'].isin([val_year]), :]
    
    # 컬럼 정렬 (일관성 유지)
    common_columns = train_data.columns.to_list()
    train_data = train_data[common_columns]
    val_data = val_data[common_columns]
    
    # 설명변수 분리
    X_train = train_data.drop(columns=DROP_COLUMNS)
    X_val = val_data.drop(columns=DROP_COLUMNS)
    
    # 범주형 & 연속형 변수 분리
    original_categorical_cols = X_train.select_dtypes(
        include=['object', 'category', 'int64']
    ).columns
    numerical_cols = X_train.select_dtypes(include=['float64']).columns
    
    # resnet_like 모델이고 pure 데이터 샘플일 때 One-Hot Encoding 사용
    use_onehot = (model_name == 'resnet_like' and data_sample == 'pure')
    
    if use_onehot:
        # 범주형 변수 One-Hot Encoding (train 데이터 기준으로 학습)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
        ohe.fit(X_train[original_categorical_cols])
        
        # One-Hot 인코딩 적용
        X_train_cat_encoded = pd.DataFrame(
            ohe.transform(X_train[original_categorical_cols]),
            columns=ohe.get_feature_names_out(original_categorical_cols),
            index=X_train.index
        )
        X_val_cat_encoded = pd.DataFrame(
            ohe.transform(X_val[original_categorical_cols]),
            columns=ohe.get_feature_names_out(original_categorical_cols),
            index=X_val.index
        )
        
        # 원래 범주형 컬럼 제거 후 One-Hot 인코딩된 컬럼 추가
        X_train = X_train.drop(columns=original_categorical_cols)
        X_val = X_val.drop(columns=original_categorical_cols)
        
        X_train = pd.concat([X_train, X_train_cat_encoded], axis=1)
        X_val = pd.concat([X_val, X_val_cat_encoded], axis=1)
        
        # 연속형 변수 Scaler 변환
        X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
        
        # 모든 features를 numerical로 합쳐서 텐서 변환
        X_val_all = torch.tensor(
            X_val.values, dtype=torch.float32
        )
        X_val_num = X_val_all
        X_val_cat = torch.zeros((X_val_all.shape[0], 0), dtype=torch.long)  # 빈 텐서
        
        categorical_cols = pd.Index([])  # 빈 인덱스
    else:
        # 범주형 변수 Label Encoding (train 데이터 기준으로 학습)
        label_encoders = {}
        for col in original_categorical_cols:
            le = LabelEncoder()
            le.fit(X_train[col])
            label_encoders[col] = le
        
        # 변환 적용
        for col in original_categorical_cols:
            X_val[col] = label_encoders[col].transform(X_val[col])
        
        # 연속형 변수 Scaler 변환
        X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
        
        # 텐서 변환
        X_val_num = torch.tensor(
            X_val[numerical_cols].values, dtype=torch.float32
        )
        X_val_cat = torch.tensor(
            X_val[original_categorical_cols].values, dtype=torch.long
        )
        
        categorical_cols = original_categorical_cols
    
    # Target 값 추출
    y_val = val_data['multi_class'].values
    
    return X_val_num, X_val_cat, categorical_cols, numerical_cols, y_val


def _predict_dl_model(
    model: torch.nn.Module,
    X_test_num: torch.Tensor,
    X_test_cat: torch.Tensor,
    model_name: str,
    device: str,
    batch_size: int = 1024
) -> np.ndarray:
    """Deep learning 모델로 예측을 수행합니다.
    
    Args:
        model: PyTorch 모델
        X_test_num: Numerical features tensor
        X_test_cat: Categorical features tensor
        model_name: 모델 이름 ('ft_transformer', 'resnet_like', 'deepgbm')
        device: 연산 장치 ('cpu' 또는 'cuda')
        batch_size: 배치 크기 (메모리 효율을 위해, 기본값: 1024)
    
    Returns:
        예측 확률 (n_samples, n_classes)
        
    Raises:
        ValueError: 알 수 없는 모델 이름인 경우
    """
    model.eval()
    model.to(device)
    
    n_samples = X_test_num.shape[0]
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_num = X_test_num[i:end_idx].to(device)
            batch_cat = X_test_cat[i:end_idx].to(device)
            
            # 모델별 입력 형식 처리
            # 모든 딥러닝 모델이 (x_num, x_cat) 두 개의 텐서를 받음
            # resnet_like는 내부에서 torch.cat으로 결합함
            if model_name in DL_MODELS:
                # Feature 수 불일치 체크 (resnet_like의 경우)
                if model_name == 'resnet_like' and hasattr(model, 'input_layer'):
                    expected_features = model.input_layer.weight.shape[1]
                    actual_features = batch_num.shape[1] + batch_cat.shape[1]
                    
                    if expected_features != actual_features:
                        error_msg = (
                            f"Feature 수 불일치: 모델은 {expected_features}개 feature를 기대하지만 "
                            f"실제 데이터는 {actual_features}개 feature를 제공합니다. "
                            f"(numerical: {batch_num.shape[1]}, categorical: {batch_cat.shape[1]}) "
                            f"모델이 다른 데이터 샘플 타입으로 학습되었을 수 있습니다."
                        )
                        raise ValueError(error_msg)
                
                try:
                    output = model(batch_num, batch_cat)
                except RuntimeError as e:
                    raise
            else:
                raise ValueError(
                    f"Unknown model_name: {model_name}. "
                    f"Must be one of {DL_MODELS}"
                )
            
            # Softmax 적용하여 확률 변환
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    return np.vstack(all_probs)
