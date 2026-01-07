import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, f1_score
import joblib


import sys
# 파일 위치 기반으로 models 디렉토리 경로 설정
current_file_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.abspath(os.path.join(current_file_dir, '../../models'))
sys.path.insert(0, models_path)
from ft_transformer import FTTransformer
from resnet_like import ResNetLike
from deepgbm import DeepGBM
import warnings
warnings.filterwarnings('ignore')

# Python 및 Numpy 시드 고정
seed = 42
random.seed(seed)
np.random.seed(seed)

# PyTorch 시드 고정
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경에서 동일한 시드 적용

# PyTorch 연산의 결정적 모드 설정
torch.backends.cudnn.deterministic = True  # 실행마다 동일한 결과를 보장
torch.backends.cudnn.benchmark = True  # 성능 최적화를 활성화 (가능한 한 빠른 연산 수행)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    제거했던 파생 변수들을 복구
    
    Args:
        df: 데이터프레임
    
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

def preprocessing(df):
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
    df.loc[df['wind_dir']=='정온', 'wind_dir'] = "0"
    df['wind_dir'] = df['wind_dir'].astype('int')
    df = df[['temp_C', 'precip_mm', 'wind_speed', 'wind_dir', 'hm',
       'vap_pressure', 'dewpoint_C', 'loc_pressure', 'sea_pressure',
       'solarRad', 'snow_cm', 'cloudcover', 'lm_cloudcover', 'low_cloudbase',
       'groundtemp', 'O3', 'NO2', 'PM10', 'PM25', 'year',
       'month', 'hour', 'ground_temp - temp_C', 'hour_sin', 'hour_cos',
       'month_sin', 'month_cos','multi_class']].copy()
    return df


# 데이터셋 준비 함수
def prepare_dataset(region, data_sample='pure', target='multi', fold=3):

    # 파일 위치 기반으로 데이터 디렉토리 경로 설정
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.abspath(os.path.join(current_file_dir, '../../../data'))
    
    # 데이터 경로 지정
    dat_path = os.path.join(data_base_dir, f"data_for_modeling/{region}_train.csv")
    if data_sample == 'pure':
        train_path = dat_path
    else:
        train_path = os.path.join(data_base_dir, f'data_oversampled/{data_sample}/{data_sample}_{fold}_{region}.csv')
    test_path = os.path.join(data_base_dir, f"data_for_modeling/{region}_test.csv")
    drop_col = ['multi_class','year']
    target_col = f'{target}_class'
    
    # 데이터 로드
    region_dat = preprocessing(pd.read_csv(dat_path, index_col=0))
    if data_sample == 'pure':
        region_train = region_dat.loc[~region_dat['year'].isin([2021-fold]), :]
    else:
        region_train = preprocessing(pd.read_csv(train_path))
    region_val = region_dat.loc[region_dat['year'].isin([2021-fold]), :]
    region_test = preprocessing(pd.read_csv(test_path))

    # 컬럼 정렬 (일관성 유지)
    common_columns = region_train.columns.to_list()
    train_data = region_train[common_columns]
    val_data = region_val[common_columns]
    test_data = region_test[common_columns]

    # 설명변수 & 타겟 분리
    X_train = train_data.drop(columns=drop_col)
    y_train = train_data[target_col]
    X_val = val_data.drop(columns=drop_col)
    y_val = val_data[target_col]
    X_test = test_data.drop(columns=drop_col)
    y_test = test_data[target_col]

    # 범주형 & 연속형 변수 분리
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'int64']).columns
    numerical_cols = X_train.select_dtypes(include=['float64']).columns

    # 범주형 변수 Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col])  # Train 데이터 기준으로 학습
        label_encoders[col] = le

    # 변환 적용
    for col in categorical_cols:
        X_train[col] = label_encoders[col].transform(X_train[col])
        X_val[col] = label_encoders[col].transform(X_val[col])
        X_test[col] = label_encoders[col].transform(X_test[col])

    # 연속형 변수 Quantile Transformation
    scaler = QuantileTransformer(output_distribution='normal')
    scaler.fit(X_train[numerical_cols])  # Train 데이터 기준으로 학습

    # 변환 적용
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_val, X_test, y_train, y_val, y_test, categorical_cols, numerical_cols



# 데이터 변환 및 dataloader 생성 함수
def prepare_dataloader(region, data_sample='pure', target='multi', fold=3, random_state=None):

    # 파일 위치 기반으로 데이터 디렉토리 경로 설정
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.abspath(os.path.join(current_file_dir, '../../../data'))
    
    # 데이터 경로 지정
    dat_path = os.path.join(data_base_dir, f"data_for_modeling/{region}_train.csv")
    if data_sample == 'pure':
        train_path = dat_path
    else:
        train_path = os.path.join(data_base_dir, f'data_oversampled/{data_sample}/{data_sample}_{fold}_{region}.csv')
    test_path = os.path.join(data_base_dir, f"data_for_modeling/{region}_test.csv")
    drop_col = ['multi_class','year']
    target_col = f'{target}_class'
    
    # 데이터 로드
    region_dat = preprocessing(pd.read_csv(dat_path, index_col=0))
    if data_sample == 'pure':
        region_train = region_dat.loc[~region_dat['year'].isin([2021-fold]), :]
    else:
        region_train = preprocessing(pd.read_csv(train_path))
    region_val = region_dat.loc[region_dat['year'].isin([2021-fold]), :]
    region_test = preprocessing(pd.read_csv(test_path))

    # 컬럼 정렬 (일관성 유지)
    common_columns = region_train.columns.to_list()
    train_data = region_train[common_columns]
    val_data = region_val[common_columns]
    test_data = region_test[common_columns]

    # 설명변수 & 타겟 분리
    X_train = train_data.drop(columns=drop_col)
    y_train = train_data[target_col]
    X_val = val_data.drop(columns=drop_col)
    y_val = val_data[target_col]
    X_test = test_data.drop(columns=drop_col)
    y_test = test_data[target_col]

    # 범주형 & 연속형 변수 분리
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'int64']).columns
    numerical_cols = X_train.select_dtypes(include=['float64']).columns

    # 범주형 변수 Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col])  # Train 데이터 기준으로 학습
        label_encoders[col] = le

    # 변환 적용
    for col in categorical_cols:
        X_train[col] = label_encoders[col].transform(X_train[col])
        X_val[col] = label_encoders[col].transform(X_val[col])
        X_test[col] = label_encoders[col].transform(X_test[col])

    # 연속형 변수 Quantile Transformation
    scaler = QuantileTransformer(output_distribution='normal')
    scaler.fit(X_train[numerical_cols])  # Train 데이터 기준으로 학습

    # 변환 적용
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 연속형 변수와 범주형 변수 분리
    X_train_num = torch.tensor(X_train[numerical_cols].values, dtype=torch.float32)
    X_train_cat = torch.tensor(X_train[categorical_cols].values, dtype=torch.long)

    X_val_num = torch.tensor(X_val[numerical_cols].values, dtype=torch.float32)
    X_val_cat = torch.tensor(X_val[categorical_cols].values, dtype=torch.long)

    X_test_num = torch.tensor(X_test[numerical_cols].values, dtype=torch.float32)
    X_test_cat = torch.tensor(X_test[categorical_cols].values, dtype=torch.long)

    # 레이블 변환
    if target == "binary":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # 이진 분류 → float32
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    elif target == "multi":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # 다중 분류 → long
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    else:
        raise ValueError("target must be 'binary' or 'multi'")

    # TensorDataset 생성
    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train_tensor)
    val_dataset = TensorDataset(X_val_num, X_val_cat, y_val_tensor)
    test_dataset = TensorDataset(X_test_num, X_test_cat, y_test_tensor)

    # DataLoader 생성
    if random_state == None:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(random_state))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return X_train, categorical_cols, numerical_cols, train_loader, val_loader, test_loader

# 데이터 변환 및 dataloader 생성 함수 (batch_size 파라미터 추가 버전)
def prepare_dataloader_with_batchsize(region, data_sample='pure', target='multi', fold=3, random_state=None, batch_size=64):
    # 파일 위치 기반으로 데이터 디렉토리 경로 설정
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.abspath(os.path.join(current_file_dir, '../../../data'))
    
    # 데이터 경로 지정
    dat_path = os.path.join(data_base_dir, f"data_for_modeling/{region}_train.csv")
    if data_sample == 'pure':
        train_path = dat_path
    else:
        train_path = os.path.join(data_base_dir, f'data_oversampled/{data_sample}/{data_sample}_{fold}_{region}.csv')
    test_path = os.path.join(data_base_dir, f"data_for_modeling/{region}_test.csv")
    drop_col = ['multi_class','year']
    target_col = f'{target}_class'
    
    # 데이터 로드
    region_dat = preprocessing(pd.read_csv(dat_path, index_col=0))
    if data_sample == 'pure':
        region_train = region_dat.loc[~region_dat['year'].isin([2021-fold]), :]
    else:
        region_train = preprocessing(pd.read_csv(train_path))
    region_val = region_dat.loc[region_dat['year'].isin([2021-fold]), :]
    region_test = preprocessing(pd.read_csv(test_path))

    # 컬럼 정렬 (일관성 유지)
    common_columns = region_train.columns.to_list()
    train_data = region_train[common_columns]
    val_data = region_val[common_columns]
    test_data = region_test[common_columns]

    # 설명변수 & 타겟 분리
    X_train = train_data.drop(columns=drop_col)
    y_train = train_data[target_col]
    X_val = val_data.drop(columns=drop_col)
    y_val = val_data[target_col]
    X_test = test_data.drop(columns=drop_col)
    y_test = test_data[target_col]

    # 범주형 & 연속형 변수 분리
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'int64']).columns
    numerical_cols = X_train.select_dtypes(include=['float64']).columns

    # 범주형 변수 Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X_train[col])  # Train 데이터 기준으로 학습
        label_encoders[col] = le

    # 변환 적용
    for col in categorical_cols:
        X_train[col] = label_encoders[col].transform(X_train[col])
        X_val[col] = label_encoders[col].transform(X_val[col])
        X_test[col] = label_encoders[col].transform(X_test[col])

    # 연속형 변수 Quantile Transformation
    scaler = QuantileTransformer(output_distribution='normal')
    scaler.fit(X_train[numerical_cols])  # Train 데이터 기준으로 학습

    # 변환 적용
    X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
    X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 연속형 변수와 범주형 변수 분리
    X_train_num = torch.tensor(X_train[numerical_cols].values, dtype=torch.float32)
    X_train_cat = torch.tensor(X_train[categorical_cols].values, dtype=torch.long)

    X_val_num = torch.tensor(X_val[numerical_cols].values, dtype=torch.float32)
    X_val_cat = torch.tensor(X_val[categorical_cols].values, dtype=torch.long)

    X_test_num = torch.tensor(X_test[numerical_cols].values, dtype=torch.float32)
    X_test_cat = torch.tensor(X_test[categorical_cols].values, dtype=torch.long)

    # 레이블 변환
    if target == "binary":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)  # 이진 분류 → float32
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    elif target == "multi":
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # 다중 분류 → long
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    else:
        raise ValueError("target must be 'binary' or 'multi'")

    # TensorDataset 생성
    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train_tensor)
    val_dataset = TensorDataset(X_val_num, X_val_cat, y_val_tensor)
    test_dataset = TensorDataset(X_test_num, X_test_cat, y_test_tensor)

    # DataLoader 생성 (batch_size 파라미터 사용)
    if random_state == None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(random_state))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return X_train, categorical_cols, numerical_cols, train_loader, val_loader, test_loader, y_train, scaler


def calculate_csi(y_true, pred):

    cm = confusion_matrix(y_true, pred)  # 변수 이름을 cm으로 변경
    # 혼동 행렬에서 H, F, M 추출
    H = (cm[0, 0] + cm[1, 1])
    
    F = (cm[1, 0] + cm[2, 0] +
         cm[0, 1] + cm[2, 1])
    
    M = (cm[0, 2] + cm[1, 2])
    
    # CSI 계산
    CSI = H / (H + F + M + 1e-10)
    return CSI

def sample_weight(y_train):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),  # 고유 클래스
        y=y_train                   # 학습 데이터 레이블
    )
    sample_weights = np.array([class_weights[label] for label in y_train])

    return sample_weights

# 하이퍼파라미터 최적화 함수 정의
def objective(trial, model_choose, region, data_sample='pure', target='multi', n_folds=3, random_state=42):
    # GPU 사용 가능 여부 확인 및 device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_scores = []

    # --- 1. 하이퍼파라미터 탐색 범위 정의 (수정됨) ---
    if model_choose == "ft_transformer":
        d_token = trial.suggest_int("d_token", 64, 256, step=32)
        n_blocks = trial.suggest_int("n_blocks", 2, 6) # 깊이 축소로 과적합 방지
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        # d_token은 n_heads의 배수여야 함 (FT-Transformer의 구조적 제약 대응)
        if d_token % n_heads != 0:
            d_token = (d_token // n_heads) * n_heads
            
        attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.4)
        ffn_dropout = trial.suggest_float("ffn_dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True) # 범위 확대
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)  # 더 공격적인 범위로 확장
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # Batch Size 추가

    elif model_choose == 'resnet_like':
        d_main = trial.suggest_int("d_main", 64, 256, step=32)
        d_hidden = trial.suggest_int("d_hidden", 64, 512, step=64)
        n_blocks = trial.suggest_int("n_blocks", 2, 5) # 너무 깊지 않게 조절
        dropout_first = trial.suggest_float("dropout_first", 0.1, 0.4)
        dropout_second = trial.suggest_float("dropout_second", 0.0, 0.2)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)  # 더 공격적인 범위로 확장
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # Batch Size 추가

    elif model_choose == 'deepgbm':
        # DeepGBM의 경우 모델 특성에 맞춰 ResNet 블록 및 임베딩 차원 조절
        d_main = trial.suggest_int("d_main", 64, 256, step=32)
        d_hidden = trial.suggest_int("d_hidden", 64, 256, step=64)
        n_blocks = trial.suggest_int("n_blocks", 2, 6)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)  # 더 공격적인 범위로 확장
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # Batch Size 추가

    # --- 2. Fold별 학습 및 교차 검증 ---
    for fold in range(1, n_folds + 1):
        X_train_df, categorical_cols, numerical_cols, train_loader, val_loader, _, y_train, _ = prepare_dataloader_with_batchsize(
            region, data_sample=data_sample, target=target, fold=fold, random_state=random_state, batch_size=batch_size
        )

        # 모델 초기화
        if model_choose == "ft_transformer":
            model = FTTransformer(
                num_features=len(numerical_cols),
                cat_cardinalities=[len(X_train_df[col].unique()) for col in categorical_cols],
                d_token=d_token,
                n_blocks=n_blocks,
                n_heads=n_heads,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                num_classes=3
            ).to(device)
        elif model_choose == 'resnet_like':
            input_dim = len(numerical_cols) + len(categorical_cols)
            model = ResNetLike(
                input_dim=input_dim,
                d_main=d_main, 
                d_hidden=d_hidden, 
                n_blocks=n_blocks, 
                dropout_first=dropout_first, 
                dropout_second=dropout_second,
                num_classes=3
            ).to(device)
        elif model_choose == 'deepgbm':
            model = DeepGBM(
                num_features=len(numerical_cols),
                cat_features=[len(X_train_df[col].unique()) for col in categorical_cols],
                d_main=d_main,
                d_hidden=d_hidden,
                n_blocks=n_blocks,
                dropout=dropout,
                num_classes=3
            ).to(device)

        # 클래스 가중치 계산 및 손실 함수 설정 (Label Smoothing 적용)
        if target == 'multi':
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            # 클래스별 가중치 로그 출력
            unique_classes = np.unique(y_train)
            class_counts = {cls: np.sum(y_train == cls) for cls in unique_classes}
            print(f"  Fold {fold} - 클래스별 가중치: {dict(zip(unique_classes, class_weights))} (클래스별 샘플 수: {class_counts})")
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.0)  # Label Smoothing 추가
        else:
            criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 학습률 스케줄러 추가: 성능 정체 시 LR을 0.5배 감소 (검증 CSI 기준)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # 학습 설정 (에폭 및 페이션스 상향)
        epochs = 200 
        patience = 12 # 딥러닝의 정체 구간을 고려하여 소폭 상향
        best_fold_csi = 0
        counter = 0

        for epoch in range(epochs):
            model.train()
            for x_num_batch, x_cat_batch, y_batch in train_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(x_num_batch, x_cat_batch)
                loss = criterion(y_pred, y_batch if target == 'multi' else y_batch.float())
                loss.backward()
                optimizer.step()

            # Validation 평가
            model.eval()
            y_pred_val, y_true_val = [], []
            with torch.no_grad():
                for x_num_batch, x_cat_batch, y_batch in val_loader:
                    x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                    output = model(x_num_batch, x_cat_batch)
                    pred = output.argmax(dim=1) if target == 'multi' else (torch.sigmoid(output) >= 0.5).long()
                    
                    y_pred_val.extend(pred.cpu().numpy())
                    y_true_val.extend(y_batch.cpu().numpy())

            # CSI 계산 및 스케줄러 업데이트
            val_csi = calculate_csi(y_true_val, y_pred_val)
            scheduler.step(val_csi)

            # Optuna Pruning 적용 (첫 번째 Fold에서 조기 종료 판단 강화)
            trial.report(val_csi, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Early Stopping 체크
            if val_csi > best_fold_csi:
                best_fold_csi = val_csi
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

        val_scores.append(best_fold_csi)

    # 모든 fold의 평균 성능 반환
    return np.mean(val_scores)


# 최적화된 하이퍼파라미터로 최종 모델 학습 및 저장 함수
def train_final_model(best_params, model_choose, region, data_sample='pure', target='multi', n_folds=3, random_state=42):
    """
    최적화된 하이퍼파라미터로 최종 모델을 학습하고 저장합니다.
    
    Args:
        best_params: 최적화된 하이퍼파라미터 딕셔너리
        model_choose: 모델 선택 ('ft_transformer', 'resnet_like', 'deepgbm')
        region: 지역명
        data_sample: 데이터 샘플 타입 ('pure', 'smote', etc.)
        target: 타겟 타입 ('multi', 'binary')
        n_folds: 교차 검증 fold 수
        random_state: 랜덤 시드
    
    Returns:
        저장된 모델 경로 리스트
    """
    # GPU 사용 가능 여부 확인 및 device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    scalers = []  # scaler 리스트 추가
    
    print("최종 모델 학습 시작...")
    
    for fold in range(1, n_folds + 1):
        print(f"Fold {fold} 학습 중...")
        
        # 최적화된 batch_size 사용
        batch_size = best_params.get("batch_size", 64)
        X_train_df, categorical_cols, numerical_cols, train_loader, val_loader, _, y_train, scaler = prepare_dataloader_with_batchsize(
            region, data_sample=data_sample, target=target, fold=fold, random_state=random_state, batch_size=batch_size
        )
        
        # 모델 초기화
        if model_choose == "ft_transformer":
            d_token = best_params["d_token"]
            n_heads = best_params.get("n_heads", 8)
            # d_token은 n_heads의 배수여야 함 (FT-Transformer의 구조적 제약 대응)
            if d_token % n_heads != 0:
                d_token = (d_token // n_heads) * n_heads
            
            model = FTTransformer(
                num_features=len(numerical_cols),
                cat_cardinalities=[len(X_train_df[col].unique()) for col in categorical_cols],
                d_token=d_token,
                n_blocks=best_params["n_blocks"],
                n_heads=n_heads,
                attention_dropout=best_params["attention_dropout"],
                ffn_dropout=best_params["ffn_dropout"],
                num_classes=3
            ).to(device)
        elif model_choose == 'resnet_like':
            input_dim = len(numerical_cols) + len(categorical_cols)
            model = ResNetLike(
                input_dim=input_dim,
                d_main=best_params["d_main"],
                d_hidden=best_params["d_hidden"],
                n_blocks=best_params["n_blocks"],
                dropout_first=best_params["dropout_first"],
                dropout_second=best_params["dropout_second"],
                num_classes=3
            ).to(device)
        elif model_choose == 'deepgbm':
            model = DeepGBM(
                num_features=len(numerical_cols),
                cat_features=[len(X_train_df[col].unique()) for col in categorical_cols],
                d_main=best_params["d_main"],
                d_hidden=best_params["d_hidden"],
                n_blocks=best_params["n_blocks"],
                dropout=best_params["dropout"],
                num_classes=3
            ).to(device)
        else:
            raise ValueError(f"Unknown model_choose: {model_choose}")
        
        # 클래스 가중치 계산 및 손실 함수 설정 (Label Smoothing 적용)
        if target == 'multi':
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.0)  # Label Smoothing 추가
        else:
            criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        # 학습 설정
        epochs = 200
        patience = 12
        best_fold_csi = 0
        counter = 0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            for x_num_batch, x_cat_batch, y_batch in train_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(x_num_batch, x_cat_batch)
                loss = criterion(y_pred, y_batch if target == 'multi' else y_batch.float())
                loss.backward()
                optimizer.step()
            
            # Validation 평가
            model.eval()
            y_pred_val, y_true_val = [], []
            with torch.no_grad():
                for x_num_batch, x_cat_batch, y_batch in val_loader:
                    x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                    output = model(x_num_batch, x_cat_batch)
                    pred = output.argmax(dim=1) if target == 'multi' else (torch.sigmoid(output) >= 0.5).long()
                    
                    y_pred_val.extend(pred.cpu().numpy())
                    y_true_val.extend(y_batch.cpu().numpy())
            
            # CSI 계산 및 스케줄러 업데이트
            val_csi = calculate_csi(y_true_val, y_pred_val)
            scheduler.step(val_csi)
            
            # Early Stopping 체크
            if val_csi > best_fold_csi:
                best_fold_csi = val_csi
                counter = 0
                best_model = copy.deepcopy(model)
            else:
                counter += 1
            
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}, Best CSI: {best_fold_csi:.4f}")
                break
        
        if best_model is None:
            best_model = model
        
        scalers.append(scaler)  # scaler 저장 (fold 순서대로)
        models.append(best_model)
        print(f"  Fold {fold} 학습 완료 (검증 CSI: {best_fold_csi:.4f})")
    
    # 모델 저장 경로 설정
    save_dir = f'../save_model/{model_choose}_optima'
    os.makedirs(save_dir, exist_ok=True)
    
    # 파일명 생성
    if data_sample == 'pure':
        model_filename = f'{model_choose}_pure_{region}.pkl'
    else:
        model_filename = f'{model_choose}_{data_sample}_{region}.pkl'
    
    model_path = f'{save_dir}/{model_filename}'
    
    # 리스트에 담아 한 번에 저장
    joblib.dump(models, model_path)
    print(f"\n모든 모델 저장 완료: {model_path} (총 {len(models)}개 fold)")
    
    # Scaler 별도 저장
    scaler_save_dir = f'../save_model/{model_choose}_optima/scaler'
    os.makedirs(scaler_save_dir, exist_ok=True)
    
    # 파일명 생성 (모델과 동일한 패턴)
    if data_sample == 'pure':
        scaler_filename = f'{model_choose}_pure_{region}_scaler.pkl'
    else:
        scaler_filename = f'{model_choose}_{data_sample}_{region}_scaler.pkl'
    
    scaler_path = f'{scaler_save_dir}/{scaler_filename}'
    joblib.dump(scalers, scaler_path)
    print(f"Scaler 저장 완료: {scaler_path} (총 {len(scalers)}개 fold)")
    
    return model_path
