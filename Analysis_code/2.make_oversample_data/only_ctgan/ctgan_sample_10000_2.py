import pandas as pd
import numpy as np
import os
from pathlib import Path
import optuna
from ctgan import CTGAN
import torch
import warnings

# ==================== 상수 정의 ====================
REGIONS = ['incheon', 'seoul', 'busan', 'daegu', 'daejeon', 'gwangju']
TRAIN_YEARS = [2018, 2020]
TARGET_SAMPLES_CLASS_0 = 10000
TARGET_SAMPLES_CLASS_1_BASE = 10000
RANDOM_STATE = 42

# Optuna 최적화 설정
CLASS_0_TRIALS = 50
CLASS_1_TRIALS = 30

# 클래스별 하이퍼파라미터 탐색 범위
CLASS_0_HP_RANGES = {
    'embedding_dim': (64, 128),
    'generator_dim': [(64, 64), (128, 128)],
    'discriminator_dim': [(64, 64), (128, 128)],
    'pac': [4, 8],
    'batch_size': [64, 128, 256],
    'discriminator_steps': (1, 3)
}

CLASS_1_HP_RANGES = {
    'embedding_dim': (128, 512),
    'generator_dim': [(128, 128), (256, 256)],
    'discriminator_dim': [(128, 128), (256, 256)],
    'pac': [4, 8],
    'batch_size': [256, 512, 1024],
    'discriminator_steps': (1, 5)
}

# 제거할 열 목록
COLUMNS_TO_DROP = ['ground_temp - temp_C', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

# ==================== 유틸리티 함수 ====================

def setup_environment():
    """환경 설정 (GPU, 경고 무시)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    warnings.filterwarnings("ignore", category=UserWarning, module="optuna.distributions")
    return device


def load_and_preprocess_data(file_path: str, train_years: list) -> tuple:
    """
    데이터 로드 및 전처리
    
    Args:
        file_path: 데이터 파일 경로
        train_years: 학습에 사용할 연도 리스트
    
    Returns:
        (data, X, y): 원본 데이터, 특징 데이터, 타겟 데이터
    """
    data = pd.read_csv(file_path, index_col=0)
    data = data.loc[data['year'].isin(train_years), :]
    data['cloudcover'] = data['cloudcover'].astype('int')
    data['lm_cloudcover'] = data['lm_cloudcover'].astype('int')
    
    X = data.drop(columns=['multi_class', 'binary_class'])
    y = data['multi_class']
    
    # 불필요한 열 제거
    X.drop(columns=COLUMNS_TO_DROP, inplace=True)
    
    return data, X, y


def get_categorical_feature_names(df: pd.DataFrame) -> list:
    """범주형 변수의 열 이름 반환"""
    return [col for col, dtype in zip(df.columns, df.dtypes) if dtype != 'float64']


def create_ctgan_objective(data: pd.DataFrame, class_label: int,
                           categorical_features: list,
                           hp_ranges: dict) -> callable:
    """
    Optuna 최적화를 위한 목적 함수 생성
    
    Args:
        data: 학습 데이터
        class_label: 클래스 레이블 (0 또는 1)
        categorical_features: 범주형 변수 이름 리스트
        hp_ranges: 하이퍼파라미터 탐색 범위
    
    Returns:
        Optuna 목적 함수
    """
    class_data = data[data['multi_class'] == class_label]
    
    def objective(trial):
        # 하이퍼파라미터 탐색 범위 설정
        embedding_dim = trial.suggest_int("embedding_dim", *hp_ranges['embedding_dim'])
        generator_dim = trial.suggest_categorical("generator_dim", hp_ranges['generator_dim'])
        discriminator_dim = trial.suggest_categorical("discriminator_dim", hp_ranges['discriminator_dim'])
        pac = trial.suggest_categorical("pac", hp_ranges['pac'])
        batch_size = trial.suggest_categorical("batch_size", hp_ranges['batch_size'])
        discriminator_steps = trial.suggest_int("discriminator_steps", *hp_ranges['discriminator_steps'])
        
        # CTGAN 모델 생성
        ctgan = CTGAN(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            pac=pac
        )
        ctgan.set_random_state(RANDOM_STATE)
        
        # 모델 학습
        ctgan.fit(class_data, discrete_columns=categorical_features)
        
        # 샘플 생성
        generated_data = ctgan.sample(len(class_data) * 2)
        
        # 평가: 샘플의 연속형 변수 분포 비교
        real_visi = class_data['visi']
        generated_visi = generated_data['visi']
        
        # 분포 간 차이(MSE) 계산
        mse = ((real_visi.mean() - generated_visi.mean())**2 + 
               (real_visi.std() - generated_visi.std())**2)
        return -mse
    
    return objective


def optimize_and_generate_samples(data: pd.DataFrame, class_label: int,
                                  categorical_features: list,
                                  hp_ranges: dict, n_trials: int,
                                  target_samples: int) -> tuple:
    """
    CTGAN 최적화 및 샘플 생성
    
    Args:
        data: 학습 데이터
        class_label: 클래스 레이블 (0 또는 1)
        categorical_features: 범주형 변수 이름 리스트
        hp_ranges: 하이퍼파라미터 탐색 범위
        n_trials: Optuna 최적화 시도 횟수
        target_samples: 생성할 샘플 수
    
    Returns:
        (생성된 샘플 데이터프레임, 학습된 CTGAN 모델)
    """
    # 목적 함수 생성
    objective = create_ctgan_objective(data, class_label, categorical_features, hp_ranges)
    
    # Optuna로 최적화 수행
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    # 최적 하이퍼파라미터로 CTGAN 학습 및 샘플 생성
    best_params = study.best_params
    ctgan = CTGAN(
        embedding_dim=best_params["embedding_dim"],
        generator_dim=best_params["generator_dim"],
        discriminator_dim=best_params["discriminator_dim"],
        batch_size=best_params["batch_size"],
        discriminator_steps=best_params["discriminator_steps"],
        pac=best_params["pac"]
    )
    ctgan.set_random_state(RANDOM_STATE)
    
    # 최종 학습 및 샘플 생성
    class_data = data[data['multi_class'] == class_label]
    ctgan.fit(class_data, discrete_columns=categorical_features)
    generated_samples = ctgan.sample(target_samples)
    
    return generated_samples, ctgan


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    제거했던 파생 변수들을 복구
    
    Args:
        df: 데이터프레임
    
    Returns:
        파생 변수가 추가된 데이터프레임
    """
    df = df.copy()
    df['binary_class'] = df['multi_class'].apply(lambda x: 0 if x == 2 else 1)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['ground_temp - temp_C'] = df['groundtemp'] - df['temp_C']
    return df


def process_region(file_path: str, output_path: str, model_save_dir: Path) -> None:
    """
    특정 지역의 데이터에 CTGAN만 적용하여 증강
    
    Args:
        file_path: 입력 데이터 파일 경로
        output_path: 출력 데이터 파일 경로
        model_save_dir: 모델 저장 디렉토리 경로
    """
    # 지역명 추출 (파일 경로에서)
    region_name = Path(file_path).stem.replace('_train', '')
    
    # 데이터 로드 및 전처리
    original_data, X, y = load_and_preprocess_data(file_path, TRAIN_YEARS)
    
    # 원본 데이터에 multi_class 추가
    train_data = X.copy()
    train_data['multi_class'] = y
    
    # CTGAN을 위한 범주형 변수 이름 추출
    categorical_features = get_categorical_feature_names(train_data)
    
    # 클래스별 샘플 수 계산
    count_class_0 = (y == 0).sum()
    count_class_1 = (y == 1).sum()
    target_samples_class_1 = TARGET_SAMPLES_CLASS_1_BASE - count_class_1
    
    # 클래스 0에 대한 CTGAN 최적화 및 샘플 생성
    print(f"Processing {file_path}: Optimizing CTGAN for class 0...")
    generated_0, ctgan_model_0 = optimize_and_generate_samples(
        train_data, 0, categorical_features,
        CLASS_0_HP_RANGES, CLASS_0_TRIALS, TARGET_SAMPLES_CLASS_0
    )
    
    # 클래스 1에 대한 CTGAN 최적화 및 샘플 생성
    print(f"Processing {file_path}: Optimizing CTGAN for class 1...")
    generated_1, ctgan_model_1 = optimize_and_generate_samples(
        train_data, 1, categorical_features,
        CLASS_1_HP_RANGES, CLASS_1_TRIALS, target_samples_class_1
    )
    
    # 모델 저장 디렉토리 생성
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 클래스 0 모델 저장
    model_path_0 = model_save_dir / f'ctgan_only_10000_2_{region_name}_class0.pkl'
    ctgan_model_0.save(str(model_path_0))
    print(f"Saved CTGAN model for class 0: {model_path_0}")
    
    # 클래스 1 모델 저장
    model_path_1 = model_save_dir / f'ctgan_only_10000_2_{region_name}_class1.pkl'
    ctgan_model_1.save(str(model_path_1))
    print(f"Saved CTGAN model for class 1: {model_path_1}")
    
    # 클래스별 가시도 범위로 필터링
    well_generated_0 = generated_0[
        (generated_0['visi'] >= 0) & (generated_0['visi'] < 100)
    ]
    well_generated_1 = generated_1[
        (generated_1['visi'] >= 100) & (generated_1['visi'] < 500)
    ]
    
    # 증강된 데이터만 저장 (CTGAN으로 생성된 샘플만)
    augmented_only = pd.concat([well_generated_0, well_generated_1], axis=0)
    augmented_only = add_derived_features(augmented_only)
    augmented_only.reset_index(drop=True, inplace=True)
    # augmented_only 폴더에 저장
    output_path_obj = Path(output_path)
    augmented_dir = output_path_obj.parent.parent / 'augmented_only'
    augmented_dir.mkdir(parents=True, exist_ok=True)
    augmented_output_path = augmented_dir / output_path_obj.name
    augmented_only.to_csv(augmented_output_path, index=False)
    
    # 원본 데이터와 필터링된 CTGAN 샘플 병합
    ctgan_data = pd.concat([train_data, well_generated_0, well_generated_1], axis=0)
    
    # 파생 변수 추가
    ctgan_data = add_derived_features(ctgan_data)
    
    # 증강된 데이터만 결과 출력
    aug_count_0 = len(augmented_only[augmented_only['multi_class'] == 0])
    aug_count_1 = len(augmented_only[augmented_only['multi_class'] == 1])
    print(f"Saved augmented data only {augmented_output_path}: Class 0={aug_count_0} | Class 1={aug_count_1}")
    
    # 클래스 2 제거 후 원본 클래스 2 데이터 추가
    filtered_data = ctgan_data[ctgan_data['multi_class'] != 2]
    original_class_2 = original_data[original_data['multi_class'] == 2]
    final_data = pd.concat([filtered_data, original_class_2], axis=0)
    final_data.reset_index(drop=True, inplace=True)
    
    # 출력 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 결과 저장
    final_data.to_csv(output_path, index=False)
    
    # 결과 출력
    count_0 = len(final_data[final_data['multi_class'] == 0])
    count_1 = len(final_data[final_data['multi_class'] == 1])
    count_2 = len(final_data[final_data['multi_class'] == 2])
    print(f"Saved {output_path}: Class 0={count_0} | Class 1={count_1} | Class 2={count_2}")


# ==================== 메인 실행 ====================

if __name__ == "__main__":
    setup_environment()
    
    file_paths = [f'../../../data/data_for_modeling/{region}_train.csv' for region in REGIONS]
    output_paths = [f'../../../data/data_oversampled/ctgan10000/ctgan10000_2_{region}.csv' for region in REGIONS]
    model_save_dir = Path('../../save_model/oversampling_models')
    
    for file_path, output_path in zip(file_paths, output_paths):
        process_region(file_path, output_path, model_save_dir)

