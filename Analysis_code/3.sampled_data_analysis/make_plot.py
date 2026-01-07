"""
데이터 시각화 모듈: Original과 Synthetic 데이터 비교 시각화

이 모듈은 원본 데이터와 합성 데이터를 로드하고, 전처리한 후
UMAP을 사용하여 차원 축소 및 시각화를 수행합니다.
"""

import os
# TensorFlow 로그 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=모두, 1=INFO 제외, 2=INFO/WARNING 제외, 3=ERROR만
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN 경고 숨기기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path


@dataclass
class PlotConfig:
    """시각화 설정값을 관리하는 클래스"""
    cols_to_drop: List[str] = None
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_random_state: int = 42
    umap_n_jobs: int = 1  # random_state 설정 시 병렬 처리 불가 (경고 방지)
    figsize: Tuple[int, int] = (16, 6)
    alpha: float = 0.6  # Original과 Synthetic 데이터 모두 동일한 투명도
    visibility_threshold: int = 500
    scale_on_original_only: bool = True  # True: 원본 기준 스케일링 (데이터 누설 방지), False: 합쳐서 스케일링
    
    def __post_init__(self):
        """기본값 설정"""
        if self.cols_to_drop is None:
            self.cols_to_drop = [
                'wind_dir',              # 문자열 (에러 발생)
                'multi_class',           # 타겟 변수 (시각화용 색깔로만 사용)
                'binary_class',          # 타겟 변수
                'year', 'month', 'hour', # sin/cos 변수와 중복
                'ground_temp - temp_C',  # 단순 선형 결합 (정보 중복)
                'visi'
            ]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력 데이터프레임에 시간 관련 파생변수를 추가합니다.
    
    Args:
        df: 입력 데이터프레임 (hour, month 컬럼 필요)
        
    Returns:
        시간 특성이 추가된 데이터프레임
    """
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def create_binary_class(visi: pd.Series, threshold: int = 500) -> pd.Series:
    """
    가시도(visi) 값을 기반으로 이진 분류를 생성합니다.
    
    Args:
        visi: 가시도 값 시리즈
        threshold: 이진 분류 임계값 (기본값: 500)
        
    Returns:
        이진 분류 결과 (1: < threshold, 0: >= threshold)
    """
    return visi.apply(lambda x: 1 if x < threshold else (0 if x >= threshold else np.nan))


def load_region_data(
    region: str,
    data_dir: str = "../../data/data_for_modeling"
) -> pd.DataFrame:
    """
    특정 지역의 원본 데이터를 로드합니다.
    
    Args:
        region: 지역명 ('incheon', 'seoul', 'busan', 'daegu', 'daejeon', 'gwangju')
        data_dir: 데이터 디렉토리 경로
        
    Returns:
        로드된 지역 데이터프레임
    """
    file_path = f"{data_dir}/{region}_train.csv"
    df = pd.read_csv(file_path)
    
    # 필요한 컬럼만 선택
    required_cols = [
        'temp_C', 'precip_mm', 'wind_speed', 'wind_dir', 'hm', 'vap_pressure',
        'dewpoint_C', 'loc_pressure', 'sea_pressure', 'solarRad', 'snow_cm',
        'cloudcover', 'lm_cloudcover', 'low_cloudbase', 'groundtemp', 'O3',
        'NO2', 'PM10', 'PM25', 'year', 'month', 'hour', 'visi', 'multi_class',
        'binary_class', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'ground_temp - temp_C'
    ]
    
    # 존재하는 컬럼만 선택
    available_cols = [col for col in required_cols if col in df.columns]
    df = df.loc[:, available_cols].copy()
    
    return df


def load_and_preprocess_data(
    synthetic_path: str,
    config: PlotConfig,
    region: Optional[str] = None,
    fold_idx: Optional[int] = None,
    data_dir: str = "../../data/data_for_modeling",
    original_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    원본 및 합성 데이터를 로드하고 전처리합니다.
    
    Args:
        synthetic_path: 합성 데이터 파일 경로
        config: PlotConfig 객체
        region: 지역명 ('incheon', 'seoul', 'busan', 'daegu', 'daejeon', 'gwangju')
                original_path가 None일 때 사용
        fold_idx: fold 인덱스 (0, 1, 2 중 하나), None이면 전체 데이터
                  original_path가 None일 때 사용
        data_dir: 원본 데이터 디렉토리 경로 (region 사용 시)
        original_path: 원본 데이터 파일 경로 (지정하면 region/fold 무시)
        
    Returns:
        (전처리된 원본 데이터, 전처리된 합성 데이터) 튜플
    """
    # 원본 데이터 로드
    if original_path is not None:
        # 기존 방식: 파일 경로로 직접 로드
        original_data = pd.read_csv(original_path)
    elif region is not None:
        # 새로운 방식: 지역과 fold로 로드
        original_data = load_region_data(region, data_dir)
        
        # fold에 따라 필터링
        if fold_idx is not None:
            fold = [[2018, 2019], [2018, 2020], [2019, 2020]]
            if 0 <= fold_idx < len(fold):
                years = fold[fold_idx]
                original_data = original_data.loc[original_data['year'].isin(years), :].copy()
    else:
        raise ValueError("original_path 또는 region을 지정해야 합니다.")
    
    # 합성 데이터 로드
    synthetic_data = pd.read_csv(synthetic_path)
    
    # 이진 분류 생성
    original_data['binary_class'] = create_binary_class(
        original_data['visi'], 
        config.visibility_threshold
    )
    synthetic_data['binary_class'] = create_binary_class(
        synthetic_data['visi'],
        config.visibility_threshold
    )
    
    # 시간 특성 추가
    original_data = add_time_features(original_data)
    synthetic_data = add_time_features(synthetic_data)
    
    # multi_class 필터링 (Original만)
    original_data = original_data.loc[original_data['multi_class'].isin([0, 1]), :]
    
    # 라벨 추가
    original_data['Label'] = 'Original'
    synthetic_data['Label'] = 'Synthetic'
    
    # 불필요한 컬럼 제거
    original_data = original_data.drop(config.cols_to_drop, axis=1)
    synthetic_data = synthetic_data.drop(config.cols_to_drop, axis=1)
    
    return original_data, synthetic_data


def prepare_features_for_visualization(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    config: PlotConfig
) -> Tuple[np.ndarray, pd.Series, StandardScaler]:
    """
    시각화를 위한 피처를 준비하고 스케일링합니다.
    
    중요: 데이터 누설을 방지하기 위해 기본적으로 원본 데이터로만 scaler를 fit하고,
    합성 데이터는 transform만 합니다. 이렇게 하면 합성 데이터의 분포가 원본 데이터의
    스케일링에 영향을 주지 않습니다.
    
    Args:
        original_data: 원본 데이터프레임
        synthetic_data: 합성 데이터프레임
        config: PlotConfig 객체 (scale_on_original_only 설정 포함)
        
    Returns:
        (스케일링된 피처, 라벨, 스케일러) 튜플
    """
    # 피처와 라벨 분리
    original_features = original_data.drop('Label', axis=1)
    synthetic_features = synthetic_data.drop('Label', axis=1)
    
    if config.scale_on_original_only:
        # 방법 1: 원본 데이터로만 scaler fit (데이터 누설 방지, 권장)
        # 이 방법은 합성 데이터가 원본 데이터의 스케일링에 영향을 주지 않습니다.
        scaler = StandardScaler()
        scaled_original = scaler.fit_transform(original_features)
        scaled_synthetic = scaler.transform(synthetic_features)
        
        # 스케일링된 데이터 합치기
        scaled_features = np.vstack([scaled_original, scaled_synthetic])
        
        # 라벨 합치기
        labels = pd.concat([
            original_data['Label'],
            synthetic_data['Label']
        ], ignore_index=True)
    else:
        # 방법 2: 합쳐서 스케일링 (데이터 누설 있음, 비교 목적일 때만 사용)
        # 주의: 이 방법은 합성 데이터의 분포가 원본 스케일링에 영향을 줍니다.
        combined_df = pd.concat([original_data, synthetic_data], ignore_index=True)
        features = combined_df.drop('Label', axis=1)
        labels = combined_df['Label']
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    
    return scaled_features, labels, scaler


def plot_umap_comparison(
    scaled_features: np.ndarray,
    labels: pd.Series,
    config: PlotConfig,
    region: Optional[str] = None,
    fold_idx: Optional[int] = None,
    method: Optional[str] = None,
    sample_size: Optional[int] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    UMAP을 사용하여 차원 축소 후 Original과 Synthetic 데이터를 비교 시각화합니다.
    
    핵심: 원본 데이터가 정의한 공간(Manifold) 위에 합성 데이터를 투영합니다.
    - Original 데이터로만 UMAP을 fit하여 공간 구조를 학습
    - Synthetic 데이터는 학습된 공간에 transform만 적용
    - 이렇게 하면 합성 데이터가 원본 데이터의 공간 형성에 영향을 주지 않습니다.
    
    Args:
        scaled_features: 스케일링된 피처 배열
        labels: 데이터 라벨 (Original/Synthetic)
        config: PlotConfig 객체
        region: 지역명 (표시용)
        fold_idx: fold 인덱스 (표시용)
        method: 증강 방법 ('ctgan', 'smotenc_ctgan', 'smote') (표시용)
        sample_size: 샘플 수 (표시용)
        ax: matplotlib axes 객체 (None이면 새 figure 생성)
        
    Returns:
        matplotlib Figure 객체
    """
    print("UMAP 실행 중... (Original 기준 학습 후 Synthetic 변환)")
    
    # 1. 데이터 분리 (Labels를 이용해서 다시 나눔)
    is_original = labels == 'Original'
    original_data = scaled_features[is_original]
    synthetic_data = scaled_features[~is_original]

    # 2. UMAP 모델 생성
    umap_model = umap.UMAP(
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        random_state=config.umap_random_state,
        n_jobs=config.umap_n_jobs
    )

    # 3. [핵심] Original 데이터로만 공간 학습 (Fit)
    # 원본 데이터의 구조(Manifold)만 학습합니다.
    original_embedding = umap_model.fit_transform(original_data)

    # 4. [핵심] 학습된 공간에 Synthetic 데이터 투영 (Transform)
    # 합성 데이터는 공간 형성에 관여하지 않고, 이미 만들어진 공간에 위치만 찾습니다.
    synthetic_embedding = umap_model.transform(synthetic_data)

    # 5. 결과 합치기 (시각화를 위해)
    umap_results = np.vstack([original_embedding, synthetic_embedding])
    
    # 순서 보장을 위해 라벨도 다시 정리 (Original이 앞, Synthetic이 뒤)
    combined_labels = pd.concat([
        labels[is_original], 
        labels[~is_original]
    ], ignore_index=True)

    # 결과를 데이터프레임으로 변환
    df_umap = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
    df_umap['Label'] = combined_labels

    # 지역, fold, method, sample_size 정보 문자열 생성 (title에 사용)
    # 첫 번째 줄: Method, Region, Fold 정보
    title_parts_line1 = []
    
    # Method 정보 추가
    if method is not None:
        method_display = method.upper()
        if sample_size is not None:
            method_display += f" (n={sample_size})"
        title_parts_line1.append(f"Method: {method_display}")
    
    # Region 및 Fold 정보 추가
    if region is not None:
        if fold_idx is not None:
            fold = [[2018, 2019], [2018, 2020], [2019, 2020]]
            if 0 <= fold_idx < len(fold):
                years = fold[fold_idx]
                fold_display = fold_idx + 1  # fold를 +1해서 표시
                title_parts_line1.append(f"Region: {region.upper()} | Fold {fold_display}: {years[0]}-{years[1]}")
            else:
                title_parts_line1.append(f"Region: {region.upper()}")
        else:
            title_parts_line1.append(f"Region: {region.upper()}")
    elif fold_idx is not None:
        fold = [[2018, 2019], [2018, 2020], [2019, 2020]]
        if 0 <= fold_idx < len(fold):
            years = fold[fold_idx]
            fold_display = fold_idx + 1  # fold를 +1해서 표시
            title_parts_line1.append(f"Fold {fold_display}: {years[0]}-{years[1]}")
    
    # 두 줄로 나누기: 첫 번째 줄은 정보, 두 번째 줄은 "UMAP: Original vs Synthetic"
    title_line1 = " - ".join(title_parts_line1) if title_parts_line1 else ""
    title_line2 = "UMAP: Original vs Synthetic"
    
    # 두 줄 제목 결합
    if title_line1:
        title_str = f"{title_line1}\n{title_line2}"
    else:
        title_str = title_line2

    # Figure 및 Axes 설정 (단일 플롯)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure if hasattr(ax, 'figure') else plt.gcf()
    
    # 전체 데이터의 UMAP 범위 계산
    x_min = df_umap['UMAP1'].min() - 1
    x_max = df_umap['UMAP1'].max() + 1
    y_min = df_umap['UMAP2'].min() - 1
    y_max = df_umap['UMAP2'].max() + 1
    
    # Synthetic 데이터 시각화 (빨간색, 먼저 그려서 뒤에 위치)
    sns.scatterplot(
        data=df_umap.loc[df_umap['Label'] == 'Synthetic'],
        x='UMAP1', y='UMAP2',
        color='red',
        alpha=config.alpha,
        label='Synthetic',
        ax=ax,
        s=60  # 점 크기 증가
    )
    
    # Original 데이터 시각화 (파란색, 나중에 그려서 앞에 위치하여 더 잘 보이게)
    sns.scatterplot(
        data=df_umap.loc[df_umap['Label'] == 'Original'],
        x='UMAP1', y='UMAP2',
        color='blue',
        alpha=config.alpha,
        label='Original',
        ax=ax,
        s=60  # 점 크기 증가
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('UMAP1', fontsize=21)
    ax.set_ylabel('UMAP2', fontsize=21)
    ax.set_title(title_str, fontsize=22, fontweight='bold')
    ax.tick_params(labelsize=20)  # x축, y축 tick label 크기
    ax.legend(title='Label', loc='best', fontsize=20, title_fontsize=22)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_synthetic_path(
    method: str,
    region: str,
    sample_size: Optional[int] = None,
    fold_idx: Optional[int] = None,
    base_dir: str = "../../data/data_oversampled"
) -> str:
    """
    합성 데이터 파일 경로를 생성합니다.
    
    Args:
        method: 증강 방법 ('ctgan', 'smotenc_ctgan', 'smote')
        region: 지역명
        sample_size: 샘플 수 (ctgan, smotenc_ctgan인 경우 필수, smote인 경우 무시)
        fold_idx: fold 인덱스 (0, 1, 2 중 하나, None이면 0 사용)
        base_dir: 기본 디렉토리 경로
        
    Returns:
        합성 데이터 파일 경로
    """
    # fold_idx 기본값 설정 (None이면 0, 즉 fold 1)
    if fold_idx is None:
        fold_idx = 0
    fold_num = fold_idx + 1  # 파일명은 1부터 시작 (fold_idx는 0부터)
    
    if method == 'ctgan':
        if sample_size is None:
            raise ValueError("ctgan 방법은 sample_size가 필요합니다 (7000, 10000, 20000 중 선택)")
        if sample_size not in [7000, 10000, 20000]:
            raise ValueError(f"sample_size는 7000, 10000, 20000 중 하나여야 합니다. 입력값: {sample_size}")
        return f"{base_dir}/augmented_only/ctgan{sample_size}_{fold_num}_{region}.csv"
    
    elif method == 'smotenc_ctgan':
        if sample_size is None:
            raise ValueError("smotenc_ctgan 방법은 sample_size가 필요합니다 (7000, 10000, 20000 중 선택)")
        if sample_size not in [7000, 10000, 20000]:
            raise ValueError(f"sample_size는 7000, 10000, 20000 중 하나여야 합니다. 입력값: {sample_size}")
        return f"{base_dir}/augmented_only/smotenc_ctgan{sample_size}_{fold_num}_{region}.csv"
    
    elif method == 'smote':
        # smote는 sample_size를 사용하지 않으므로 무시
        # smote 파일도 augmented_only에 있다고 가정 (fold 번호 포함 여부 확인 필요)
        return f"{base_dir}/augmented_only/smote_{fold_num}_{region}.csv"
    
    else:
        raise ValueError(f"지원하지 않는 method입니다: {method}. 'ctgan', 'smotenc_ctgan', 'smote' 중 하나를 선택하세요.")


def main(
    method: str = "ctgan",
    sample_size: Optional[int] = 7000,
    config: Optional[PlotConfig] = None,
    region: Optional[str] = "busan",
    fold_idx: Optional[int] = 0,
    data_dir: str = "../../data/data_for_modeling",
    original_path: Optional[str] = None,
    synthetic_path: Optional[str] = None,
    base_dir: str = "../../data/data_oversampled"
) -> None:
    """
    전체 파이프라인을 실행하는 메인 함수.
    
    Args:
        method: 증강 방법 ('ctgan', 'smotenc_ctgan', 'smote')
        sample_size: 샘플 수 (ctgan, smotenc_ctgan인 경우: 7000, 10000, 20000 중 선택, smote인 경우 무시)
        config: PlotConfig 객체 (None이면 기본값 사용)
        region: 지역명 ('incheon', 'seoul', 'busan', 'daegu', 'daejeon', 'gwangju')
                original_path가 None일 때 사용
        fold_idx: fold 인덱스 (0, 1, 2 중 하나), None이면 전체 데이터
                  original_path가 None일 때 사용
        data_dir: 원본 데이터 디렉토리 경로 (region 사용 시)
        original_path: 원본 데이터 파일 경로 (지정하면 region/fold 무시)
        synthetic_path: 합성 데이터 파일 경로 (지정하면 method/sample_size 무시)
        base_dir: 합성 데이터 기본 디렉토리 경로
    """
    if config is None:
        config = PlotConfig()
    
    # 합성 데이터 경로 생성
    if synthetic_path is None:
        if region is None:
            raise ValueError("synthetic_path를 지정하지 않으면 region이 필요합니다.")
        synthetic_path = generate_synthetic_path(method, region, sample_size, fold_idx, base_dir)
    
    # 데이터 로드 및 전처리
    original_data, synthetic_data = load_and_preprocess_data(
        synthetic_path=synthetic_path,
        config=config,
        region=region,
        fold_idx=fold_idx,
        data_dir=data_dir,
        original_path=original_path
    )
    
    # 피처 준비 및 스케일링
    scaled_features, labels, scaler = prepare_features_for_visualization(
        original_data, synthetic_data, config
    )
    
    # UMAP 시각화
    plot_umap_comparison(
        scaled_features, 
        labels, 
        config,
        region=region,
        fold_idx=fold_idx,
        method=method,
        sample_size=sample_size
    )
    plt.show()


def generate_all_plots(
    output_dir: str = "images",
    config: Optional[PlotConfig] = None,
    data_dir: str = "../../data/data_for_modeling",
    base_dir: str = "../../data/data_oversampled"
) -> None:
    """
    논문 게재를 위한 모든 조합의 plot을 생성하고 저장합니다.
    
    생성되는 조합:
    - 지역: incheon, seoul, busan, daegu, daejeon, gwangju (6개)
    - Fold: 0, 1, 2 (3개)
    - Method: ctgan (n=10000), smotenc_ctgan (n=10000), smote (3개)
    
    총: (6 지역 × 3 fold × 1 ctgan) + (6 지역 × 3 fold × 1 smotenc_ctgan) + (6 지역 × 3 fold × 1 smote) = 54개
    
    Args:
        output_dir: 저장할 디렉토리 경로
        config: PlotConfig 객체 (None이면 기본값 사용)
        data_dir: 원본 데이터 디렉토리 경로
        base_dir: 합성 데이터 기본 디렉토리 경로
    """
    if config is None:
        config = PlotConfig()
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 조합 정의
    regions = ['incheon', 'seoul', 'busan', 'daegu', 'daejeon', 'gwangju']
    fold_indices = [0, 1, 2]
    methods_with_size = [
        ('ctgan', 10000),
        ('smotenc_ctgan', 10000)
    ]
    methods_without_size = [('smote', None)]
    
    total_plots = len(regions) * len(fold_indices) * (len(methods_with_size) + len(methods_without_size))
    current_plot = 0
    
    print(f"총 {total_plots}개의 plot을 생성합니다...")
    print("=" * 60)
    
    # Method와 sample_size가 있는 경우 (ctgan, smotenc_ctgan)
    for method, sample_size in methods_with_size:
        for region in regions:
            for fold_idx in fold_indices:
                current_plot += 1
                try:
                    print(f"[{current_plot}/{total_plots}] {method} (size={sample_size}) - {region.upper()} - Fold {fold_idx + 1} 생성 중...")
                    
                    # 합성 데이터 경로 생성
                    synthetic_path = generate_synthetic_path(method, region, sample_size, fold_idx, base_dir)
                    
                    # 데이터 로드 및 전처리
                    original_data, synthetic_data = load_and_preprocess_data(
                        synthetic_path=synthetic_path,
                        config=config,
                        region=region,
                        fold_idx=fold_idx,
                        data_dir=data_dir,
                        original_path=None
                    )
                    
                    # 피처 준비 및 스케일링
                    scaled_features, labels, scaler = prepare_features_for_visualization(
                        original_data, synthetic_data, config
                    )
                    
                    # UMAP 시각화
                    fig = plot_umap_comparison(
                        scaled_features, 
                        labels, 
                        config,
                        region=region,
                        fold_idx=fold_idx,
                        method=method,
                        sample_size=sample_size
                    )
                    
                    # 파일명 생성: method_sample_size_region_fold_years.png
                    fold = [[2018, 2019], [2018, 2020], [2019, 2020]]
                    years = fold[fold_idx]
                    filename = f"{method}_{sample_size}_{region}_fold{fold_idx + 1}_{years[0]}-{years[1]}.png"
                    filepath = output_path / filename
                    
                    # 저장 (논문 게재 품질)
                    fig.savefig(
                        filepath, 
                        dpi=600,                    # 해상도 (300dpi는 대부분 저널 요구사항)
                        bbox_inches='tight',        # 여백 자동 제거
                        pad_inches=0.1,             # tight일 때 약간의 여백 유지 (가독성)
                        facecolor='white',          # 배경색 (흰색)
                        edgecolor='none',           # 테두리 없음
                        format='png',               # 파일 형식 (pdf로 변경 가능)
                        transparent= True           # 투명 배경 여부
                    )
                    plt.close(fig)
                    
                    print(f"  ✓ 저장 완료: {filename}")
                    
                except Exception as e:
                    print(f"  ✗ 오류 발생: {str(e)}")
                    continue
    
    # Method만 있고 sample_size가 없는 경우 (smote)
    for method, _ in methods_without_size:
        for region in regions:
            for fold_idx in fold_indices:
                current_plot += 1
                try:
                    print(f"[{current_plot}/{total_plots}] {method} - {region.upper()} - Fold {fold_idx + 1} 생성 중...")
                    
                    # 합성 데이터 경로 생성
                    synthetic_path = generate_synthetic_path(method, region, None, fold_idx, base_dir)
                    
                    # 데이터 로드 및 전처리
                    original_data, synthetic_data = load_and_preprocess_data(
                        synthetic_path=synthetic_path,
                        config=config,
                        region=region,
                        fold_idx=fold_idx,
                        data_dir=data_dir,
                        original_path=None
                    )
                    
                    # 피처 준비 및 스케일링
                    scaled_features, labels, scaler = prepare_features_for_visualization(
                        original_data, synthetic_data, config
                    )
                    
                    # UMAP 시각화
                    fig = plot_umap_comparison(
                        scaled_features, 
                        labels, 
                        config,
                        region=region,
                        fold_idx=fold_idx,
                        method=method,
                        sample_size=None  # smote는 sample_size가 없음
                    )
                    
                    # 파일명 생성: method_region_fold_years.png
                    fold = [[2018, 2019], [2018, 2020], [2019, 2020]]
                    years = fold[fold_idx]
                    filename = f"{method}_{region}_fold{fold_idx + 1}_{years[0]}-{years[1]}.png"
                    filepath = output_path / filename
                    
                    # 저장 (논문 게재 품질)
                    fig.savefig(
                        filepath, 
                        dpi=300,                    # 해상도 (300dpi는 대부분 저널 요구사항)
                        bbox_inches='tight',        # 여백 자동 제거
                        pad_inches=0.1,             # tight일 때 약간의 여백 유지 (가독성)
                        facecolor='white',          # 배경색 (흰색)
                        edgecolor='none',           # 테두리 없음
                        format='png',               # 파일 형식 (pdf로 변경 가능)
                        transparent=False           # 투명 배경 여부
                    )
                    plt.close(fig)
                    
                    print(f"  ✓ 저장 완료: {filename}")
                    
                except Exception as e:
                    print(f"  ✗ 오류 발생: {str(e)}")
                    continue
    
    print("=" * 60)
    print(f"모든 plot 생성 완료! 총 {current_plot}개 파일이 {output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    # 단일 plot 생성 (기본)
    # main()
    
    # 모든 조합의 plot 생성 (논문용)
    generate_all_plots(output_dir="images")
