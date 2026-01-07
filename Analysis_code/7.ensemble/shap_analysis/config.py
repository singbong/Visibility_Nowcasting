"""SHAP 분석 설정 및 상수 정의."""

from typing import Tuple


class SHAPConfig:
    """SHAP 분석 설정을 관리하는 클래스."""
    
    # 상수 정의
    FOLD_YEARS = [[2018, 2019], [2018, 2019], [2019, 2020]]
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
    
    # 기본 설정값
    DEFAULT_N_FOLDS = 3
    DEFAULT_N_BACKGROUND_SAMPLES = 1000  # 통계적 신뢰성을 위해 200 → 1000으로 증가
    DEFAULT_N_TEST_SAMPLES = 2000         # 통계적 신뢰성을 위해 500 → 2000으로 증가
    DEFAULT_FIGSIZE = (20, 8)
    DEFAULT_TOP_N = 3
    DEFAULT_DEVICE = 'cpu'
    DEFAULT_PLOT_MARGIN = 0.05
    DEFAULT_RANDOM_SEED = 42
    
    # Seed offset 상수 (재현성을 위한 fold별 seed 분리)
    SEED_OFFSET_TRAIN = 1000
    SEED_OFFSET_TEST = 2000
    
    # 데이터 전처리 상수
    CALM_WIND_DIR = '정온'
    
    # 수치 계산 상수
    EPSILON = 1e-10
    
    # Bootstrap 및 통계 검정 상수
    N_BOOTSTRAP = 1000  # Bootstrap 반복 횟수
    CONFIDENCE_LEVEL = 0.95  # 신뢰 수준 (95%)
    
    # WBC Threshold 옵션
    WBC_THRESHOLD_METHODS = ['mean', 'median', 'q25']
    DEFAULT_WBC_THRESHOLD_METHOD = 'mean'
    
    # Multiple testing correction
    ALPHA = 0.05  # 유의 수준
    FDR_METHOD = 'fdr_bh'  # Benjamini-Hochberg FDR correction
    
    def __init__(
        self,
        n_folds: int = DEFAULT_N_FOLDS,
        n_background_samples: int = DEFAULT_N_BACKGROUND_SAMPLES,
        n_test_samples: int = DEFAULT_N_TEST_SAMPLES,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        top_n: int = DEFAULT_TOP_N,
        device: str = DEFAULT_DEVICE,
        random_seed: int = DEFAULT_RANDOM_SEED
    ):
        self.n_folds = n_folds
        self.n_background_samples = n_background_samples
        self.n_test_samples = n_test_samples
        self.figsize = figsize
        self.top_n = top_n
        self.device = device
        self.random_seed = random_seed
        self._validate()
    
    def _validate(self):
        """설정값 검증."""
        if self.n_folds <= 0:
            raise ValueError("n_folds must be positive")
        if self.n_background_samples <= 0:
            raise ValueError("n_background_samples must be positive")
        if self.n_test_samples <= 0:
            raise ValueError("n_test_samples must be positive")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")


# 기존 상수들 (하위 호환성)
FOLD_YEARS = SHAPConfig.FOLD_YEARS
TEST_YEAR = SHAPConfig.TEST_YEAR
DROP_COLUMNS = SHAPConfig.DROP_COLUMNS
FEATURE_COLUMNS = SHAPConfig.FEATURE_COLUMNS
