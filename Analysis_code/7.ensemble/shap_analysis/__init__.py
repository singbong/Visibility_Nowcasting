"""SHAP 분석 패키지.

이 패키지는 SHAP 분석을 위한 모든 기능을 제공합니다.
"""

# Config
from .config import (
    SHAPConfig,
    FOLD_YEARS,
    TEST_YEAR,
    DROP_COLUMNS,
    FEATURE_COLUMNS
)

# Data classes
from .analyzers import ImportanceData, SHAPResult

# Analyzers
from .analyzers import (
    SHAPAnalyzer,
    TreeSHAPAnalyzer,
    DeepSHAPAnalyzer
)

# Data loader
from .data_loader import DataLoader

# Preprocessors
from .preprocessors import (
    PreprocessingUtils,
    ShapValueExtractor,
    ModelWrapper,
    suppress_shap_output
)

# Visualizers
from .visualizers import SHAPVisualizer

# Public API (하위 호환성)
from .api import (
    # SHAP 분석
    analyze_shap_values_across_folds,
    analyze_dl_model_shap,
    
    # Wasserstein Distance
    calculate_interclass_wd,
    
    # 엔트로피 계산
    calculate_uncertainty_entropy,
    calculate_wbc_ratio,
    calculate_wbc_ratio_with_correction,
    
    # 메트릭
    calculate_csi,
    csi_metric,
    eval_metric_csi
)

__all__ = [
    # Config
    'SHAPConfig',
    'FOLD_YEARS',
    'TEST_YEAR',
    'DROP_COLUMNS',
    'FEATURE_COLUMNS',
    
    # Data classes
    'ImportanceData',
    'SHAPResult',
    
    # Analyzers
    'SHAPAnalyzer',
    'TreeSHAPAnalyzer',
    'DeepSHAPAnalyzer',
    
    # Data loader
    'DataLoader',
    
    # Preprocessors
    'PreprocessingUtils',
    'ShapValueExtractor',
    'ModelWrapper',
    'suppress_shap_output',
    
    # Visualizers
    'SHAPVisualizer',
    
    # Public API
    'analyze_shap_values_across_folds',
    'analyze_dl_model_shap',
    'calculate_interclass_wd',
    'calculate_uncertainty_entropy',
    'calculate_wbc_ratio',
    'calculate_wbc_ratio_with_correction',
    'calculate_csi',
    'csi_metric',
    'eval_metric_csi',
]
