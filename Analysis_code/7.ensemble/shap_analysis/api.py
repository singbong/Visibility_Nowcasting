"""SHAP 분석을 위한 공개 API 및 하위 호환성 래퍼 함수."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analyzers import DeepSHAPAnalyzer, TreeSHAPAnalyzer
from .calculators import (
    calculate_interclass_wd as _calculate_interclass_wd,
    calculate_uncertainty_entropy as _calculate_uncertainty_entropy,
    calculate_wbc_ratio as _calculate_wbc_ratio
)
from .config import SHAPConfig
from .data_loader import DataLoader

# model_utils lazy import를 위한 경로 설정
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ensemble_dir = os.path.dirname(_current_dir)
if _ensemble_dir not in sys.path:
    sys.path.insert(0, _ensemble_dir)


# ==========================================
# 메트릭 함수 (model_utils에서 re-export)
# ==========================================

def _get_metrics():
    """model_utils의 메트릭 함수들을 lazy import합니다."""
    from model_utils import calculate_csi, csi_metric, eval_metric_csi
    return calculate_csi, csi_metric, eval_metric_csi


def calculate_csi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """CSI(Critical Success Index) 점수를 계산합니다 (model_utils에서 re-export)."""
    calculate_csi_func, _, _ = _get_metrics()
    return calculate_csi_func(y_true, y_pred)


def csi_metric(y_true: np.ndarray, pred_prob: np.ndarray) -> Tuple[str, float, bool]:
    """LightGBM용 CSI 메트릭 함수 (model_utils에서 re-export)."""
    _, csi_metric_func, _ = _get_metrics()
    return csi_metric_func(y_true, pred_prob)


def eval_metric_csi(y_true: np.ndarray, pred_prob: np.ndarray) -> float:
    """XGBoost용 CSI 메트릭 함수 (model_utils에서 re-export)."""
    _, _, eval_metric_csi_func = _get_metrics()
    return eval_metric_csi_func(y_true, pred_prob)


# ==========================================
# SHAP 분석 래퍼 함수
# ==========================================

def analyze_shap_values_across_folds(
    model_name: str,
    region: str,
    data_sample: str,
    n_folds: int = 3,
    figsize: Tuple[int, int] = (20, 10),
    show_plot: bool = True,
    show_train_plot: bool = True,
    show_test_plot: bool = True,
    suppress_warnings: bool = True,
    top_n: int = 3
) -> Dict[str, Any]:
    """Tree 모델 SHAP 분석 (하위 호환성 래퍼 함수).
    
    Deprecated: Use TreeSHAPAnalyzer instead.
    """
    config = SHAPConfig(
        n_folds=n_folds,
        figsize=figsize,
        top_n=top_n
    )
    data_loader = DataLoader(config)
    analyzer = TreeSHAPAnalyzer(config, data_loader)
    result = analyzer.analyze(
        model_name, region, data_sample,
        show_plot=show_plot,
        show_train_plot=show_train_plot,
        show_test_plot=show_test_plot
    )
    return result.to_dict()


def analyze_dl_model_shap(
    model_name: str,
    region: str,
    data_sample: str,
    n_folds: int = 3,
    n_background_samples: int = 200,
    n_test_samples: int = 500,
    device: str = 'cpu',
    figsize: Tuple[int, int] = (20, 8),
    show_plot: bool = True,
    show_train_plot: bool = True,
    show_test_plot: bool = True,
    top_n: int = 3
) -> Optional[Dict[str, Any]]:
    """Deep Learning 모델 SHAP 분석 (하위 호환성 래퍼 함수).
    
    Deprecated: Use DeepSHAPAnalyzer instead.
    """
    config = SHAPConfig(
        n_folds=n_folds,
        n_background_samples=n_background_samples,
        n_test_samples=n_test_samples,
        figsize=figsize,
        top_n=top_n,
        device=device
    )
    data_loader = DataLoader(config)
    analyzer = DeepSHAPAnalyzer(config, data_loader, device=device)
    result = analyzer.analyze(
        model_name, region, data_sample,
        show_plot=show_plot,
        show_train_plot=show_train_plot,
        show_test_plot=show_test_plot
    )
    return result.to_dict() if result else None


def calculate_interclass_wd(
    region: str,
    shap_result: Dict[str, Any],
    top_n: int = 5
) -> Dict[str, Any]:
    """Train C1-C2 vs Train C1-Test C2 간의 Wasserstein Distance 분석 (공개 API).
    
    각 feature별 WD를 계산한 후 단순 평균하여 클래스 간 분리도 변화를 측정합니다.
    
    Args:
        region: 지역명 (예: 'seoul', 'busan')
        shap_result: analyze_shap_values_across_folds() 또는 analyze_dl_model_shap()의 반환값
        top_n: 상위 몇 개 feature 사용 (기본값: 5)
    
    Returns:
        결과 딕셔너리:
        - wd_base: Train C1-C2 간 평균 WD
        - wd_test: Train C1-Test C2 간 평균 WD
        - wd_change_pct: 거리 변화율 (%)
        - wd_base_per_feature: 각 feature별 Train C1-C2 WD
        - wd_test_per_feature: 각 feature별 Train C1-Test C2 WD
        - shares: 각 feature별 SHAP share (비율, 참고용)
        - sample_counts: 각 그룹별 샘플 수
    
    Note:
        - Train C1: Pure 데이터의 train 중 class 1 (2018-2020)
        - Train C2: Pure 데이터의 train 중 class 2 (2018-2020)
        - Test C2: Test 데이터 중 class 2 (2021)
        - StandardScaler는 Train C1+C2 결합 데이터로 fit (클래스 간 공정한 비교)
        - WD는 각 feature별 거리의 단순 평균으로 계산 (SHAP 가중치 미사용)
    """
    return _calculate_interclass_wd(region, shap_result, top_n)


# ==========================================
# 엔트로피 계산 래퍼 함수
# ==========================================

def calculate_uncertainty_entropy(
    model_configs: List[Dict[str, Any]],
    region: str,
    device: str = 'cpu',
    n_folds: int = 3,
    target_class: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """2단계 소프트보팅 후 엔트로피를 측정하는 함수 (공개 API).
    
    Baseline (OOF) 엔트로피와 Test 엔트로피를 계산합니다.
    
    Args:
        model_configs: 모델 설정 리스트, 각 딕셔너리는 다음 키를 포함:
            - 'model_name': 모델 이름 (예: 'xgb', 'ft_transformer')
            - 'data_sample': 데이터 샘플 타입 (예: 'pure', 'smote', 'ctgan10000')
        region: 지역명 (예: 'seoul', 'busan')
        device: 연산 장치 ('cpu' 또는 'cuda', 기본값: 'cpu')
        n_folds: Fold 개수 (기본값: 3)
        target_class: 엔트로피를 계산할 클래스 (기본값: 2)
    
    Returns:
        tuple: (baseline_entropies, test_entropies)
            - baseline_entropies: Baseline (OOF) 엔트로피 배열
            - test_entropies: Test 엔트로피 배열
    
    Example:
        >>> model_configs = [
        ...     {'model_name': 'ft_transformer', 'data_sample': 'ctgan10000'},
        ...     {'model_name': 'xgb', 'data_sample': 'smote'}
        ... ]
        >>> baseline_entropies, test_entropies = calculate_uncertainty_entropy(
        ...     model_configs=model_configs,
        ...     region='seoul',
        ...     device='cuda',
        ...     n_folds=3,
        ...     target_class=2
        ... )
    
    Note:
        - Baseline 엔트로피: 각 fold의 검증 데이터에서 target_class만 추출하여 계산 (OOF 방식)
        - Test 엔트로피: 2단계 소프트보팅 후 test 데이터에서 target_class만 추출하여 계산
        - 2단계 소프트보팅:
          1단계: 각 fold 내에서 여러 모델의 확률을 평균
          2단계: 3개 fold의 앙상블 모델 확률을 평균
    """
    return _calculate_uncertainty_entropy(
        model_configs=model_configs,
        region=region,
        device=device,
        n_folds=n_folds,
        target_class=target_class
    )


def calculate_wbc_ratio(
    model_configs: List[Dict[str, Any]],
    regions: List[str],
    device: str = 'cpu',
    n_folds: int = 3,
    target_classes: List[int] = None,
    threshold_method: str = 'mean'
) -> pd.DataFrame:
    """Wrong-but-Confident (WBC) 비율을 지역별로 계산하는 함수 (공개 API).
    
    각 지역별로 Test 데이터에서 target_classes 샘플 중 엔트로피가 낮은(확신이 높은) 
    상태에서 틀린 샘플의 비율을 계산합니다.
    
    Args:
        model_configs: 모델 설정 리스트, 각 딕셔너리는 다음 키를 포함:
            - 'model_name': 모델 이름 (예: 'xgb', 'ft_transformer')
            - 'data_sample': 데이터 샘플 타입 (예: 'pure', 'smote', 'ctgan10000')
        regions: 지역명 리스트 (예: ['seoul', 'busan'])
        device: 연산 장치 ('cpu' 또는 'cuda', 기본값: 'cpu')
        n_folds: Fold 개수 (기본값: 3)
        target_classes: 분석할 클래스 리스트 (기본값: [2], 단일 int 값도 받을 수 있음)
        threshold_method: Confidence threshold 계산 방법 (기본값: 'mean')
            - 'mean': Baseline 엔트로피 평균 사용
            - 'median': Baseline 엔트로피 중앙값 사용
            - 'q25': Baseline 엔트로피 25th percentile 사용
    
    Returns:
        pd.DataFrame: 지역별 통계를 담은 데이터프레임
            - region: 지역명
            - total_samples: 전체 target_classes 샘플 수
            - confident_samples: 확신이 높은 샘플 수 (엔트로피 < threshold)
            - wrong_samples: 틀린 샘플 수
            - wbc_samples: Wrong-but-Confident 샘플 수
            - wbc_ratio_in_confident: 확신이 높은 샘플 중 WBC 비율 (%)
            - wbc_ratio_in_errors: 틀린 샘플 중 WBC 비율 (%)
            - baseline_entropy_mean: Baseline 엔트로피 평균
            - mean_entropy_test: 전체 test 샘플의 엔트로피 평균
            - entropy_diff: 엔트로피 변화량 (baseline - test)
            - mean_entropy_confident: 확신이 높은 샘플의 평균 엔트로피
            - mean_entropy_wrong: 틀린 샘플의 평균 엔트로피
            - threshold_method: 사용된 threshold 계산 방법
            - threshold_value: 실제 사용된 threshold 값
    
    Note:
        - 확신이 높은 샘플: Baseline 엔트로피가 threshold보다 낮은 샘플
        - WBC 샘플: 확신이 높으면서 틀린 샘플
        - 2단계 소프트보팅:
          1단계: 각 fold 내에서 여러 모델의 확률을 평균
          2단계: 3개 fold의 앙상블 모델 확률을 평균
    """
    return _calculate_wbc_ratio(
        model_configs=model_configs,
        regions=regions,
        device=device,
        n_folds=n_folds,
        target_classes=target_classes,
        threshold_method=threshold_method
    )


def calculate_wbc_ratio_with_correction(
    model_configs: List[Dict[str, Any]],
    regions: List[str],
    device: str = 'cpu',
    n_folds: int = 3,
    target_classes: List[int] = None,
    threshold_method: str = 'mean',
    alpha: float = None
) -> pd.DataFrame:
    """Multiple testing correction을 적용한 WBC 비율 계산 함수 (공개 API).
    
    여러 지역에 대해 동시에 테스트할 때 발생하는 Type I error 증가를 
    Benjamini-Hochberg FDR correction으로 보정합니다.
    
    Args:
        model_configs: 모델 설정 리스트
        regions: 지역명 리스트
        device: 연산 장치 (기본값: 'cpu')
        n_folds: Fold 개수 (기본값: 3)
        target_classes: 분석할 클래스 리스트 (기본값: [2])
        threshold_method: Confidence threshold 계산 방법 (기본값: 'mean')
        alpha: 유의 수준 (기본값: 0.05)
    
    Returns:
        pd.DataFrame: calculate_wbc_ratio 결과에 FDR correction 정보 추가
            - 기존 컬럼들
            - p_value: 귀무가설 검정 p-value
            - p_adjusted: FDR 보정된 p-value
            - significant: 유의성 여부
    
    Note:
        - FDR correction은 Benjamini-Hochberg 방법 사용
        - 실제 p-value 계산은 연구 설계에 따라 구현 필요
    """
    from .calculators import calculate_wbc_ratio_with_correction as _calculate_wbc_ratio_with_correction
    
    return _calculate_wbc_ratio_with_correction(
        model_configs=model_configs,
        regions=regions,
        device=device,
        n_folds=n_folds,
        target_classes=target_classes,
        threshold_method=threshold_method,
        alpha=alpha
    )
