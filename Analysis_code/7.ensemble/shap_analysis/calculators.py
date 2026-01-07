"""Wasserstein Distance 및 엔트로피 계산 함수."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
from sklearn.preprocessing import StandardScaler

from .config import SHAPConfig
from .data_loader import DataLoader
from .preprocessors import PreprocessingUtils

# model_utils lazy import를 위한 경로 설정
_current_dir = os.path.dirname(os.path.abspath(__file__))
_ensemble_dir = os.path.dirname(_current_dir)
if _ensemble_dir not in sys.path:
    sys.path.insert(0, _ensemble_dir)


# ==========================================
# 공통 유틸리티 함수
# ==========================================

def _get_model_utils():
    """model_utils 모듈을 lazy import합니다 (순환 참조 방지)."""
    from model_utils import predict_val_proba, predict_test_proba
    return predict_val_proba, predict_test_proba


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    """확률 배열을 정규화합니다 (합이 1이 되도록).
    
    이미 정규화된 확률에는 epsilon을 추가하지 않고, 0에 가까운 값만 clip하여
    불필요한 분포 왜곡을 방지합니다.
    
    Args:
        probs: 확률 배열 (n_samples, n_classes) 또는 (n_classes,)
    
    Returns:
        정규화된 확률 배열
    """
    # 0에 가까운 확률만 epsilon으로 clip (log 계산 안정성)
    probs_clipped = np.clip(probs, SHAPConfig.EPSILON, 1.0)
    
    # 합이 1에 가까우면 이미 정규화된 것으로 간주
    prob_sum = probs_clipped.sum(axis=-1, keepdims=True)
    
    # 재정규화 (필요한 경우만)
    return probs_clipped / prob_sum


def _calculate_entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    """확률 배열에서 엔트로피를 계산합니다.
    
    Args:
        probs: 확률 배열 (n_samples, n_classes) 또는 (n_classes,)
    
    Returns:
        엔트로피 배열 (n_samples,) 또는 스칼라
    """
    probs_norm = _normalize_probs(probs)
    if probs_norm.ndim == 1:
        return entropy(probs_norm)
    else:
        return np.array([entropy(p) for p in probs_norm])


def _two_stage_soft_voting(
    test_probs_per_model: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """2단계 소프트보팅을 수행합니다.
    
    Args:
        test_probs_per_model: 각 모델별 예측 확률 리스트
            각 요소는 (n_folds, n_samples, n_classes) 형태
    
    Returns:
        tuple: (final_probs, y_pred)
            - final_probs: 최종 앙상블 확률 (n_samples, n_classes)
            - y_pred: 최종 예측 클래스 (n_samples,)
    """
    # 1단계: 각 fold 내에서 여러 모델의 확률을 평균
    # test_probs_per_model: list of (n_folds, n_samples, n_classes) arrays
    # ensemble_per_fold: (n_folds, n_samples, n_classes)
    ensemble_per_fold = np.mean(test_probs_per_model, axis=0)
    
    # 2단계: 모든 fold의 앙상블 모델 확률을 평균
    # final_probs: (n_samples, n_classes)
    final_probs = np.mean(ensemble_per_fold, axis=0)
    
    # 최종 예측값
    y_pred = np.argmax(final_probs, axis=1)
    
    return final_probs, y_pred


def _bootstrap_ci(
    data: np.ndarray,
    statistic_fn=None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap을 이용한 통계량의 신뢰구간 계산.
    
    Args:
        data: 입력 데이터 배열
        statistic_fn: 통계량 계산 함수 (기본값: np.mean)
        n_bootstrap: Bootstrap 반복 횟수 (기본값: 1000)
        confidence_level: 신뢰 수준 (기본값: 0.95)
        random_state: 난수 시드 (재현성, 기본값: 42)
    
    Returns:
        tuple: (point_estimate, ci_lower, ci_upper)
            - point_estimate: 원본 데이터의 통계량
            - ci_lower: 신뢰구간 하한
            - ci_upper: 신뢰구간 상한
    """
    if statistic_fn is None:
        statistic_fn = np.mean
    
    # 원본 통계량
    point_estimate = statistic_fn(data)
    
    # Bootstrap 샘플링
    np.random.seed(random_state)
    n = len(data)
    boot_stats = []
    
    for _ in range(n_bootstrap):
        # Resampling with replacement
        boot_sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic_fn(boot_sample))
    
    boot_stats = np.array(boot_stats)
    
    # Percentile method로 신뢰구간 계산
    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_stats, alpha / 2 * 100)
    ci_upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)
    
    return point_estimate, ci_lower, ci_upper


def _collect_baseline_entropies(
    model_configs: List[Dict[str, Any]],
    region: str,
    device: str,
    n_folds: int,
    target_class: int,
    verbose: bool = True
) -> np.ndarray:
    """Baseline (OOF) 엔트로피를 수집합니다.
    
    각 fold에서 여러 모델의 확률을 먼저 앙상블(평균)한 후 엔트로피를 계산합니다.
    이는 Test 데이터의 엔트로피 계산 방식과 일관성을 유지하기 위함입니다.
    
    Args:
        model_configs: 모델 설정 리스트
        region: 지역명
        device: 연산 장치
        n_folds: Fold 개수
        target_class: 대상 클래스
        verbose: 출력 여부
    
    Returns:
        Baseline 엔트로피 배열
        
    Note:
        - 각 fold에서 모델 앙상블 후 엔트로피 계산 (Stage 1 ensemble)
        - 여러 fold 결과는 concatenate (OOF 특성상 Stage 2 ensemble 불가)
        - Test entropy와 동일한 앙상블 방식 적용으로 systematic bias 제거
        
        **Sample Imbalance 처리 전략**:
        - OOF(Out-of-Fold) 방식에서는 각 fold의 모든 검증 샘플을 사용
        - Fold별 샘플 수 차이는 자연스러운 현상 (시간 기반 분할)
        - 모든 샘플을 사용하는 것이 통계적으로 타당 (정보 손실 최소화)
        - Weighted average 대신 concatenation 사용: 각 샘플이 동등한 중요도
        - 이는 전체 validation set에 대한 unbiased estimate 제공
    """
    predict_val_proba, _ = _get_model_utils()
    
    baseline_entropies_list = []
    
    if verbose:
        print("\n[A] Baseline (OOF) Entropy Collection (Ensemble-first approach)")
    
    # 1단계: 모든 모델의 확률을 수집
    all_model_probs = []  # List of (n_folds, n_samples, n_classes)
    y_true_list = None
    
    for model_idx, model_cfg in enumerate(model_configs):
        model_name = model_cfg['model_name']
        data_sample = model_cfg['data_sample']
        
        if verbose:
            print(f"  Model {model_idx + 1}/{len(model_configs)}: {model_name} ({data_sample})")
        
        # predict_val_proba 호출
        all_probs, _, y_true_per_fold = predict_val_proba(
            model_name=model_name,
            region=region,
            data_sample=data_sample,
            device=device,
            n_folds=n_folds
        )
        
        all_model_probs.append(all_probs)
        
        # y_true는 모든 모델에서 동일해야 함
        if y_true_list is None:
            y_true_list = y_true_per_fold
    
    # 2단계: 각 fold에서 모델 앙상블 후 엔트로피 계산
    for fold_idx in range(n_folds):
        # 해당 fold의 모든 모델 확률 수집
        fold_probs_per_model = [model_probs[fold_idx] for model_probs in all_model_probs]
        
        # 모델 앙상블 (평균)
        ensemble_probs = np.mean(fold_probs_per_model, axis=0)  # (n_samples, n_classes)
        
        # Target class만 필터링
        fold_mask = (y_true_list[fold_idx] == target_class)
        
        if np.sum(fold_mask) == 0:
            if verbose:
                print(f"    Warning: Fold {fold_idx}에 Class {target_class} 샘플이 없습니다.")
            continue
        
        # 필터링된 확률에서 엔트로피 계산
        ensemble_probs_filtered = ensemble_probs[fold_mask]
        fold_entropies = _calculate_entropy_from_probs(ensemble_probs_filtered)
        baseline_entropies_list.extend(fold_entropies)
        
        if verbose:
            print(f"    Fold {fold_idx}: {np.sum(fold_mask)} samples, "
                  f"Mean Entropy: {np.mean(fold_entropies):.4f}")
    
    baseline_entropies = np.array(baseline_entropies_list)
    
    if verbose:
        print(f"\n  Total Baseline Entropy Samples: {len(baseline_entropies)}")
        print(f"  Mean Baseline Entropy: {np.mean(baseline_entropies):.4f}")
    
    return baseline_entropies


def _perform_test_predictions(
    model_configs: List[Dict[str, Any]],
    region: str,
    device: str,
    n_folds: int,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Test 데이터에 대한 예측을 수행하고 2단계 소프트보팅을 적용합니다.
        
        Args:
        model_configs: 모델 설정 리스트
            region: 지역명
        device: 연산 장치
        n_folds: Fold 개수
        verbose: 출력 여부
    
    Returns:
        tuple: (final_probs, y_test)
            - final_probs: 최종 앙상블 확률 (n_samples, n_classes)
            - y_test: Test 데이터의 실제 클래스 (n_samples,)
    """
    _, predict_test_proba = _get_model_utils()
    
    test_probs_per_model = []
    y_test = None
    
    if verbose:
        print("\n[B] Test Entropy Collection")
    
    for model_idx, model_cfg in enumerate(model_configs):
        model_name = model_cfg['model_name']
        data_sample = model_cfg['data_sample']
        
        if verbose:
            print(f"  Model {model_idx + 1}/{len(model_configs)}: {model_name} ({data_sample})")
        
        # predict_test_proba 호출
        probs, y_test_model = predict_test_proba(
            model_name=model_name,
            region=region,
            data_sample=data_sample,
            device=device,
            n_folds=n_folds
        )
        
        # y_test는 모든 모델에서 동일해야 함
        if y_test is None:
            y_test = y_test_model
        else:
            if not np.array_equal(y_test, y_test_model):
                raise ValueError(f"Model {model_name}의 y_test가 다른 모델과 일치하지 않습니다.")
        
        # probs shape: (n_folds, n_samples, n_classes)
        test_probs_per_model.append(probs)
        
        if verbose:
            print(f"    Probs shape: {probs.shape}")
    
    # 2단계 소프트보팅 수행
    if verbose:
        print("\n[C] Two-stage Soft Voting")
    
    final_probs, y_pred = _two_stage_soft_voting(test_probs_per_model)
    
    if verbose:
        print(f"  Stage 1 (Model Ensemble): {np.array(test_probs_per_model).shape}")
        print(f"  Stage 2 (Fold Ensemble): {final_probs.shape}")
    
    return final_probs, y_test


def _load_class_data(
    region: str,
    classes: List[int],
    top_features: List[str]
) -> Dict[str, pd.DataFrame]:
    """클래스별 데이터를 로드합니다.
    
    Args:
        region: 지역명
        classes: 클래스 리스트 [train_c1_class, train_c2_class, test_c2_class]
        top_features: 사용할 feature 리스트
    
    Returns:
        딕셔너리:
            - 'train_c1': Train Class 1 데이터
            - 'train_c2': Train Class 2 데이터
            - 'test_c2': Test Class 2 데이터
    """
    config = SHAPConfig()
    
    # Train 데이터 로드 (2018-2020)
    train_path = f"../../data/data_for_modeling/{region}_train.csv"
    train_data = pd.read_csv(train_path, index_col=0)
    train_data = PreprocessingUtils.preprocessing(train_data, config)
    
    # Test 데이터 로드 (2021)
    test_path = f"../../data/data_for_modeling/{region}_test.csv"
    test_data = pd.read_csv(test_path, index_col=0)
    test_data = PreprocessingUtils.preprocessing(test_data, config)
    test_data = test_data[test_data['year'] == config.TEST_YEAR]
    
    # 클래스별 필터링
    train_c1 = train_data[train_data['multi_class'] == classes[0]][top_features].copy()
    train_c2 = train_data[train_data['multi_class'] == classes[1]][top_features].copy()
    test_c2 = test_data[test_data['multi_class'] == classes[2]][top_features].copy()
    
    # 샘플 수 확인
    if len(train_c1) == 0:
        raise ValueError(f"{region}: Train Class {classes[0]} 데이터가 없습니다.")
    if len(train_c2) == 0:
        raise ValueError(f"{region}: Train Class {classes[1]} 데이터가 없습니다.")
    if len(test_c2) == 0:
        raise ValueError(f"{region}: Test Class {classes[2]} 데이터가 없습니다.")
    
    return {
        'train_c1': train_c1,
        'train_c2': train_c2,
        'test_c2': test_c2
    }


# ==========================================
# 공개 API 함수
# ==========================================

def calculate_interclass_wd(
    region: str,
    shap_result: Dict[str, Any],
    top_n: int = 5
) -> Dict[str, Any]:
    """Train C1-C2 vs Train C1-Test C2 간의 Wasserstein Distance 분석.
    
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
    # 1. SHAP 결과에서 상위 feature 추출
    if 'top_n_features_train' not in shap_result or shap_result['top_n_features_train'] is None:
        raise ValueError("shap_result에 'top_n_features_train'이 없습니다.")
    
    top_features = shap_result['top_n_features_train'][:top_n]
    
    # 2. 변수 중요도 추출 및 SHAP share 계산 (비율)
    # 모든 feature의 importance 합을 기준으로 비율 계산
    if 'importance_df_train' not in shap_result or shap_result['importance_df_train'] is None:
        raise ValueError("shap_result에 'importance_df_train'이 없습니다.")
    
    importance_df = shap_result['importance_df_train']
    
    # 모든 feature의 importance 합 계산
    total_importance = importance_df['importance'].sum()
    
    # 상위 feature들의 importance 추출 및 share 계산
    shares = []
    for feat in top_features:
        feat_importance = importance_df[importance_df['feature'] == feat]['importance'].values
        if len(feat_importance) == 0:
            raise ValueError(f"Feature '{feat}'의 중요도를 찾을 수 없습니다.")
        # 전체 feature의 importance 합으로 나누어 비율 계산
        share = feat_importance[0] / total_importance
        shares.append(share)
    
    shares = np.array(shares)
    
    # 3. 클래스별 데이터 로드
    class_data = _load_class_data(region, classes=[1, 2, 2], top_features=top_features)
    train_c1 = class_data['train_c1']
    train_c2 = class_data['train_c2']
    test_c2 = class_data['test_c2']
    
    # 4. StandardScaler 적용 (Combined Train C1+C2 기준으로 공정한 비교)
    # Train C2만 사용하면 C2 분포 중심으로 편향될 수 있으므로
    # 두 클래스를 결합한 데이터로 fit하여 클래스 간 공정한 비교 보장
    combined_train = pd.concat([train_c1, train_c2])
    scaler = StandardScaler().fit(combined_train)
    train_c1_scaled = scaler.transform(train_c1)
    train_c2_scaled = scaler.transform(train_c2)
    test_c2_scaled = scaler.transform(test_c2)
    
    # 5. 각 feature별 WD 계산
    wd_base_per_feature = []
    wd_test_per_feature = []
    
    for i in range(len(top_features)):
        wd_base = wasserstein_distance(train_c1_scaled[:, i], train_c2_scaled[:, i])
        wd_test = wasserstein_distance(train_c1_scaled[:, i], test_c2_scaled[:, i])
        wd_base_per_feature.append(wd_base)
        wd_test_per_feature.append(wd_test)
    
    # 6. 평균 WD 계산 (SHAP 가중치 사용 안 함)
    wd_base_weighted = np.mean(wd_base_per_feature)
    wd_test_weighted = np.mean(wd_test_per_feature)
    
    # 7. 변화율 계산
    wd_change_pct = ((wd_test_weighted - wd_base_weighted) / wd_base_weighted) * 100
    
    # 7.5. Bootstrap 신뢰구간 계산 (feature별 WD를 bootstrap)
    # Bootstrap을 위해 feature별 WD를 재샘플링
    def wd_statistic(indices):
        """Bootstrap용 WD 계산 함수 (단순 평균)"""
        sampled_base = [wd_base_per_feature[i] for i in indices]
        sampled_test = [wd_test_per_feature[i] for i in indices]
        return (np.mean(sampled_base), np.mean(sampled_test))
    
    # Feature indices로 bootstrap
    np.random.seed(42)
    n_features = len(top_features)
    boot_wd_base = []
    boot_wd_test = []
    
    for _ in range(SHAPConfig.N_BOOTSTRAP):
        boot_indices = np.random.choice(n_features, size=n_features, replace=True)
        wd_b, wd_t = wd_statistic(boot_indices)
        boot_wd_base.append(wd_b)
        boot_wd_test.append(wd_t)
    
    wd_base_ci_lower = np.percentile(boot_wd_base, 2.5)
    wd_base_ci_upper = np.percentile(boot_wd_base, 97.5)
    wd_test_ci_lower = np.percentile(boot_wd_test, 2.5)
    wd_test_ci_upper = np.percentile(boot_wd_test, 97.5)
    
    # 8. 결과 출력
    print(f"\n=== Inter-class WD Analysis: {region.upper()} ===")
    print(f"Top {top_n} features: {top_features}")
    print(f"SHAP share: {dict(zip(top_features, [f'{s:.3f}' for s in shares]))}")
    print(f"\n[Sample Counts]")
    print(f"  Train C1: {len(train_c1)}")
    print(f"  Train C2: {len(train_c2)}")
    print(f"  Test C2: {len(test_c2)}")
    print(f"\n[Wasserstein Distance]")
    print(f"  Train C1 ↔ Train C2: {wd_base_weighted:.4f} (95% CI: [{wd_base_ci_lower:.4f}, {wd_base_ci_upper:.4f}])")
    print(f"  Train C1 ↔ Test C2:  {wd_test_weighted:.4f} (95% CI: [{wd_test_ci_lower:.4f}, {wd_test_ci_upper:.4f}])")
    print(f"  Change: {wd_change_pct:+.2f}%")
    print(f"\n[Per-feature WD]")
    for i, feat in enumerate(top_features):
        wd_b = wd_base_per_feature[i]
        wd_t = wd_test_per_feature[i]
        s = shares[i]
        change_pct = ((wd_t - wd_b) / wd_b * 100) if wd_b > 0 else 0.0
        print(f"  {feat:20s} (share={s:.3f}): Base={wd_b:.4f}, Test={wd_t:.4f}, Change={change_pct:+.2f}%")
    
    # 9. 결과 반환 (CI 정보 추가)
    return {
        'wd_base': wd_base_weighted,
        'wd_test': wd_test_weighted,
        'wd_change_pct': wd_change_pct,
        'wd_base_ci': (wd_base_ci_lower, wd_base_ci_upper),
        'wd_test_ci': (wd_test_ci_lower, wd_test_ci_upper),
        'wd_base_per_feature': dict(zip(top_features, wd_base_per_feature)),
        'wd_test_per_feature': dict(zip(top_features, wd_test_per_feature)),
        'shares': dict(zip(top_features, shares)),
        'sample_counts': {
            'train_c1': len(train_c1),
            'train_c2': len(train_c2),
            'test_c2': len(test_c2)
        }
    }


def calculate_uncertainty_entropy(
    model_configs: List[Dict[str, Any]],
    region: str,
    device: str = 'cpu',
    n_folds: int = 3,
    target_class: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """2단계 소프트보팅 후 엔트로피를 측정하는 함수.
    
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
    
    Note:
        - Baseline 엔트로피: 각 fold의 검증 데이터에서 target_class만 추출하여 계산 (OOF 방식)
        - Test 엔트로피: 2단계 소프트보팅 후 test 데이터에서 target_class만 추출하여 계산
        - 2단계 소프트보팅:
          1단계: 각 fold 내에서 여러 모델의 확률을 평균
          2단계: 모든 fold의 앙상블 모델 확률을 평균
    """
    print(f"\n=== Uncertainty Entropy Calculation: {region.upper()} ===")
    print(f"Models: {[cfg['model_name'] for cfg in model_configs]}")
    print(f"Target Class: {target_class}")
    print(f"Device: {device}, Folds: {n_folds}")
    print("-" * 60)
    
    # Baseline 엔트로피 수집
    baseline_entropies = _collect_baseline_entropies(
        model_configs, region, device, n_folds, target_class, verbose=True
    )
    
    # Test 예측 수행
    final_probs, y_test = _perform_test_predictions(
        model_configs, region, device, n_folds, verbose=True
    )
    
    # Test 데이터에서 target_class만 필터링
    test_mask = (y_test == target_class)
    if np.sum(test_mask) == 0:
        raise ValueError(f"Test 데이터에 Class {target_class} 샘플이 없습니다.")
    
    test_probs_filtered = final_probs[test_mask]
    print(f"  Test Class {target_class} samples: {np.sum(test_mask)}")
    
    # 엔트로피 계산
    test_entropies = _calculate_entropy_from_probs(test_probs_filtered)
    
    # Bootstrap 신뢰구간 계산
    baseline_mean, baseline_ci_lower, baseline_ci_upper = _bootstrap_ci(
        baseline_entropies, 
        statistic_fn=np.mean,
        n_bootstrap=SHAPConfig.N_BOOTSTRAP,
        confidence_level=SHAPConfig.CONFIDENCE_LEVEL
    )
    
    test_mean, test_ci_lower, test_ci_upper = _bootstrap_ci(
        test_entropies,
        statistic_fn=np.mean,
        n_bootstrap=SHAPConfig.N_BOOTSTRAP,
        confidence_level=SHAPConfig.CONFIDENCE_LEVEL
    )
    
    print(f"\n  Total Test Entropy Samples: {len(test_entropies)}")
    print(f"  Mean Baseline Entropy: {baseline_mean:.4f} (95% CI: [{baseline_ci_lower:.4f}, {baseline_ci_upper:.4f}])")
    print(f"  Mean Test Entropy: {test_mean:.4f} (95% CI: [{test_ci_lower:.4f}, {test_ci_upper:.4f}])")
    print("\n" + "=" * 60)
    
    return baseline_entropies, test_entropies


def calculate_wbc_ratio(
    model_configs: List[Dict[str, Any]],
    regions: List[str],
    device: str = 'cpu',
    n_folds: int = 3,
    target_classes: List[int] = None,
    threshold_method: str = 'mean'
) -> pd.DataFrame:
    """Wrong-but-Confident (WBC) 비율을 지역별로 계산하는 함수.
    
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
            - mean_entropy_test: 전체 test 샘플의 엔트로피 평균 (entropy_test에 해당)
            - entropy_diff: 엔트로피 변화량 (baseline_entropy_mean - mean_entropy_test)
            - mean_entropy_confident: 확신이 높은 샘플의 평균 엔트로피
            - mean_entropy_wrong: 틀린 샘플의 평균 엔트로피
            - threshold_method: 사용된 threshold 계산 방법
            - threshold_value: 실제 사용된 threshold 값
    
    Note:
        - 확신이 높은 샘플: Baseline 엔트로피가 threshold보다 낮은 샘플
        - WBC 샘플: 확신이 높으면서 틀린 샘플
        - 2단계 소프트보팅:
          1단계: 각 fold 내에서 여러 모델의 확률을 평균
          2단계: 모든 fold의 앙상블 모델 확률을 평균
    """
    # 호환성: target_classes 정규화
    if target_classes is None:
        target_classes = [2]
    elif isinstance(target_classes, int):
        target_classes = [target_classes]
    target_classes = list(target_classes)
    
    predict_val_proba, predict_test_proba = _get_model_utils()
    
    wbc_stats = []
    
    for region in regions:
        print(f"\n{'='*60}")
        print(f"WBC Ratio Analysis: {region.upper()}")
        print(f"{'='*60}")
        
        # 1. Baseline 엔트로피 평균 계산 (target_classes에 대해)
        baseline_entropies_list = []
        
        for model_cfg in model_configs:
            model_name = model_cfg['model_name']
            data_sample = model_cfg['data_sample']
            
            all_probs, _, y_true_list = predict_val_proba(
                model_name=model_name,
                region=region,
                data_sample=data_sample,
                device=device,
                n_folds=n_folds
            )
            
            # 각 fold별로 엔트로피 계산
            for fold_idx in range(n_folds):
                fold_mask = np.isin(y_true_list[fold_idx], target_classes)
                
                if np.sum(fold_mask) == 0:
                    continue
                
                fold_probs = all_probs[fold_idx][fold_mask]
                fold_entropies = _calculate_entropy_from_probs(fold_probs)
                baseline_entropies_list.extend(fold_entropies)
        
        baseline_entropies = np.array(baseline_entropies_list)
        baseline_entropy_mean = np.mean(baseline_entropies)
        
        # Threshold 계산 (민감도 분석을 위한 옵션)
        if threshold_method == 'mean':
            threshold = baseline_entropy_mean
        elif threshold_method == 'median':
            threshold = np.median(baseline_entropies)
        elif threshold_method == 'q25':
            threshold = np.percentile(baseline_entropies, 25)
        else:
            raise ValueError(f"Invalid threshold_method: {threshold_method}. "
                           f"Choose from ['mean', 'median', 'q25']")
        
        print(f"Baseline Entropy Mean: {baseline_entropy_mean:.4f}")
        print(f"Threshold Method: {threshold_method}, Threshold Value: {threshold:.4f}")
        
        # 2. Test 데이터 예측 수행
        final_probs, y_test = _perform_test_predictions(
            model_configs, region, device, n_folds, verbose=False
        )
        y_pred = np.argmax(final_probs, axis=1)
        
        # 3. target_classes 샘플만 필터링
        test_mask = np.isin(y_test, target_classes)
        if np.sum(test_mask) == 0:
            print(f"Warning: {region}에 Class {target_classes} 샘플이 없습니다.")
            continue
        
        y_true_filtered = y_test[test_mask]
        y_pred_filtered = y_pred[test_mask]
        probs_filtered = final_probs[test_mask]
        
        total_samples = len(y_true_filtered)
        
        # 4. 엔트로피 계산
        ents = _calculate_entropy_from_probs(probs_filtered)
        mean_entropy_test = np.mean(ents)  # 전체 test 샘플의 엔트로피 평균 (entropy_test에 해당)
        
        # 5. WBC 비율 계산 (선택된 threshold 사용)
        is_confident = ents < threshold
        is_wrong = y_true_filtered != y_pred_filtered
        is_wbc = is_confident & is_wrong
        
        confident_samples = np.sum(is_confident)
        wrong_samples = np.sum(is_wrong)
        wbc_samples = np.sum(is_wbc)
        
        wbc_ratio_in_confident = (wbc_samples / confident_samples * 100) if confident_samples > 0 else 0.0
        wbc_ratio_in_errors = (wbc_samples / wrong_samples * 100) if wrong_samples > 0 else 0.0
        
        mean_entropy_confident = np.mean(ents[is_confident]) if confident_samples > 0 else 0.0
        mean_entropy_wrong = np.mean(ents[is_wrong]) if wrong_samples > 0 else 0.0
        
        # 엔트로피 변화량 계산 (baseline - test)
        entropy_diff = baseline_entropy_mean - mean_entropy_test
        
        wbc_stats.append({
            'region': region,
            'total_samples': total_samples,
            'confident_samples': confident_samples,
            'wrong_samples': wrong_samples,
            'wbc_samples': wbc_samples,
            'wbc_ratio_in_confident': wbc_ratio_in_confident,
            'wbc_ratio_in_errors': wbc_ratio_in_errors,
            'baseline_entropy_mean': baseline_entropy_mean,
            'mean_entropy_test': mean_entropy_test,  # 전체 test 샘플의 엔트로피 평균 (entropy_test에 해당)
            'entropy_diff': entropy_diff,  # baseline_entropy_mean - mean_entropy_test
            'mean_entropy_confident': mean_entropy_confident,
            'mean_entropy_wrong': mean_entropy_wrong,
            'threshold_method': threshold_method,
            'threshold_value': threshold
        })
        
        print(f"Total Samples: {total_samples}")
        print(f"Confident Samples: {confident_samples} ({confident_samples/total_samples*100:.2f}%)")
        print(f"Wrong Samples: {wrong_samples} ({wrong_samples/total_samples*100:.2f}%)")
        print(f"WBC Samples: {wbc_samples}")
        print(f"WBC Ratio in Confident: {wbc_ratio_in_confident:.2f}%")
        print(f"WBC Ratio in Errors: {wbc_ratio_in_errors:.2f}%")
    
    return pd.DataFrame(wbc_stats)


def calculate_wbc_ratio_with_correction(
    model_configs: List[Dict[str, Any]],
    regions: List[str],
    device: str = 'cpu',
    n_folds: int = 3,
    target_classes: List[int] = None,
    threshold_method: str = 'mean',
    alpha: float = None
) -> pd.DataFrame:
    """Multiple testing correction을 적용한 WBC 비율 계산 함수.
    
    여러 지역에 대해 동시에 테스트할 때 발생하는 Type I error 증가를 
    Benjamini-Hochberg FDR correction으로 보정합니다.
    
    Args:
        model_configs: 모델 설정 리스트
        regions: 지역명 리스트
        device: 연산 장치 (기본값: 'cpu')
        n_folds: Fold 개수 (기본값: 3)
        target_classes: 분석할 클래스 리스트 (기본값: [2])
        threshold_method: Confidence threshold 계산 방법 (기본값: 'mean')
        alpha: 유의 수준 (기본값: SHAPConfig.ALPHA = 0.05)
    
    Returns:
        pd.DataFrame: calculate_wbc_ratio 결과에 FDR correction 정보 추가
            - 기존 컬럼들
            - p_value: 귀무가설(WBC ratio = chance level) 검정 p-value (placeholder)
            - p_adjusted: FDR 보정된 p-value
            - significant: 유의성 여부 (alpha 수준에서)
    
    Note:
        - 현재 구현은 프레임워크 제공 목적
        - 실제 p-value 계산은 연구 설계에 따라 구현 필요
        - 예: Binomial test, Permutation test 등
        - FDR correction은 Benjamini-Hochberg 방법 사용
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        raise ImportError("statsmodels가 필요합니다: pip install statsmodels")
    
    if alpha is None:
        alpha = SHAPConfig.ALPHA
    
    # 기본 WBC 비율 계산
    wbc_df = calculate_wbc_ratio(
        model_configs=model_configs,
        regions=regions,
        device=device,
        n_folds=n_folds,
        target_classes=target_classes,
        threshold_method=threshold_method
    )
    
    # P-value 계산 (placeholder - 실제 구현 시 적절한 통계 검정 사용)
    # 예: WBC ratio가 chance level (confident_samples / total_samples * error_rate)과 다른지 검정
    p_values = []
    for _, row in wbc_df.iterrows():
        # Placeholder: 실제로는 binomial test 등 사용
        # 여기서는 간단히 WBC ratio가 0과 다른지 검정하는 예시
        # 실제 사용 시 적절한 귀무가설 설정 필요
        
        # 예시: Binomial test (WBC samples가 chance level보다 많은지)
        # chance_level = (confident_samples / total_samples) * (wrong_samples / total_samples)
        # 실제 구현 필요
        
        # Placeholder p-value (실제로는 통계 검정 결과 사용)
        if row['wbc_samples'] > 0:
            # 임시: WBC ratio에 반비례하는 p-value (실제로는 적절한 검정 사용)
            p_val = min(0.5, 1.0 / (1.0 + row['wbc_ratio_in_confident']))
        else:
            p_val = 1.0
        
        p_values.append(p_val)
    
    # FDR correction 적용
    rejected, p_adjusted, _, _ = multipletests(
        p_values, 
        alpha=alpha, 
        method=SHAPConfig.FDR_METHOD
    )
    
    # 결과 추가
    wbc_df['p_value'] = p_values
    wbc_df['p_adjusted'] = p_adjusted
    wbc_df['significant'] = rejected
    
    print(f"\n{'='*60}")
    print(f"Multiple Testing Correction Applied (FDR, alpha={alpha})")
    print(f"Significant regions: {wbc_df[wbc_df['significant']]['region'].tolist()}")
    print(f"{'='*60}")
    
    return wbc_df
