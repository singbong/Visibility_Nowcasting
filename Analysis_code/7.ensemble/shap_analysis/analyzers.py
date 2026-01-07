"""SHAP 분석 클래스들."""

import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import torch
from joblib import load
from scipy.stats import t as t_dist

from .config import SHAPConfig
from .data_loader import DataLoader
from .preprocessors import ModelWrapper, ShapValueExtractor, suppress_shap_output

# 모델 클래스 import를 위한 경로 설정
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.abspath(os.path.join(current_file_dir, '../../models'))
except NameError:
    cwd = os.getcwd()
    if 'Analysis_code' in cwd:
        analysis_code_dir = cwd[:cwd.index('Analysis_code') + len('Analysis_code')]
        models_path = os.path.join(analysis_code_dir, 'models')
    else:
        models_path = '/workspace/visibility_prediction/Analysis_code/models'

if models_path not in sys.path:
    sys.path.insert(0, models_path)

from ft_transformer import FTTransformer
from resnet_like import ResNetLike
from deepgbm import DeepGBM

# 경고 무시
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, message='.*unrecognized nn.Module.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*LayerNorm.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*unrecognized.*')
warnings.filterwarnings('ignore', module='shap.*')


# ==========================================
# 데이터 클래스
# ==========================================

@dataclass
class ImportanceData:
    """Feature importance 데이터를 담는 클래스."""
    df: pd.DataFrame
    top_n_features: List[str]
    top_n_share: float
    importance_std: Optional[np.ndarray] = None  # Fold 간 표준편차
    importance_ci_lower: Optional[np.ndarray] = None  # 95% 신뢰구간 하한
    importance_ci_upper: Optional[np.ndarray] = None  # 95% 신뢰구간 상한


@dataclass
class SHAPResult:
    """SHAP 분석 결과를 담는 클래스."""
    shap_train: Optional[np.ndarray]
    X_train: Optional[pd.DataFrame]
    shap_test: Optional[np.ndarray]
    X_test: Optional[pd.DataFrame]
    feature_names: List[str]
    importance_train: Optional[ImportanceData]
    importance_test: Optional[ImportanceData]
    importance_df_combined: Optional[pd.DataFrame]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """기존 dict 형태로 변환 (하위 호환성)."""
        return {
            'shap_train': self.shap_train,
            'X_train': self.X_train,
            'shap_test': self.shap_test,
            'X_test': self.X_test,
            'feature_names': self.feature_names,
            'top_n_features_train': self.importance_train.top_n_features if self.importance_train else None,
            'top_n_share_train': self.importance_train.top_n_share if self.importance_train else None,
            'importance_df_train': self.importance_train.df if self.importance_train else None,
            'top_n_features_test': self.importance_test.top_n_features if self.importance_test else None,
            'top_n_share_test': self.importance_test.top_n_share if self.importance_test else None,
            'importance_df_test': self.importance_test.df if self.importance_test else None,
            'importance_df_combined': self.importance_df_combined,
            **self.metadata
        }


# ==========================================
# 유틸리티 함수
# ==========================================

def _stratified_sample(
    data: pd.DataFrame,
    n_samples: int,
    random_state: int,
    exclude_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """클래스 비율을 유지하며 샘플링 (stratified sampling).
    
    Args:
        data: 샘플링할 데이터 (multi_class 컬럼 필수)
        n_samples: 샘플링할 개수
        random_state: 랜덤 시드
        exclude_indices: 제외할 인덱스 (선택사항)
    
    Returns:
        샘플링된 인덱스 배열
    """
    if 'multi_class' not in data.columns:
        # multi_class 컬럼이 없으면 단순 랜덤 샘플링
        available_indices = np.setdiff1d(np.arange(len(data)), exclude_indices) if exclude_indices is not None else np.arange(len(data))
        if len(available_indices) < n_samples:
            return available_indices
        np.random.seed(random_state)
        return np.random.choice(available_indices, n_samples, replace=False)
    
    # 사용 가능한 인덱스 결정
    if exclude_indices is not None:
        available_indices = np.setdiff1d(np.arange(len(data)), exclude_indices)
    else:
        available_indices = np.arange(len(data))
    
    if len(available_indices) < n_samples:
        return available_indices
    
    # 사용 가능한 데이터 추출 (인덱스는 원본 유지)
    available_data = data.iloc[available_indices].copy()
    
    # 클래스별 분포 계산
    class_counts = available_data['multi_class'].value_counts().sort_index()
    total_samples = len(available_data)
    
    # 각 클래스별 샘플 수 계산 (비율 유지)
    np.random.seed(random_state)
    sampled_indices = []
    
    for class_label, class_count in class_counts.items():
        # 해당 클래스의 샘플 수 계산 (비율 유지)
        class_proportion = class_count / total_samples
        n_class_samples = max(1, int(np.round(n_samples * class_proportion)))
        
        # 해당 클래스의 인덱스 추출 (원본 인덱스 유지)
        class_mask = available_data['multi_class'] == class_label
        class_indices = available_data[class_mask].index.values  # 원본 인덱스
        
        # 샘플링
        if len(class_indices) <= n_class_samples:
            sampled_class_indices = class_indices.tolist()
        else:
            sampled_class_indices = np.random.choice(
                class_indices, n_class_samples, replace=False
            ).tolist()
        
        sampled_indices.extend(sampled_class_indices)
    
    # 정확히 n_samples 개가 되도록 조정
    if len(sampled_indices) > n_samples:
        # 초과분 제거 (랜덤하게)
        np.random.seed(random_state + 1)
        sampled_indices = np.random.choice(sampled_indices, n_samples, replace=False).tolist()
    elif len(sampled_indices) < n_samples:
        # 부족분 추가 (남은 인덱스에서)
        remaining_indices = np.setdiff1d(available_indices, sampled_indices)
        if len(remaining_indices) > 0:
            n_needed = n_samples - len(sampled_indices)
            additional_indices = np.random.choice(
                remaining_indices, min(n_needed, len(remaining_indices)), replace=False
            ).tolist()
            sampled_indices.extend(additional_indices)
    
    return np.array(sampled_indices)


def _print_class_distribution(
    data: pd.DataFrame, 
    indices: np.ndarray, 
    label: str = "",
    original_data: Optional[pd.DataFrame] = None,
    original_indices: Optional[np.ndarray] = None
) -> None:
    """Class 분포를 출력하는 유틸리티 함수 (비율 비교 포함).
    
    Args:
        data: 데이터프레임 (multi_class 컬럼 포함)
        indices: 출력할 행 인덱스 (라벨 기반 또는 위치 기반)
        label: 출력 레이블 (선택사항)
        original_data: 원본 데이터 (비율 비교용, 선택사항)
        original_indices: 원본 인덱스 (비율 비교용, 선택사항)
    """
    if 'multi_class' not in data.columns:
        return
    
    try:
        # 먼저 indices가 data.index에 있는지 확인 (라벨 기반)
        if len(indices) > 0 and np.all(np.isin(indices, data.index)):
            # 라벨 기반 인덱싱
            class_dist = data.loc[indices]['multi_class'].value_counts().sort_index()
        elif len(indices) > 0 and indices.min() >= 0 and indices.max() < len(data):
            # 위치 기반 인덱싱 (0부터 len(data)-1 범위)
            class_dist = data.iloc[indices]['multi_class'].value_counts().sort_index()
        else:
            return  # 유효하지 않은 인덱스
    except (IndexError, KeyError):
        return  # 에러 발생 시 출력하지 않음
    
    label_text = f"{label} " if label else ""
    print(f"{label_text}Class distribution: {class_dist.to_dict()}")
    
    # 비율 정보 추가
    total = len(indices)
    if total > 0:
        proportions = {k: f"{v/total*100:.1f}%" for k, v in class_dist.items()}
        print(f"{label_text}Class proportions: {proportions}")
    
    # 원본 데이터와 비교 (제공된 경우)
    if original_data is not None and original_indices is not None and 'multi_class' in original_data.columns:
        try:
            # 원본 데이터도 동일한 방식으로 처리
            if len(original_indices) > 0 and np.all(np.isin(original_indices, original_data.index)):
                original_dist = original_data.loc[original_indices]['multi_class'].value_counts().sort_index()
            elif len(original_indices) > 0 and original_indices.min() >= 0 and original_indices.max() < len(original_data):
                original_dist = original_data.iloc[original_indices]['multi_class'].value_counts().sort_index()
            else:
                return  # 유효하지 않은 인덱스
            
            original_total = len(original_indices)
            
            if original_total > 0:
                print(f"{label_text}Original class distribution: {original_dist.to_dict()}")
                original_proportions = {k: v/original_total for k, v in original_dist.items()}
                sample_proportions = {k: v/total for k, v in class_dist.items()}
                
                # 비율 차이 계산
                all_classes = set(original_proportions.keys()) | set(sample_proportions.keys())
                proportion_diff = {}
                for cls in all_classes:
                    orig_prop = original_proportions.get(cls, 0)
                    samp_prop = sample_proportions.get(cls, 0)
                    diff = samp_prop - orig_prop
                    proportion_diff[cls] = f"{diff*100:+.2f}%"
                print(f"{label_text}Proportion difference (sample - original): {proportion_diff}")
        except (IndexError, KeyError):
            pass  # 원본 비교 실패 시 무시


# ==========================================
# SHAP Analyzer 베이스 클래스
# ==========================================

class SHAPAnalyzer(ABC):
    """SHAP 분석을 위한 추상 베이스 클래스."""
    
    def __init__(self, config: SHAPConfig, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
    
    @abstractmethod
    def create_explainer(self, model: Any, background_data: Any) -> Any:
        """SHAP Explainer 생성."""
        pass
    
    @abstractmethod
    def calculate_shap_values(self, explainer: Any, sample_data: Any) -> Any:
        """SHAP 값 계산."""
        pass
    
    def _calculate_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        period_name: str = '',
        shap_values_list: Optional[List[np.ndarray]] = None
    ) -> ImportanceData:
        """누적 설명력(Cumulative Share) 계산.
        
        Args:
            shap_values: SHAP 값 배열 (평균 또는 단일)
            feature_names: Feature 이름 리스트
            period_name: 기간 이름 (출력용)
            shap_values_list: Fold별 SHAP 값 리스트 (불확실성 계산용, optional)
        """
        global_importances = np.abs(shap_values).mean(axis=0)
        
        # 불확실성 계산 (fold별 SHAP 값이 제공된 경우)
        importance_std = None
        importance_ci_lower = None
        importance_ci_upper = None
        
        if shap_values_list is not None and len(shap_values_list) > 1:
            # Fold별 importance 계산
            fold_importances = []
            for shap_vals in shap_values_list:
                fold_imp = np.abs(shap_vals).mean(axis=0)
                fold_importances.append(fold_imp)
            
            # 표준편차 계산
            fold_importances_array = np.array(fold_importances)
            importance_std = np.std(fold_importances_array, axis=0)
            
            # 95% 신뢰구간 계산 (정규분포 가정, t-분포 사용)
            n_folds = len(shap_values_list)
            t_critical = t_dist.ppf(0.975, n_folds - 1)  # 95% 신뢰구간
            se = importance_std / np.sqrt(n_folds)
            importance_ci_lower = global_importances - t_critical * se
            importance_ci_upper = global_importances + t_critical * se
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': global_importances
        })
        
        if importance_std is not None:
            importance_df['importance_std'] = importance_std
            importance_df['importance_ci_lower'] = importance_ci_lower
            importance_df['importance_ci_upper'] = importance_ci_upper
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        total_importance = importance_df['importance'].sum()
        top_n_df = importance_df.head(self.config.top_n)
        top_n_importance = top_n_df['importance'].sum()
        top_n_share = (top_n_importance / total_importance) * 100
        top_n_features = top_n_df['feature'].tolist()
        
        if period_name:
            print(f"{period_name} - Top {self.config.top_n} Dominant Features: {top_n_features}")
            print(f"{period_name} - Cumulative Importance Share (Top {self.config.top_n}): {top_n_share:.2f}%")
        else:
            print(f"Top {self.config.top_n} Dominant Features: {top_n_features}")
            print(f"Cumulative Importance Share (Top {self.config.top_n}): {top_n_share:.2f}%")
        print(f"{'-'*40}")
        
        return ImportanceData(
            df=importance_df,
            top_n_features=top_n_features,
            top_n_share=top_n_share,
            importance_std=importance_std,
            importance_ci_lower=importance_ci_lower,
            importance_ci_upper=importance_ci_upper
        )
    
    def _create_combined_importance_df(
        self,
        importance_train: Optional[ImportanceData],
        importance_test: Optional[ImportanceData],
        feature_names: List[str]
    ) -> Optional[pd.DataFrame]:
        """Train과 Test 중요도 DataFrame 통합."""
        if importance_train is None and importance_test is None:
            return None
        
        if importance_train is not None:
            train_df = importance_train.df[['feature', 'importance']].copy()
            train_df.columns = ['feature', 'importance_train']
            total_importance_train = train_df['importance_train'].sum()
            train_df['share_train'] = (train_df['importance_train'] / total_importance_train) * 100
        else:
            train_df = pd.DataFrame({'feature': feature_names})
            train_df['importance_train'] = np.nan
            train_df['share_train'] = np.nan
        
        if importance_test is not None:
            test_df = importance_test.df[['feature', 'importance']].copy()
            test_df.columns = ['feature', 'importance_test']
            total_importance_test = test_df['importance_test'].sum()
            test_df['share_test'] = (test_df['importance_test'] / total_importance_test) * 100
        else:
            test_df = pd.DataFrame({'feature': feature_names})
            test_df['importance_test'] = np.nan
            test_df['share_test'] = np.nan
        
        importance_df_combined = pd.merge(train_df, test_df, on='feature', how='outer')
        
        if importance_train is not None:
            importance_df_combined = importance_df_combined.sort_values(
                'importance_train', ascending=False
            ).reset_index(drop=True)
        elif importance_test is not None:
            importance_df_combined = importance_df_combined.sort_values(
                'importance_test', ascending=False
            ).reset_index(drop=True)
        else:
            importance_df_combined = importance_df_combined.sort_values('feature').reset_index(drop=True)
        
        return importance_df_combined


# ==========================================
# TreeSHAPAnalyzer 구현
# ==========================================

class TreeSHAPAnalyzer(SHAPAnalyzer):
    """Tree 기반 모델(XGBoost, LightGBM) SHAP 분석 클래스."""
    
    def create_explainer(self, model: Any, background_data: Any = None) -> shap.TreeExplainer:
        """TreeExplainer 생성."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*deprecated binary model format.*')
            return shap.TreeExplainer(model.get_booster())
    
    def calculate_shap_values(self, explainer: shap.TreeExplainer, sample_data: pd.DataFrame) -> np.ndarray:
        """Tree SHAP 값 계산 (fog = class 0 + class 1)."""
        shap_values = explainer.shap_values(sample_data)
        if isinstance(shap_values, list):
            # fog = class 0 + class 1
            if len(shap_values) >= 2:
                return shap_values[0] + shap_values[1]
            return shap_values[0]
        return shap_values
    
    def analyze(
        self,
        model_name: str,
        region: str,
        data_sample: str,
        show_plot: bool = True,
        show_train_plot: bool = True,
        show_test_plot: bool = True,
        suppress_warnings: bool = True
    ) -> SHAPResult:
        """Tree 모델 SHAP 분석 수행."""
        import sys
        
        # Pickle 역직렬화를 위해 __main__ 모듈에 필요한 함수 등록
        try:
            from model_utils import calculate_csi, eval_metric_csi, csi_metric
            main_module = sys.modules.get('__main__')
            if main_module is not None:
                main_module.calculate_csi = calculate_csi
                if model_name == 'xgb':
                    main_module.eval_metric_csi = eval_metric_csi
                elif model_name == 'lgb':
                    main_module.csi_metric = csi_metric
        except ImportError:
            pass
        
        # 모델 로드
        model_path = f"../save_model/{model_name}_optima/{model_name}_{data_sample}_{region}.pkl"
        models = load(model_path)
        
        if len(models) != self.config.n_folds:
            raise ValueError(f"로드된 모델 개수({len(models)})가 n_folds({self.config.n_folds})와 일치하지 않습니다.")
        
        # 각 Fold에 대해 SHAP 값 계산
        oof_shap_values = []
        oof_X_data = []
        test_shap_values = []
        
        for fold_idx in range(self.config.n_folds):
            _, val_data, test_data = self.data_loader.prepare_for_tree_model(
                region, data_sample, fold_idx
            )
            
            X_val = val_data.drop(columns=self.config.DROP_COLUMNS)
            if fold_idx == 0:
                X_test = test_data.drop(columns=self.config.DROP_COLUMNS)
            
            # Class 분포 확인 (논문 보고용)
            _print_class_distribution(val_data, val_data.index, f"Fold {fold_idx + 1} - Validation data")
            
            explainer = self.create_explainer(models[fold_idx])
            shap_val = self.calculate_shap_values(explainer, X_val)
            oof_shap_values.append(shap_val)
            oof_X_data.append(X_val)
            
            shap_test = self.calculate_shap_values(explainer, X_test)
            test_shap_values.append(shap_test)
        
        # 데이터 통합
        all_shap_train = np.concatenate(oof_shap_values, axis=0)
        all_X_train = pd.concat(oof_X_data, axis=0)
        all_shap_test = np.mean(test_shap_values, axis=0)
        all_shap_test_std = np.std(test_shap_values, axis=0)  # Fold 간 표준편차
        feature_names = list(all_X_train.columns)
        
        # 중요도 계산
        importance_train = self._calculate_importance(
            all_shap_train, feature_names, period_name='Train Period',
            shap_values_list=oof_shap_values  # Fold별 SHAP 값 전달
        )
        importance_test = self._calculate_importance(
            all_shap_test, feature_names, period_name='Test Period',
            shap_values_list=test_shap_values  # Fold별 SHAP 값 전달
        )
        print()
        
        importance_df_combined = self._create_combined_importance_df(
            importance_train, importance_test, feature_names
        )
        
        # 시각화
        if show_plot:
            from .visualizers import SHAPVisualizer
            visualizer = SHAPVisualizer(self.config)
            visualizer.plot_summary(
                all_shap_train, all_X_train, all_shap_test, X_test,
                feature_names, importance_train, importance_test,
                show_train_plot=show_train_plot, show_test_plot=show_test_plot,
                model_name=model_name, data_sample=data_sample, region=region
            )
        
        return SHAPResult(
            shap_train=all_shap_train,
            X_train=all_X_train,
            shap_test=all_shap_test,
            X_test=X_test,
            feature_names=feature_names,
            importance_train=importance_train,
            importance_test=importance_test,
            importance_df_combined=importance_df_combined,
            metadata={'model_name': model_name, 'region': region, 'data_sample': data_sample}
        )


# ==========================================
# DeepSHAPAnalyzer 구현
# ==========================================

class DeepSHAPAnalyzer(SHAPAnalyzer):
    """Deep Learning 모델 SHAP 분석 클래스."""
    
    def __init__(self, config: SHAPConfig, data_loader: DataLoader, device: str = 'cpu'):
        super().__init__(config, data_loader)
        self.device = device
    
    def create_explainer(
        self,
        model: torch.nn.Module,
        background_data: Tuple[torch.Tensor, torch.Tensor, pd.Index],
        model_name: str
    ) -> shap.DeepExplainer:
        """DeepExplainer 생성."""
        X_bg_num, X_bg_cat, numerical_cols = background_data
        wrapped_model = ModelWrapper(model, numerical_cols)
        wrapped_model.to(self.device)
        wrapped_model.eval()
        
        with suppress_shap_output():
            if model_name in ('ft_transformer', 'deepgbm'):
                X_bg_cat_float = X_bg_cat.float()
                explainer = shap.DeepExplainer(wrapped_model, [X_bg_num, X_bg_cat_float])
            elif model_name == 'resnet_like':
                X_bg_combined = torch.cat([X_bg_num, X_bg_cat.float()], dim=1)
                explainer = shap.DeepExplainer(wrapped_model, X_bg_combined)
            else:
                raise ValueError(f"Unknown model_name: {model_name}")
        
        return explainer
    
    def calculate_shap_values(
        self,
        explainer: shap.DeepExplainer,
        sample_data: Tuple[torch.Tensor, torch.Tensor],
        model_name: str
    ) -> Any:
        """Deep SHAP 값 계산 (fog = class 0 + class 1).
        
        Note: 반환된 SHAP 값은 ShapValueExtractor.extract()에서 
        class 0과 class 1을 합산하여 fog 값으로 변환됩니다.
        """
        X_sample_num, X_sample_cat = sample_data
        
        with suppress_shap_output():
            if model_name in ('ft_transformer', 'deepgbm'):
                X_sample_cat_float = X_sample_cat.float()
                shap_values = explainer.shap_values([X_sample_num, X_sample_cat_float])
            elif model_name == 'resnet_like':
                X_sample_combined = torch.cat([X_sample_num, X_sample_cat.float()], dim=1)
                shap_values = explainer.shap_values(X_sample_combined)
            else:
                raise ValueError(f"Unknown model_name: {model_name}")
        
        return shap_values
    
    def _sample_background_data(
        self,
        X_val_num: torch.Tensor,
        X_val_cat: torch.Tensor,
        X_val: pd.DataFrame,
        fold_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Background 데이터 샘플링 (stratified sampling 적용)."""
        n_bg = min(self.config.n_background_samples, len(X_val_num))
        
        # Stratified sampling으로 클래스 비율 유지
        bg_indices = _stratified_sample(
            X_val, n_bg, self.config.random_seed + fold_idx
        )
        
        X_bg_num = X_val_num[bg_indices].to(self.device)
        X_bg_cat = X_val_cat[bg_indices].to(self.device)
        
        # Class 분포 확인 (논문 보고용, 원본과 비교)
        print(f"  Background data sampling (stratified, n={len(bg_indices)}):")
        _print_class_distribution(
            X_val, bg_indices, "  Background data",
            original_data=X_val, original_indices=np.arange(len(X_val))
        )
        
        return X_bg_num, X_bg_cat, bg_indices
    
    def _sample_train_data(
        self,
        X_val_num: torch.Tensor,
        X_val_cat: torch.Tensor,
        X_val: pd.DataFrame,
        bg_indices: np.ndarray,
        fold_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Train 데이터 샘플링 (background와 겹치지 않게, stratified sampling 적용)."""
        n_train = min(self.config.n_test_samples, len(X_val_num))
        available_indices = np.setdiff1d(np.arange(len(X_val_num)), bg_indices)
        if len(available_indices) == 0:
            raise ValueError("Background data와 train data가 겹칩니다.")
        if len(available_indices) < n_train:
            train_indices = available_indices
        else:
            # Stratified sampling으로 클래스 비율 유지 (background 제외)
            train_indices = _stratified_sample(
                X_val, n_train, 
                self.config.random_seed + fold_idx + self.config.SEED_OFFSET_TRAIN,
                exclude_indices=bg_indices
            )
        X_train_sample_num = X_val_num[train_indices].to(self.device)
        X_train_sample_cat = X_val_cat[train_indices].to(self.device)
        
        # Class 분포 확인 (논문 보고용, 원본과 비교)
        print(f"  Train data sampling (stratified, n={len(train_indices)}):")
        _print_class_distribution(
            X_val, train_indices, "  Train data",
            original_data=X_val, original_indices=available_indices
        )
        
        return X_train_sample_num, X_train_sample_cat, train_indices
    
    def _sample_test_data(
        self,
        X_test_num: torch.Tensor,
        X_test_cat: torch.Tensor,
        X_test: pd.DataFrame,
        fold_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Test 데이터 샘플링 (완전 분리, stratified sampling 적용)."""
        n_test = min(self.config.n_test_samples, len(X_test_num))
        
        # Stratified sampling으로 클래스 비율 유지
        test_indices = _stratified_sample(
            X_test, n_test, 
            self.config.random_seed + fold_idx + self.config.SEED_OFFSET_TEST
        )
        
        X_test_sample_num = X_test_num[test_indices].to(self.device)
        X_test_sample_cat = X_test_cat[test_indices].to(self.device)
        
        # Class 분포 확인 (논문 보고용, 원본과 비교)
        print(f"  Test data sampling (stratified, n={len(test_indices)}):")
        _print_class_distribution(
            X_test, test_indices, "  Test data",
            original_data=X_test, original_indices=np.arange(len(X_test))
        )
        
        return X_test_sample_num, X_test_sample_cat, test_indices
    
    def _compute_shap_for_fold(
        self,
        model: torch.nn.Module,
        model_name: str,
        numerical_cols: pd.Index,
        X_bg_num: torch.Tensor,
        X_bg_cat: torch.Tensor,
        X_train_sample_num: torch.Tensor,
        X_train_sample_cat: torch.Tensor,
        X_test_sample_num: torch.Tensor,
        X_test_sample_cat: torch.Tensor,
        X_val: pd.DataFrame,
        train_indices: np.ndarray,
        X_test: pd.DataFrame,
        test_indices: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """단일 fold에 대한 SHAP 값 계산."""
        explainer = self.create_explainer(
            model, (X_bg_num, X_bg_cat, numerical_cols), model_name
        )
        
        shap_values_train = self.calculate_shap_values(
            explainer, (X_train_sample_num, X_train_sample_cat), model_name
        )
        shap_val_train = ShapValueExtractor.extract(shap_values_train)
        X_train_original = X_val.iloc[train_indices].copy()
        
        shap_values_test = self.calculate_shap_values(
            explainer, (X_test_sample_num, X_test_sample_cat), model_name
        )
        shap_val_test = ShapValueExtractor.extract(shap_values_test)
        X_test_original = X_test.iloc[test_indices].copy()
        
        return {
            'shap_train': shap_val_train,
            'X_train': X_train_original,
            'shap_test': shap_val_test,
            'X_test': X_test_original
        }
    
    def _aggregate_results(
        self,
        all_shap_train: List[np.ndarray],
        all_shap_test: List[np.ndarray],
        all_train_data: List[pd.DataFrame],
        all_test_data: List[pd.DataFrame],
        feature_names: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fold별 SHAP 결과를 통합."""
        all_shap_train_combined = (
            np.concatenate(all_shap_train, axis=0) if len(all_shap_train) > 0 else None
        )
        all_train_data_combined = (
            pd.concat(all_train_data, axis=0).reset_index(drop=True) if len(all_train_data) > 0 else None
        )
        
        if len(all_shap_test) > 0:
            all_shap_test_combined = np.mean(all_shap_test, axis=0)
            all_test_data_combined = all_test_data[0] if len(all_test_data) > 0 else None
        else:
            all_shap_test_combined = None
            all_test_data_combined = None
        
        return all_shap_train_combined, all_shap_test_combined, all_train_data_combined, all_test_data_combined
    
    def analyze(
        self,
        model_name: str,
        region: str,
        data_sample: str,
        show_plot: bool = True,
        show_train_plot: bool = True,
        show_test_plot: bool = True
    ) -> Optional[SHAPResult]:
        """Deep Learning 모델 SHAP 분석 수행."""
        print(f"[{model_name.upper()}] {region} - {data_sample} SHAP Analysis Start...")
        
        # 모델 및 scaler 로드
        try:
            model_path = f"../save_model/{model_name}_optima/{model_name}_{data_sample}_{region}.pkl"
            models = load(model_path)
            
            if data_sample == 'pure':
                scaler_filename = f'{model_name}_pure_{region}_scaler.pkl'
            else:
                scaler_filename = f'{model_name}_{data_sample}_{region}_scaler.pkl'
            scaler_path = f'../save_model/{model_name}_optima/scaler/{scaler_filename}'
            scalers = load(scaler_path)
        except FileNotFoundError as e:
            print(f"모델 또는 scaler 파일을 찾을 수 없습니다: {e}")
            return None
        except Exception as e:
            print(f"모델 또는 scaler 로드 실패: {e}")
            return None
        
        if len(models) != self.config.n_folds or len(scalers) != self.config.n_folds:
            raise ValueError(
                f"모델 개수({len(models)}) 또는 scaler 개수({len(scalers)})가 "
                f"n_folds({self.config.n_folds})와 일치하지 않습니다."
            )
        
        # 각 fold의 SHAP 값 저장
        all_shap_train = []
        all_shap_test = []
        all_train_data = []
        all_test_data = []
        feature_names = None
        
        for fold_idx in range(self.config.n_folds):
            print(f"  Processing Fold {fold_idx + 1}...")
            
            try:
                (X_train, X_val, X_test, categorical_cols, numerical_cols,
                 X_train_num, X_train_cat, X_val_num, X_val_cat, X_test_num, X_test_cat) = \
                    self.data_loader.prepare_for_dl_model(region, data_sample, fold_idx, scalers[fold_idx])
                
                if feature_names is None:
                    feature_names = list(numerical_cols) + list(categorical_cols)
                
                model = models[fold_idx]
                model.to(self.device)
                model.eval()
                
                # 배경 데이터 샘플링
                X_bg_num, X_bg_cat, bg_indices = self._sample_background_data(
                    X_val_num, X_val_cat, X_val, fold_idx
                )
                
                # Validation 데이터 샘플링
                X_train_sample_num, X_train_sample_cat, train_indices = self._sample_train_data(
                    X_val_num, X_val_cat, X_val, bg_indices, fold_idx
                )
                
                # 테스트 데이터 샘플링
                X_test_sample_num, X_test_sample_cat, test_indices = self._sample_test_data(
                    X_test_num, X_test_cat, X_test, fold_idx
                )
                
                # SHAP Explainer 생성 및 계산
                try:
                    fold_results = self._compute_shap_for_fold(
                        model, model_name, numerical_cols,
                        X_bg_num, X_bg_cat,
                        X_train_sample_num, X_train_sample_cat,
                        X_test_sample_num, X_test_sample_cat,
                        X_val, train_indices, X_test, test_indices
                    )
                    if fold_results:
                        all_shap_train.append(fold_results['shap_train'])
                        all_train_data.append(fold_results['X_train'])
                        all_shap_test.append(fold_results['shap_test'])
                        all_test_data.append(fold_results['X_test'])
                except RuntimeError as e:
                    print(f"  Warning: Fold {fold_idx + 1} SHAP 계산 실패: {e}")
                    continue
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                print(f"  Warning: Fold {fold_idx + 1} 처리 실패: {e}")
                continue
        
        if len(all_shap_train) == 0 and len(all_shap_test) == 0:
            print("  모든 fold에서 SHAP 계산 실패")
            return None
        
        # 데이터 통합
        all_shap_train_combined, all_shap_test_combined, all_train_data_combined, all_test_data_combined = \
            self._aggregate_results(all_shap_train, all_shap_test, all_train_data, all_test_data, feature_names)
        
        # 중요도 계산
        if all_shap_train_combined is not None:
            importance_train = self._calculate_importance(
                all_shap_train_combined, feature_names, period_name='Train Period',
                shap_values_list=all_shap_train if len(all_shap_train) > 0 else None
            )
        else:
            importance_train = None
        
        if all_shap_test_combined is not None:
            importance_test = self._calculate_importance(
                all_shap_test_combined, feature_names, period_name='Test Period',
                shap_values_list=all_shap_test if len(all_shap_test) > 0 else None
            )
            print()
        else:
            importance_test = None
        
        importance_df_combined = self._create_combined_importance_df(
            importance_train, importance_test, feature_names
        )
        
        # 시각화
        if show_plot:
            from .visualizers import SHAPVisualizer
            visualizer = SHAPVisualizer(self.config)
            visualizer.plot_summary(
                all_shap_train_combined, all_train_data_combined,
                all_shap_test_combined, all_test_data_combined,
                feature_names, importance_train, importance_test,
                show_train_plot=show_train_plot, show_test_plot=show_test_plot,
                model_name=model_name, data_sample=data_sample, region=region
            )
        
        print(f"  [{model_name.upper()}] SHAP Analysis Complete!\n")
        
        return SHAPResult(
            shap_train=all_shap_train_combined,
            X_train=all_train_data_combined[feature_names] if all_train_data_combined is not None else None,
            shap_test=all_shap_test_combined,
            X_test=all_test_data_combined[feature_names] if all_test_data_combined is not None else None,
            feature_names=feature_names,
            importance_train=importance_train,
            importance_test=importance_test,
            importance_df_combined=importance_df_combined,
            metadata={'model_name': model_name, 'region': region, 'data_sample': data_sample}
        )
