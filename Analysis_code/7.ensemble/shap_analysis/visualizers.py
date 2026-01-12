"""SHAP 시각화 및 KDE 플롯 함수."""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from .analyzers import ImportanceData
from .config import SHAPConfig
from .data_loader import DataLoader
from .preprocessors import PreprocessingUtils


class SHAPVisualizer:
    """SHAP 시각화를 담당하는 클래스."""
    
    # Feature 이름 매핑 딕셔너리 (plot 표시용)
    FEATURE_NAME_MAPPING = {
        'hm': 'relative_humidity',
        'PM25': 'PM2.5'
    }
    
    def __init__(self, config: SHAPConfig):
        self.config = config
    
    def _calculate_xlim(
        self,
        shap_train: Optional[np.ndarray],
        shap_test: Optional[np.ndarray]
    ) -> tuple[Optional[float], Optional[float]]:
        """x축 범위 계산."""
        if shap_train is not None and shap_test is not None:
            x_min = min(shap_train.min(), shap_test.min())
            x_max = max(shap_train.max(), shap_test.max())
        elif shap_train is not None:
            x_min, x_max = shap_train.min(), shap_train.max()
        elif shap_test is not None:
            x_min, x_max = shap_test.min(), shap_test.max()
        else:
            return None, None
        
        margin = (x_max - x_min) * self.config.DEFAULT_PLOT_MARGIN
        x_min -= margin
        x_max += margin
        return x_min, x_max
    
    def _get_feature_names_with_share(
        self,
        feature_names: List[str],
        importance_data: Optional[ImportanceData]
    ) -> List[str]:
        """Feature names에 영향도 퍼센트를 추가합니다."""
        if importance_data is None or importance_data.df is None:
            # 매핑만 적용 (share 없이)
            return [self.FEATURE_NAME_MAPPING.get(f, f) for f in feature_names]
        
        # importance_df는 이미 importance 순으로 정렬되어 있음
        importance_df = importance_data.df.copy()
        total_importance = importance_df['importance'].sum()
        
        if total_importance == 0:
            # 매핑만 적용 (share 없이)
            return [self.FEATURE_NAME_MAPPING.get(f, f) for f in feature_names]
        
        # feature -> share 매핑 생성
        feature_to_share = {}
        for _, row in importance_df.iterrows():
            feature = row['feature']
            share = (row['importance'] / total_importance) * 100
            feature_to_share[feature] = share
        
        # feature_names 순서를 유지하면서 share 추가 및 이름 매핑 적용
        result_names = []
        for feature in importance_df['feature'].tolist():
            # feature 이름 매핑 적용 (hm -> relative_humidity)
            display_name = self.FEATURE_NAME_MAPPING.get(feature, feature)
            
            if feature in feature_to_share:
                share = feature_to_share[feature]
                result_names.append(f"{display_name} ({share:.1f}%)")
            else:
                result_names.append(display_name)
        
        return result_names
    
    def _reorder_by_importance(
        self,
        shap_values: np.ndarray,
        data: pd.DataFrame,
        feature_names: List[str],
        importance_data: Optional[ImportanceData]
    ) -> tuple[np.ndarray, pd.DataFrame, List[str]]:
        """영향도 순으로 SHAP 값, 데이터, feature names를 재정렬합니다.
        
        Args:
            shap_values: SHAP 값 배열 (n_samples, n_features)
            data: 데이터프레임
            feature_names: Feature 이름 리스트
            importance_data: 영향도 데이터
        
        Returns:
            재정렬된 (shap_values, data, feature_names)
        """
        if importance_data is None or importance_data.df is None:
            return shap_values, data, feature_names
        
        # importance_df는 이미 importance 순으로 정렬되어 있음
        sorted_features = importance_data.df['feature'].tolist()
        
        # feature_names에 있는 feature만 필터링 (순서 유지)
        sorted_features = [f for f in sorted_features if f in feature_names]
        
        # 원본 feature_names에서 인덱스 찾기
        feature_indices = [feature_names.index(f) for f in sorted_features]
        
        # SHAP 값 재정렬
        shap_values_reordered = shap_values[:, feature_indices]
        
        # 데이터 재정렬
        if isinstance(data, pd.DataFrame):
            data_reordered = data[sorted_features]
        else:
            data_reordered = data[:, feature_indices]
        
        return shap_values_reordered, data_reordered, sorted_features
    
    def _create_title_prefix(
        self,
        model_name: Optional[str],
        data_sample: Optional[str],
        region: Optional[str]
    ) -> str:
        """플롯 제목 prefix 생성."""
        title_parts = []
        if model_name is not None:
            title_parts.append(model_name.upper())
        if data_sample is not None:
            title_parts.append(data_sample)
        if region is not None:
            title_parts.append(region.upper())
        return " - ".join(title_parts) + "\n" if title_parts else ""
    
    def _create_single_plot(
        self,
        shap_values: np.ndarray,
        data: pd.DataFrame,
        feature_names: List[str],
        feature_names_with_share: List[str],
        importance_data: Optional[ImportanceData],
        title_prefix: str,
        period_name: str,
        x_min: Optional[float],
        x_max: Optional[float]
    ) -> None:
        """단일 SHAP summary plot 생성 (영향도 순으로 정렬)."""
        # 폰트 크기 설정
        plt.rcParams.update({'font.size': 18})  # 기본 폰트 크기 증가
        
        plt.figure(figsize=(self.config.figsize[0]//2, self.config.figsize[1]))
        
        # 영향도 순으로 재정렬
        shap_values_sorted, data_sorted, feature_names_sorted = self._reorder_by_importance(
            shap_values, data, feature_names, importance_data
        )
        
        # feature_names_with_share도 같은 순서로 재정렬
        feature_names_with_share_sorted = self._get_feature_names_with_share(
            feature_names_sorted, importance_data
        )
        
        data_subset = data_sorted[feature_names_sorted] if isinstance(data_sorted, pd.DataFrame) else data_sorted
        shap.summary_plot(
            shap_values_sorted, data_subset, 
            feature_names=feature_names_with_share_sorted, 
            show=False,
            sort=False  # 이미 정렬했으므로 SHAP의 자동 정렬 비활성화
        )
        if x_min is not None and x_max is not None:
            plt.xlim(x_min, x_max)
        
        # 축 레이블 및 tick 폰트 크기 조절
        ax = plt.gca()
        ax.tick_params(labelsize=18)  # x축, y축 tick label 크기
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # x축 레이블 크기
        
        # colorbar 및 모든 axes의 폰트 크기 조절
        fig = plt.gcf()
        for ax_fig in fig.axes:
            ax_fig.tick_params(labelsize=18)  # 모든 axes의 tick label 크기 조절
            # y축 레이블(특징 이름) 폰트 크기 조절
            if ax_fig.get_ylabel():
                ax_fig.set_ylabel(ax_fig.get_ylabel(), fontsize=13)
        
        share_text = (
            f"\nTop {self.config.top_n} Share: {importance_data.top_n_share:.1f}%"
            if importance_data else ""
        )
        period_label = "Train Period (2018-20)" if "Train" in period_name else "Test Period (2021)"
        plt.title(f"{title_prefix}{period_label}: SHAP for Fog (Class 0 + Class 1){share_text}", fontsize=20)
        plt.tight_layout()
        plt.show()
    
    def _create_dual_plots(
        self,
        shap_train: Optional[np.ndarray],
        X_train: Optional[pd.DataFrame],
        shap_test: Optional[np.ndarray],
        X_test: Optional[pd.DataFrame],
        feature_names: List[str],
        feature_names_train: List[str],
        feature_names_test: List[str],
        importance_train: Optional[ImportanceData],
        importance_test: Optional[ImportanceData],
        title_prefix: str,
        x_min: Optional[float],
        x_max: Optional[float],
        show_train_plot: bool,
        show_test_plot: bool
    ) -> None:
        """이중 SHAP summary plot 생성 (영향도 순으로 정렬)."""
        # 폰트 크기 설정
        plt.rcParams.update({'font.size': 18})  # 기본 폰트 크기 증가
        
        fig, axes = plt.subplots(1, 2, figsize=self.config.figsize)
        
        if show_train_plot and shap_train is not None:
            plt.sca(axes[0])
            
            # 영향도 순으로 재정렬
            shap_train_sorted, data_train_sorted, feature_names_sorted = self._reorder_by_importance(
                shap_train, X_train, feature_names, importance_train
            )
            
            # feature_names_with_share도 같은 순서로 재정렬
            feature_names_train_sorted = self._get_feature_names_with_share(
                feature_names_sorted, importance_train
            )
            
            data_train = data_train_sorted[feature_names_sorted] if isinstance(data_train_sorted, pd.DataFrame) else data_train_sorted
            shap.summary_plot(
                shap_train_sorted, data_train,
                feature_names=feature_names_train_sorted,
                show=False, plot_size=None,
                sort=False  # 이미 정렬했으므로 SHAP의 자동 정렬 비활성화
            )
            if x_min is not None and x_max is not None:
                plt.xlim(x_min, x_max)
            
            # 축 레이블 및 tick 폰트 크기 조절
            ax = plt.gca()
            ax.tick_params(labelsize=18)  # x축, y축 tick label 크기
            ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # x축 레이블 크기
            
            # colorbar 및 모든 axes의 폰트 크기 조절
            fig = plt.gcf()
            for ax_fig in fig.axes:
                ax_fig.tick_params(labelsize=18)  # 모든 axes의 tick label 크기 조절
                # y축 레이블(특징 이름) 폰트 크기 조절
                if ax_fig.get_ylabel():
                    ax_fig.set_ylabel(ax_fig.get_ylabel(), fontsize=18)
            
            share_text = (
                f"\nTop {self.config.top_n} Share: {importance_train.top_n_share:.1f}%"
                if importance_train else ""
            )
            plt.title(f"{title_prefix}Train Period (2018-20): SHAP for Fog (Class 0 + Class 1){share_text}", fontsize=20)
        
        if show_test_plot and shap_test is not None:
            plt.sca(axes[1])
            
            # 영향도 순으로 재정렬
            shap_test_sorted, data_test_sorted, feature_names_sorted = self._reorder_by_importance(
                shap_test, X_test, feature_names, importance_test
            )
            
            # feature_names_with_share도 같은 순서로 재정렬
            feature_names_test_sorted = self._get_feature_names_with_share(
                feature_names_sorted, importance_test
            )
            
            data_test = data_test_sorted[feature_names_sorted] if isinstance(data_test_sorted, pd.DataFrame) else data_test_sorted
            shap.summary_plot(
                shap_test_sorted, data_test,
                feature_names=feature_names_test_sorted,
                show=False, plot_size=None,
                sort=False  # 이미 정렬했으므로 SHAP의 자동 정렬 비활성화
            )
            if x_min is not None and x_max is not None:
                plt.xlim(x_min, x_max)
            
            # 축 레이블 및 tick 폰트 크기 조절
            ax = plt.gca()
            ax.tick_params(labelsize=18)  # x축, y축 tick label 크기
            ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # x축 레이블 크기
            
            # colorbar 및 모든 axes의 폰트 크기 조절
            fig = plt.gcf()
            for ax_fig in fig.axes:
                ax_fig.tick_params(labelsize=18)  # 모든 axes의 tick label 크기 조절
                # y축 레이블(특징 이름) 폰트 크기 조절
                if ax_fig.get_ylabel():
                    ax_fig.set_ylabel(ax_fig.get_ylabel(), fontsize=18)
            
            share_text = (
                f"\nTop {self.config.top_n} Share: {importance_test.top_n_share:.1f}%"
                if importance_test else ""
            )
            plt.title(f"{title_prefix}Test Period (2021): SHAP for Fog (Class 0 + Class 1){share_text}", fontsize=20)
        
        plt.tight_layout()
        plt.show()
    
    def plot_summary(
        self,
        shap_train: Optional[np.ndarray],
        X_train: Optional[pd.DataFrame],
        shap_test: Optional[np.ndarray],
        X_test: Optional[pd.DataFrame],
        feature_names: List[str],
        importance_train: Optional[ImportanceData],
        importance_test: Optional[ImportanceData],
        show_train_plot: bool = True,
        show_test_plot: bool = True,
        model_name: Optional[str] = None,
        data_sample: Optional[str] = None,
        region: Optional[str] = None
    ) -> None:
        """SHAP summary plot 생성."""
        num_plots = int(show_train_plot and shap_train is not None) + int(show_test_plot and shap_test is not None)
        if num_plots == 0:
            return
        
        x_min, x_max = self._calculate_xlim(shap_train, shap_test)
        
        # Feature names with share 준비
        feature_names_train = self._get_feature_names_with_share(feature_names, importance_train)
        feature_names_test = self._get_feature_names_with_share(feature_names, importance_test)
        
        # Title prefix 생성
        title_prefix = self._create_title_prefix(model_name, data_sample, region)
        
        if num_plots == 2:
            self._create_dual_plots(
                shap_train, X_train, shap_test, X_test,
                feature_names, feature_names_train, feature_names_test,
                importance_train, importance_test,
                title_prefix, x_min, x_max,
                show_train_plot, show_test_plot
            )
        else:
            if show_train_plot and shap_train is not None:
                data = X_train[feature_names] if isinstance(X_train, pd.DataFrame) else X_train
                self._create_single_plot(
                    shap_train, data, feature_names, feature_names_train,
                    importance_train, title_prefix, "Train Period",
                    x_min, x_max
                )
            elif show_test_plot and shap_test is not None:
                data = X_test[feature_names] if isinstance(X_test, pd.DataFrame) else X_test
                self._create_single_plot(
                    shap_test, data, feature_names, feature_names_test,
                    importance_test, title_prefix, "Test Period",
                    x_min, x_max
                )
    
    def plot_elbow(self, importance_df_combined: pd.DataFrame) -> None:
        """Elbow plot 생성."""
        elbow = importance_df_combined.loc[:, ['feature', 'importance_train', 'share_train']]
        elbow = elbow.sort_values('importance_train', ascending=False)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(elbow) + 1), elbow['share_train'].values, marker='o')
        plt.xlabel('Feature Rank')
        plt.ylabel('Share Train (%)')
        plt.title('Elbow Plot for Feature Importance (share_train)')
        plt.xticks(range(1, len(elbow) + 1))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

