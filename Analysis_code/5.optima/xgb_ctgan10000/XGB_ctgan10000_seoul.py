import pandas as pd
import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from hyperopt import fmin, tpe, Trials, hp

# 상수 정의
RANDOM_STATE = 42
N_ESTIMATORS = 4000
EARLY_STOPPING_ROUNDS = 400
MAX_EVALS = 100

# Fold 설정: (train_years, val_year)
FOLD_CONFIGS = [
    ([2018, 2019], 2020),  # Fold 1
    ([2018, 2020], 2019),  # Fold 2
    ([2019, 2020], 2018),  # Fold 3
]

def calculate_csi(Y_test, pred):
    """CSI(Critical Success Index) 점수를 계산합니다.
    
    Args:
        Y_test: 실제 레이블
        pred: 예측 레이블
        
    Returns:
        CSI 점수 (0~1 사이의 값)
    """
    cm = confusion_matrix(Y_test, pred)
    H = (cm[0, 0] + cm[1, 1])
    F = (cm[1, 0] + cm[2, 0] + cm[0, 1] + cm[2, 1])
    M = (cm[0, 2] + cm[1, 2])
    CSI = H / (H + F + M + 1e-10)
    return CSI

def eval_metric_csi(y_true, pred_prob):
    """XGBoost용 CSI 메트릭 함수.
    
    Args:
        y_true: 실제 레이블
        pred_prob: 예측 확률
        
    Returns:
        CSI 점수의 음수값
    """
    pred = np.argmax(pred_prob, axis=1)
    csi = calculate_csi(y_true, pred)
    return -1*csi

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

def create_xgb_model(search_space=None, best_params=None):
    """XGBoost 모델을 생성합니다.
    
    Args:
        search_space: 하이퍼파라미터 검색 공간 (objective_func에서 사용)
        best_params: 최적화된 하이퍼파라미터 (최종 모델 학습에서 사용)
        
    Returns:
        XGBClassifier 인스턴스
    """
    base_params = {
        'n_estimators': N_ESTIMATORS,
        'tree_method': 'hist',
        'device': 'cuda',
        'enable_categorical': True,
        'eval_metric': eval_metric_csi,
        'objective': 'multi:softprob',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
    }
    
    if search_space is not None:
        # 하이퍼파라미터 최적화 중
        params = {
            **base_params,
            'learning_rate': search_space['learning_rate'],
            'max_depth': int(search_space['max_depth']),
            'min_child_weight': int(search_space['min_child_weight']),
            'gamma': search_space['gamma'],
            'subsample': search_space['subsample'],
            'colsample_bytree': search_space['colsample_bytree'],
            'reg_alpha': search_space['reg_alpha'],
            'reg_lambda': search_space['reg_lambda'],
        }
    elif best_params is not None:
        # 최적화된 파라미터로 최종 모델 생성
        params = {
            **base_params,
            'learning_rate': best_params['learning_rate'],
            'max_depth': int(best_params['max_depth']),
            'min_child_weight': int(best_params['min_child_weight']),
            'gamma': best_params['gamma'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
        }
    else:
        params = base_params
    
    return XGBClassifier(**params)


# 데이터 로딩 및 전처리
print("데이터 로딩 중...")
# 파일 위치 기반으로 데이터 디렉토리 경로 설정
current_file_dir = os.path.dirname(os.path.abspath(__file__))
data_base_dir = os.path.abspath(os.path.join(current_file_dir, '../../../data'))
df_seoul = pd.read_csv(os.path.join(data_base_dir, "data_for_modeling/seoul_train.csv"))
df_ctgan_seoul_1 = pd.read_csv(os.path.join(data_base_dir, "data_oversampled/ctgan10000/ctgan10000_1_seoul.csv"))
df_ctgan_seoul_2 = pd.read_csv(os.path.join(data_base_dir, "data_oversampled/ctgan10000/ctgan10000_2_seoul.csv"))
df_ctgan_seoul_3 = pd.read_csv(os.path.join(data_base_dir, "data_oversampled/ctgan10000/ctgan10000_3_seoul.csv"))

print("데이터 전처리 중...")
df_ctgan_seoul_1 = preprocessing(df_ctgan_seoul_1)
df_ctgan_seoul_2 = preprocessing(df_ctgan_seoul_2)
df_ctgan_seoul_3 = preprocessing(df_ctgan_seoul_3)
df_seoul = preprocessing(df_seoul)

# CTGAN 데이터 리스트 (fold 순서와 일치)
df_ctgan_list = [df_ctgan_seoul_1, df_ctgan_seoul_2, df_ctgan_seoul_3]

def split_data(df_sampled, df_original, train_years, val_year):
    """데이터를 학습용과 검증용으로 분할합니다.
    
    Args:
        df_sampled: 샘플링된 데이터프레임
        df_original: 원본 데이터프레임
        train_years: 학습에 사용할 연도 리스트
        val_year: 검증에 사용할 연도
        
    Returns:
        (X_train, X_val, y_train, y_val) 튜플
    """
    # 학습 데이터: 샘플링된 데이터에서 train_years에 해당하는 데이터
    train_mask = df_sampled['year'].isin(train_years)
    X_train = df_sampled.loc[train_mask, df_sampled.columns != 'multi_class'].copy()
    y_train = df_sampled.loc[train_mask, 'multi_class']
    
    # 검증 데이터: 원본 데이터에서 val_year에 해당하는 데이터
    val_mask = df_original['year'] == val_year
    X_val = df_original.loc[val_mask, df_original.columns != 'multi_class'].copy()
    y_val = df_original.loc[val_mask, 'multi_class']
    
    X_train.drop(columns=['year'], inplace=True)
    X_val.drop(columns=['year'], inplace=True)
    
    return X_train, X_val, y_train, y_val


# 하이퍼파라미터 검색 공간 정의
xgb_search_space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth': hp.quniform('max_depth', 3, 12, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'gamma': hp.uniform('gamma', 0, 5),  # 트리 분할을 위한 최소 손실 감소 값
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)
}

def objective_func(search_space):
    """하이퍼파라미터 최적화를 위한 목적 함수.
    
    Args:
        search_space: 하이퍼파라미터 검색 공간
        
    Returns:
        평균 CSI 점수의 음수값 (hyperopt는 최소화를 수행하므로)
    """
    xgb_model = create_xgb_model(search_space=search_space)
    csi_scores = []

    # 각 fold에 대해 교차 검증 수행
    for df_sampled, (train_years, val_year) in zip(df_ctgan_list, FOLD_CONFIGS):
        X_train, X_val, y_train, y_val = split_data(
            df_sampled, df_seoul, train_years, val_year
        )
        
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        csi = calculate_csi(y_val, xgb_model.predict(X_val))
        csi_scores.append(csi)

    return -1*round(np.mean(csi_scores), 4)


# 하이퍼파라미터 최적화
print("하이퍼파라미터 최적화 시작...")
trials = Trials()
xgb_best = fmin(
    fn=objective_func,
    space=xgb_search_space,
    algo=tpe.suggest,
    max_evals=MAX_EVALS, 
    trials=trials
)

# 최적화 결과 분석 및 출력
print(f"\n최적화 완료. 최적 파라미터: {xgb_best}")

# Best loss (CSI 점수의 음수값이므로, 실제 CSI는 -loss)
best_loss = trials.best_trial['result']['loss']
best_csi = -best_loss
print(f"최적 CSI 점수: {best_csi:.4f} (loss: {best_loss:.4f})")

# 모든 trial의 loss 값 추출
losses = [trial['result']['loss'] for trial in trials.trials if trial['result']['status'] == 'ok']
csi_scores = [-loss for loss in losses]

print(f"\n최적화 과정 요약:")
print(f"  - 총 시도 횟수: {len(trials.trials)}")
print(f"  - 성공한 시도: {len(losses)}")
print(f"  - 최초 CSI: {csi_scores[0]:.4f}")
print(f"  - 최종 CSI: {csi_scores[-1]:.4f}")
print(f"  - 최고 CSI: {max(csi_scores):.4f}")
print(f"  - 최저 CSI: {min(csi_scores):.4f}")
print(f"  - 평균 CSI: {np.mean(csi_scores):.4f}")

# Trials 객체 저장
# 파일 위치 기반으로 base 디렉토리 경로 설정
current_file_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_file_dir))  # 5.optima 디렉토리
os.makedirs(os.path.join(base_dir, "optimization_history"), exist_ok=True)
trials_path = os.path.join(base_dir, "optimization_history/xgb_ctgan10000_seoul_trials.pkl")
joblib.dump(trials, trials_path)
print(f"\n최적화 Trials 객체가 {trials_path}에 저장되었습니다.")

# 최적화된 하이퍼파라미터로 최종 모델 학습
print("최종 모델 학습 시작...")
models = []

for fold_idx, (df_sampled, (train_years, val_year)) in enumerate(
    zip(df_ctgan_list, FOLD_CONFIGS), start=1
):
    print(f"Fold {fold_idx} 학습 중... (학습 연도: {train_years}, 검증 연도: {val_year})")
    
    X_train, X_val, y_train, y_val = split_data(
        df_sampled, df_seoul, train_years, val_year
    )
    
    xgb_model = create_xgb_model(best_params=xgb_best)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 검증 성능 출력
    val_csi = calculate_csi(y_val, xgb_model.predict(X_val))
    print(f"Fold {fold_idx} 검증 CSI: {val_csi:.4f}")
    
    models.append(xgb_model)


# 모델 저장
print("모델 저장 중...")
os.makedirs(os.path.join(base_dir, "save_model/xgb_optima"), exist_ok=True)
model_save_path = os.path.join(base_dir, "save_model/xgb_optima/xgb_ctgan10000_seoul.pkl")
joblib.dump(models, model_save_path)
print(f"모델이 {model_save_path}에 저장되었습니다.")

