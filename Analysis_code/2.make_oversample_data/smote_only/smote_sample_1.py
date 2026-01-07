import pandas as pd
import numpy as np
from pathlib import Path

from imblearn.over_sampling import SMOTENC

# 지역별 데이터 파일 경로
regions = ['incheon', 'seoul','busan', 'daegu', 'daejeon', 'gwangju']
file_paths = [f'../../../data/data_for_modeling/{region}_train.csv' for region in regions]
output_paths = [f'../../../data/data_oversampled/smote/smote_1_{region}.csv' for region in regions]

# 지역별 처리
for file_path, output_path in zip(file_paths, output_paths):
    # 데이터 로드
    original_data = pd.read_csv(file_path, index_col=0)
    data = original_data.loc[original_data['year'].isin([2018, 2019]), :]
    data['cloudcover'] = data['cloudcover'].astype('int')
    data['lm_cloudcover'] = data['lm_cloudcover'].astype('int')
    X = data.drop(columns=['multi_class', 'binary_class'])
    y = data['multi_class']

    # 불필요한 열 제거
    X.drop(columns=['ground_temp - temp_C', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'], inplace=True)

    # SMOTENC에서 사용할 범주형 변수 열 번호 설정
    categorical_features_indices = [i for i, dtype in enumerate(X.dtypes) if dtype != 'float64']

    # sampling_strategy 설정
    count_class_2 = (y == 2).sum()
    sampling_strategy = {
        0: int(np.ceil(count_class_2 / 1000) * 500),
        1: int(np.ceil(count_class_2 / 1000) * 500),
        2: count_class_2
    }

    # SMOTENC 적용
    smotenc = SMOTENC(categorical_features=categorical_features_indices, sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smotenc.fit_resample(X, y)

    # Resampled 데이터 생성
    lerp_data = X_resampled.copy()
    lerp_data['multi_class'] = y_resampled

    # 제거변수 복구
    lerp_data['binary_class'] = lerp_data['multi_class'].apply(lambda x: 0 if x == 2 else 1)
    lerp_data['hour_sin'] = np.sin(2 * np.pi * lerp_data['hour'] / 24)
    lerp_data['hour_cos'] = np.cos(2 * np.pi * lerp_data['hour'] / 24)
    lerp_data['month_sin'] = np.sin(2 * np.pi * lerp_data['month'] / 12)
    lerp_data['month_cos'] = np.cos(2 * np.pi * lerp_data['month'] / 12)
    lerp_data['ground_temp - temp_C'] = lerp_data['groundtemp'] - lerp_data['temp_C']

    # 증강된 데이터만 저장 (SMOTENC으로 증강된 부분만)
    # lerp_data의 처음 len(X)개는 원본 데이터이므로 제외
    original_data_count = len(X)
    augmented_only = lerp_data.iloc[original_data_count:].copy()  # SMOTENC으로 증강된 부분만
    augmented_only = augmented_only[augmented_only['multi_class'] != 2].copy()  # 클래스 2 제외
    augmented_only.reset_index(drop=True, inplace=True)
    # augmented_only 폴더에 저장
    output_path_obj = Path(output_path)
    augmented_dir = output_path_obj.parent.parent / 'augmented_only'
    augmented_dir.mkdir(parents=True, exist_ok=True)
    augmented_output_path = augmented_dir / output_path_obj.name
    augmented_only.to_csv(augmented_output_path, index=False)
    
    # 증강된 데이터만 결과 출력
    aug_count_0 = len(augmented_only[augmented_only['multi_class'] == 0])
    aug_count_1 = len(augmented_only[augmented_only['multi_class'] == 1])
    print(f"Saved augmented data only {augmented_output_path}: Class 0={aug_count_0} | Class 1={aug_count_1}")

    # 클래스 2 제거 후 원본 클래스 2 데이터 추가
    filtered_data = lerp_data[lerp_data['multi_class'] != 2]
    original_class_2 = data[data['multi_class'] == 2]
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