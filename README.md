### Visibility Prediction Modeling Project

This project predicts visibility (`visi`) by integrating meteorological and air pollution data (ASOS, DataOn). It addresses class imbalance using SMOTENC/CTGAN augmentation and performs multi-class/binary classification by combining GBDT models (LightGBM/XGBoost) with tabular deep learning architectures (ResNet-like, FT-Transformer, DeepGBM).

### Tech Stack

- **Development Environment**: Docker (`teddylee777/deepko:preview`)
- **Data Processing**: `pandas`, `numpy`
- **Data Preprocessing**: `scikit-learn` (`QuantileTransformer`, `LabelEncoder`)
- **EDA/Visualization**: `matplotlib`, `seaborn`
- **Sampling/Imbalance Handling**: `imbalanced-learn (SMOTENC)`, `CTGAN`, `Optuna` (CTGAN hyperparameters), region/year-based splitting
- **Modeling (GBDT)**: `LightGBM`, `XGBoost` (with GPU support, custom CSI evaluation)
- **Modeling (Deep Learning)**: `PyTorch`-based `ResNetLike`, `FTTransformer`, `DeepGBM`
- **Optimization**: `hyperopt` (LightGBM/XGBoost), `Optuna` (deep learning models, CTGAN)
- **Utilities/Storage**: `joblib`

### System Architecture (Pipeline)

1) Data Collection/Loading: `data/ASOS`, `data/dataon`
2) Merging/Preprocessing: `1.data_preprocessing/0.air_data_merge.ipynb` → `1.data_preprocessing/1.data_merge.ipynb` → `1.data_preprocessing/2.eda_preproccesing.ipynb` → `1.data_preprocessing/3.make_train_test.ipynb`
3) Data Augmentation (Imbalance Handling): `SMOTENC` → `CTGAN(+Optuna)` → Rule-based filtering in `2.make_oversample_data/`
4) Data Splitting: By region (`*_train.csv`, `*_test.csv`), year-based 3-fold holdout
5) Training: GBDT (`5.optima/*/`) and deep learning notebooks
6) Evaluation/Analysis: Custom `CSI` + F1/Accuracy, `visualization/model_visualize.ipynb`, `find_reason/*` (trend and distribution comparison)
7) Ensemble/Final: `model_voting_test_best_sample/ensemble__voting_best_sample.ipynb`, `final_test/final.ipynb`

### Quick Start Guide

1) Docker Environment Setup

This project was developed using the `teddylee777/deepko:preview` Docker image.

```bash
# Pull Docker image
docker pull teddylee777/deepko:preview

# Run container (with GPU)
docker run --gpus all -it -v $(pwd):/workspace teddylee777/deepko:preview

# Run container (CPU only)
docker run -it -v $(pwd):/workspace teddylee777/deepko:preview
```

The Docker image includes PyTorch, CUDA, and all required packages (LightGBM, XGBoost, scikit-learn, imbalanced-learn, Optuna, CTGAN, etc.) pre-installed.

2) Data Preparation
- Place raw/intermediate outputs in the `data/` directory. Training CSV/feather files are located in `data/data_for_modeling/`.
- Alternatively, download the `data/` folder from the Hugging Face repository:
```bash
git clone https://huggingface.co/bong9513/visibility_prediction
# After cloning, copy the visibility_prediction/data/ folder to the project's data/ location
```

3) Perform Oversampling (SMOTE/CTGAN)

```bash
cd Analysis_code/2.make_oversample_data
# For SMOTE only
python smote_only/smote_sample_1.py
# For SMOTENC + CTGAN
python smotenc_ctgan/smotenc_ctgan_sample_10000_1.py
```

4) Model Training or Download
   - **Option A: Train Models from Scratch**
     - GBDT optimization/training example (Seoul):
     ```bash
     cd Analysis_code/5.optima
     python lgb_smote/LGB_smote_seoul.py
     python xgb_smote/XGB_smote_seoul.py
     ```
     - Deep learning model training/evaluation: Run notebooks (`.ipynb` files in `Analysis_code/`)
   - **Option B: Use Pre-trained Models**
     - Download pre-trained models from Hugging Face repository:
     ```bash
     git clone https://huggingface.co/bong9513/visibility_prediction
     # After cloning, copy visibility_prediction/save_model/ folder to Analysis_code/save_model/
     ```

---

### Project Structure

```
visibility_prediction/
├── Analysis_code/
│   ├── 1.data_preprocessing/       # Data merging and preprocessing
│   │   ├── 0.air_data_merge.ipynb
│   │   ├── 1.data_merge.ipynb
│   │   ├── 2.eda_preproccesing.ipynb
│   │   └── 3.make_train_test.ipynb
│   ├── 2.make_oversample_data/     # Oversampling (SMOTE/CTGAN)
│   │   ├── smote_only/            # SMOTE only
│   │   ├── only_ctgan/            # CTGAN only
│   │   └── smotenc_ctgan/         # SMOTENC + CTGAN combination
│   ├── 3.sampled_data_analysis/    # Sampled data analysis
│   ├── 4.sampling_data_test/       # Sampling data performance testing
│   ├── 5.optima/                   # Model optimization and training
│   │   ├── lgb_smote/             # LightGBM (SMOTE)
│   │   ├── lgb_pure/              # LightGBM (original data)
│   │   ├── lgb_ctgan10000/        # LightGBM (CTGAN 10000)
│   │   ├── lgb_smotenc_ctgan20000/ # LightGBM (SMOTENC+CTGAN 20000)
│   │   ├── xgb_smote/             # XGBoost (SMOTE)
│   │   ├── xgb_pure/              # XGBoost (original data)
│   │   ├── xgb_ctgan10000/        # XGBoost (CTGAN 10000)
│   │   ├── xgb_smotenc_ctgan20000/ # XGBoost (SMOTENC+CTGAN 20000)
│   │   ├── resnet_like_smote/     # ResNet-like (SMOTE)
│   │   ├── resnet_like_pure/      # ResNet-like (original data)
│   │   ├── resnet_like_ctgan10000/ # ResNet-like (CTGAN 10000)
│   │   ├── resnet_like_smotenc_ctgan20000/ # ResNet-like (SMOTENC+CTGAN 20000)
│   │   ├── ft_transformer_smote/  # FT-Transformer (SMOTE)
│   │   ├── ft_transformer_pure/   # FT-Transformer (original data)
│   │   ├── ft_transformer_ctgan10000/ # FT-Transformer (CTGAN 10000)
│   │   ├── ft_transformer_smotenc_ctgan20000/ # FT-Transformer (SMOTENC+CTGAN 20000)
│   │   ├── deepgbm_smote/         # DeepGBM (SMOTE)
│   │   ├── deepgbm_pure/         # DeepGBM (original data)
│   │   ├── deepgbm_ctgan10000/    # DeepGBM (CTGAN 10000)
│   │   └── deepgbm_smotenc_ctgan20000/ # DeepGBM (SMOTENC+CTGAN 20000)
│   ├── 6.optima_models_analysis/   # Optimized model analysis
│   ├── models/                     # Deep learning model definitions and storage
│   │   ├── deepgbm.py
│   │   ├── ft_transformer.py
│   │   ├── resnet_like.py
│   │   ├── best_resnet_model.pth
│   │   └── tabnet_model.zip
│   ├── save_model/                 # Trained model storage (downloadable from Hugging Face)
│   ├── optimization_history/       # Optimization history (downloadable from Hugging Face)
│   ├── visualization/              # Model visualization
│   │   └── model_visualize.ipynb
│   ├── find_reason/                # Regional trend/causal analysis notebooks
│   ├── model_voting_test_best_sample/
│   │   └── ensemble__voting_best_sample.ipynb
│   └── final_test/
│       └── final.ipynb
├── data/
│   ├── ASOS/                       # Meteorological data
│   ├── dataon/                     # Air pollution data (large daily CSV files)
│   ├── data_for_modeling/          # Regional train/test CSV and feather files
│   ├── data_for_demo/
│   ├── data_oversampled/           # Oversampled data
│   │   ├── smote/
│   │   ├── ctgan7000/
│   │   ├── ctgan10000/
│   │   └── ctgan20000/
│   └── oversampled_data_test_for_model/ # Test oversampled data
└── README.md
```

---

### Data and Variables

- Target Variables
  - `visi`: Visibility (continuous value). Synthetic sample filtering rules use intervals: class 0 is [0,100), class 1 is [100,500), class 2 is other ranges.
  - `multi_class`: Multi-class classification label (integer 0/1/2)
  - `binary_class`: Binary label. Rule: `binary_class = 0 if multi_class == 2 else 1`

- Main Feature Groups (code-based)
  - Meteorological (ASOS): `temp_C`, `precip_mm`, `wind_speed`, `wind_dir` (calm→0 substitution), `hm`, `vap_pressure`, `dewpoint_C`, `loc_pressure`, `sea_pressure`, `solarRad`, `snow_cm`, `cloudcover` (int), `lm_cloudcover` (int), `low_cloudbase`, `groundtemp`
  - Air Pollution (DataOn): `O3`, `NO2`, `PM10`, `PM25`
  - Temporal/Cyclical: `year` (int), `month` (int), `hour` (int), `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
  - Derived: `ground_temp - temp_C` (ground-air temperature difference)

- Categorical Variables (model/sampling perspective)
  - `wind_dir`, `cloudcover`, `lm_cloudcover`, and `int` type temporal variables (`year`, `month`, `hour`) are treated as categorical in SMOTENC/GBDT (code automatically detects non-`float64` column indices)

- Preprocessing Rules (excerpt)
  - `wind_dir` with `'정온'` (calm) is replaced with "0" then converted to integer
  - `cloudcover, lm_cloudcover` converted to integer
  - Target/auxiliary columns (`multi_class, binary_class`) separated during training and recalculated as needed

---

### EDA and Preprocessing

- Merging/Cleaning
  - Index column removal: Drop `Unnamed: 0`
  - Data type consistency: `cloudcover`, `lm_cloudcover` as integer; `year`, `month`, `hour` as integer
  - Special value substitution: `wind_dir == '정온'` (calm) → "0" then convert to integer

- Feature Engineering
  - Cyclical encoding: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
  - Difference-based derivation: `ground_temp - temp_C`

- Data Scaling and Encoding
  - **Numerical Variable Scaling**: Using `sklearn.preprocessing.QuantileTransformer(output_distribution='normal')`
    - Fit on training data, then transform Val/Test
    - `random_state` not specified (default value used)
  - **Categorical Variable Encoding**: Using `sklearn.preprocessing.LabelEncoder`
    - `wind_dir`, `cloudcover`, `lm_cloudcover`, `year`, `month`, `hour`, etc.
    - Fit on training data, then transform Val/Test

- Distribution/Trend Analysis
  - Regional time series trends: `Analysis_code/find_reason/*_trend.ipynb` (seoul, incheon, busan, daegu, daejeon, gwangju)
  - Distribution comparison/change detection: `Analysis_code/find_reason/wasserstein_distance.ipynb` (quantifying distribution differences using Wasserstein distance)

- Data Splitting
  - Region-based datasets (`*_train.csv`, `*_test.csv`)
  - Year-based 3-fold holdout (2018–2020 combinations) for generalization performance validation

### Imbalance Handling and Synthetic Sampling

- SMOTENC
  - Categorical indices: Uses position indices of non-`float64` columns from input features
  - Sampling strategy examples: `{0: 10000, 1: 10000, 2: existing count}` or `{0: 500/1000, 1: ceil(n1/100)*100, 2: n2}` depending on data scale
  - Recalculation: After sampling, recover `binary_class` and cyclical/difference derivatives from `multi_class`

- CTGAN (+Optuna)
  - Targets classes 0 and 1, using Optuna to search `embedding_dim, generator_dim, discriminator_dim, pac, batch_size, discriminator_steps` before synthesis
  - Generated sample quality filter: `class 0 → 0 ≤ visi < 100`, `class 1 → 100 ≤ visi < 500`
  - After final merging, recover derived/auxiliary features (`binary_class`, cyclical/difference items)

- Output
  - Regional CSV files saved in `data/data_oversampled/smote/`, `data/data_oversampled/ctgan7000/`, `data/data_oversampled/ctgan10000/`, `data/data_oversampled/ctgan20000/`

---

### Model Architecture (Detailed)

- Deep Learning (Tabular)
  - `Analysis_code/models/resnet_like.py`
    - Input: `x_num [B, N_num]`, `x_cat [B, N_cat]` → concat → input linear (`d_main=128`) → residual blocks (`n_blocks=4`, `d_hidden=64`, `dropout_first=0.25`) → output layer
    - Output: `num_classes == 2 → 1 logit`, `> 2 → K logits`
  - `Analysis_code/models/ft_transformer.py`
    - Numerical: Linear (`num_features → d_token=192`), Categorical: `nn.Embedding(d_token)` per `cat_cardinalities` then concatenated
    - Encoder: `TransformerEncoderLayer(d_model=d_token, nhead=8, dropout≈0.2)` × `n_blocks=6` → average pooling → classification head
  - `Analysis_code/models/deepgbm.py`
    - Numerical Linear (`d_main=128`) + categorical embeddings summed → residual MLP blocks (`n_blocks=4`, `d_hidden=64`, `dropout≈0.2`) → classification head

- GBDT
  - LightGBM (`5.optima/lgb_smote/LGB_smote_seoul.py`): `objective='multiclassova'`, `n_estimators≈4000`, early stopping, GPU option available, `hyperopt` searches `max_depth, min_child_weight, num_leaves, subsample, learning_rate`
  - XGBoost (`5.optima/xgb_smote/XGB_smote_seoul.py`): `objective='multi:softprob'`, `tree_method='hist'`, `enable_categorical=True`, GPU option, `hyperopt` searches key hyperparameters, `eval_metric=CSI`

---

### Hyperparameter Optimization

All models optimize hyperparameters to maximize the CSI (Critical Success Index) score. GBDT models use `hyperopt` (TPE algorithm), while deep learning models use `Optuna` (TPE sampler).

#### LightGBM Hyperparameter Search Space

- **Optimization Library**: `hyperopt` (TPE algorithm)
- **Number of Trials**: `max_evals=100`
- **Evaluation Metric**: CSI (3-fold cross-validation average)
- **Search Space**:
  - `learning_rate`: `hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))` - log-uniform distribution, range [0.01, 0.2]
  - `max_depth`: `hp.quniform('max_depth', 3, 15, 1)` - integer uniform distribution, range [3, 15]
  - `num_leaves`: `hp.quniform('num_leaves', 20, 150, 1)` - integer uniform distribution, range [20, 150] (set smaller than 2^max_depth)
  - `min_child_weight`: `hp.quniform('min_child_weight', 1, 20, 1)` - integer uniform distribution, range [1, 20]
  - `subsample`: `hp.uniform('subsample', 0.6, 1.0)` - uniform distribution, range [0.6, 1.0]
  - `colsample_bytree`: `hp.uniform('colsample_bytree', 0.6, 1.0)` - uniform distribution, range [0.6, 1.0]
  - `reg_alpha`: `hp.uniform('reg_alpha', 0.0, 1.0)` - uniform distribution, range [0.0, 1.0] (L1 regularization)
  - `reg_lambda`: `hp.uniform('reg_lambda', 0.0, 1.0)` - uniform distribution, range [0.0, 1.0] (L2 regularization)
- **Fixed Parameters**: `n_estimators=4000`, `early_stopping_rounds=400`, `device='gpu'`, `objective='multiclassova'`, `random_state=42`

#### XGBoost Hyperparameter Search Space

- **Optimization Library**: `hyperopt` (TPE algorithm)
- **Number of Trials**: `max_evals=100`
- **Evaluation Metric**: CSI (3-fold cross-validation average, using custom `eval_metric_csi` function)
- **Search Space**:
  - `learning_rate`: `hp.loguniform('learning_rate', np.log(0.01), np.log(0.2))` - log-uniform distribution, range [0.01, 0.2]
  - `max_depth`: `hp.quniform('max_depth', 3, 12, 1)` - integer uniform distribution, range [3, 12]
  - `min_child_weight`: `hp.quniform('min_child_weight', 1, 20, 1)` - integer uniform distribution, range [1, 20]
  - `gamma`: `hp.uniform('gamma', 0, 5)` - uniform distribution, range [0, 5] (minimum loss reduction for tree splitting)
  - `subsample`: `hp.uniform('subsample', 0.6, 1.0)` - uniform distribution, range [0.6, 1.0]
  - `colsample_bytree`: `hp.uniform('colsample_bytree', 0.6, 1.0)` - uniform distribution, range [0.6, 1.0]
  - `reg_alpha`: `hp.uniform('reg_alpha', 0.0, 1.0)` - uniform distribution, range [0.0, 1.0] (L1 regularization)
  - `reg_lambda`: `hp.uniform('reg_lambda', 0.0, 1.0)` - uniform distribution, range [0.0, 1.0] (L2 regularization)
- **Fixed Parameters**: `n_estimators=4000`, `early_stopping_rounds=400`, `tree_method='hist'`, `device='cuda'`, `enable_categorical=True`, `objective='multi:softprob'`, `random_state=42`

#### FT-Transformer Hyperparameter Search Space

- **Optimization Library**: `Optuna` (TPE sampler)
- **Number of Trials**: `n_trials=100`
- **Pruning**: `MedianPruner(n_warmup_steps=10)` - observe first 10 epochs then prune
- **Evaluation Metric**: CSI (3-fold cross-validation average)
- **Search Space**:
  - `d_token`: `trial.suggest_int("d_token", 64, 256, step=32)` - integer, range [64, 256], step 32 (64, 96, 128, 160, 192, 224, 256)
  - `n_blocks`: `trial.suggest_int("n_blocks", 2, 6)` - integer, range [2, 6] (reduced depth to prevent overfitting)
  - `n_heads`: `trial.suggest_categorical("n_heads", [4, 8])` - categorical, choices [4, 8]
  - `attention_dropout`: `trial.suggest_float("attention_dropout", 0.1, 0.4)` - float, range [0.1, 0.4]
  - `ffn_dropout`: `trial.suggest_float("ffn_dropout", 0.1, 0.4)` - float, range [0.1, 0.4]
  - `lr` (learning_rate): `trial.suggest_float("lr", 1e-5, 1e-2, log=True)` - log-scale float, range [1e-5, 1e-2]
  - `weight_decay`: `trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)` - log-scale float, range [1e-4, 1e-1]
  - `batch_size`: `trial.suggest_categorical("batch_size", [32, 64, 128, 256])` - categorical, choices [32, 64, 128, 256]
- **Structural Constraint**: `d_token` must be divisible by `n_heads` (automatically adjusted in code)
- **Fixed Parameters**: `num_classes=3`, `optimizer='AdamW'`, `epochs=200`, `patience=12`, `scheduler='ReduceLROnPlateau'` (factor=0.5, patience=3), `random_state=42`

#### ResNet-like Hyperparameter Search Space

- **Optimization Library**: `Optuna` (TPE sampler)
- **Number of Trials**: `n_trials=100`
- **Pruning**: `MedianPruner(n_warmup_steps=10)` - observe first 10 epochs then prune
- **Evaluation Metric**: CSI (3-fold cross-validation average)
- **Search Space**:
  - `d_main`: `trial.suggest_int("d_main", 64, 256, step=32)` - integer, range [64, 256], step 32 (64, 96, 128, 160, 192, 224, 256)
  - `d_hidden`: `trial.suggest_int("d_hidden", 64, 512, step=64)` - integer, range [64, 512], step 64 (64, 128, 192, 256, 320, 384, 448, 512)
  - `n_blocks`: `trial.suggest_int("n_blocks", 2, 5)` - integer, range [2, 5] (controlled to avoid excessive depth)
  - `dropout_first`: `trial.suggest_float("dropout_first", 0.1, 0.4)` - float, range [0.1, 0.4]
  - `dropout_second`: `trial.suggest_float("dropout_second", 0.0, 0.2)` - float, range [0.0, 0.2]
  - `lr` (learning_rate): `trial.suggest_float("lr", 1e-5, 1e-2, log=True)` - log-scale float, range [1e-5, 1e-2]
  - `weight_decay`: `trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)` - log-scale float, range [1e-4, 1e-1]
  - `batch_size`: `trial.suggest_categorical("batch_size", [32, 64, 128, 256])` - categorical, choices [32, 64, 128, 256]
- **Fixed Parameters**: `num_classes=3`, `optimizer='AdamW'`, `epochs=200`, `patience=12`, `scheduler='ReduceLROnPlateau'` (factor=0.5, patience=3), `random_state=42`

#### DeepGBM Hyperparameter Search Space

- **Optimization Library**: `Optuna` (TPE sampler)
- **Number of Trials**: `n_trials=100`
- **Pruning**: `MedianPruner(n_warmup_steps=10)` - observe first 10 epochs then prune
- **Evaluation Metric**: CSI (3-fold cross-validation average)
- **Search Space**:
  - `d_main`: `trial.suggest_int("d_main", 64, 256, step=32)` - integer, range [64, 256], step 32 (64, 96, 128, 160, 192, 224, 256)
  - `d_hidden`: `trial.suggest_int("d_hidden", 64, 256, step=64)` - integer, range [64, 256], step 64 (64, 128, 192, 256)
  - `n_blocks`: `trial.suggest_int("n_blocks", 2, 6)` - integer, range [2, 6]
  - `dropout`: `trial.suggest_float("dropout", 0.1, 0.4)` - float, range [0.1, 0.4]
  - `lr` (learning_rate): `trial.suggest_float("lr", 1e-5, 1e-2, log=True)` - log-scale float, range [1e-5, 1e-2]
  - `weight_decay`: `trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)` - log-scale float, range [1e-4, 1e-1]
  - `batch_size`: `trial.suggest_categorical("batch_size", [32, 64, 128, 256])` - categorical, choices [32, 64, 128, 256]
- **Fixed Parameters**: `num_classes=3`, `optimizer='AdamW'`, `epochs=200`, `patience=12`, `scheduler='ReduceLROnPlateau'` (factor=0.5, patience=3), `random_state=42`

#### Common Optimization Settings

- **Cross-validation**: All models use year-based 3-fold holdout cross-validation
  - Fold 1: Train [2018, 2019] → Val 2020
  - Fold 2: Train [2018, 2020] → Val 2019
  - Fold 3: Train [2019, 2020] → Val 2018
- **Evaluation Metric**: CSI (Critical Success Index) - average CSI across all folds used as optimization objective
- **Optimization Algorithm**: TPE (Tree-structured Parzen Estimator)
- **Reproducibility**: `random_state=42` fixed

---

### Training/Validation Strategy

- Year-based 3-fold holdout (example)
  - Fold1: Train 2018–2019 → Val 2020
  - Fold2: Train 2018–2020 → Val 2019
  - Fold3: Train 2019–2020 → Val 2018
- Separate training by region (e.g., `seoul_train.csv`, etc.)

---

### Evaluation Metrics

- Custom CSI (Critical Success Index) multi-class version

```python
H = cm[0, 0] + cm[1, 1]
F = (cm[1, 0] + cm[2, 0] + cm[0, 1] + cm[2, 1])
M = (cm[0, 2] + cm[1, 2])
CSI = H / (H + F + M + 1e-10)
```

- Additional metrics: Accuracy, F1, etc. verified in notebooks/scripts

---

### Execution Methods (Detailed)

- Environment Setup
  - Use Docker image `teddylee777/deepko:preview` (see Quick Start Guide above)
  - Run container with `--gpus all` option for GPU usage
  - All required libraries pre-installed, no additional installation needed

- Data Preparation
  - `data/ASOS/`: Annual meteorological data
  - `data/dataon/`: Daily air pollution CSV files (large volume)
  - `data/data_for_modeling/`: Regional train/test sets (`*_train.csv`, `*_test.csv`, `df_*.feather`)
  - **Download from Hugging Face**: Full `data/` folder available from [Hugging Face repository](https://huggingface.co/bong9513/visibility_prediction/tree/main/data)
    ```bash
    git clone https://huggingface.co/bong9513/visibility_prediction
    # After cloning, copy visibility_prediction/data/ folder to project's data/ location
    ```

- Preprocessing/Exploration
  - `Analysis_code/1.data_preprocessing/0.air_data_merge.ipynb` → `1.data_preprocessing/1.data_merge.ipynb` → `1.data_preprocessing/2.eda_preproccesing.ipynb` → `1.data_preprocessing/3.make_train_test.ipynb`

- Oversampling
  - Run scripts in `Analysis_code/2.make_oversample_data/` (see Quick Start Guide above)

- GBDT Optimization/Training
  - **Option 1: Train Models from Scratch**
    - Run `Analysis_code/5.optima/lgb_smote/LGB_smote_seoul.py`, `5.optima/xgb_smote/XGB_smote_seoul.py` for model training
    - Output models saved as `.pkl` files in `Analysis_code/save_model/`
    - Regional scripts available for each model (seoul, incheon, busan, daegu, daejeon, gwangju)
  - **Option 2: Use Pre-trained Models**
    - Download pre-trained models and optimization history from Hugging Face repository
    - `save_model/`: Download pre-trained models from [Hugging Face repository](https://huggingface.co/bong9513/visibility_prediction/tree/main/save_model)
    - `optimization_history/`: Download optimization history files from [Hugging Face repository](https://huggingface.co/bong9513/visibility_prediction/tree/main/optimization_history)
    ```bash
    git clone https://huggingface.co/bong9513/visibility_prediction
    # After cloning, copy visibility_prediction/save_model/ and visibility_prediction/optimization_history/ folders 
    # to Analysis_code/save_model/ and Analysis_code/optimization_history/ respectively
    ```

- Deep Learning Training
  - **Option 1: Train Models from Scratch**
    - Run regional scripts in each model folder under `Analysis_code/5.optima/` (`resnet_like_*`, `ft_transformer_*`, `deepgbm_*`)
    - Example: `5.optima/resnet_like_smote/resnet_like_smote_seoul.py`
    - Model definitions located in `Analysis_code/models/` folder (`deepgbm.py`, `ft_transformer.py`, `resnet_like.py`)
    - Visualization: Use `Analysis_code/visualization/model_visualize.ipynb` for visualization
  - **Option 2: Use Pre-trained Models**
    - Download pre-trained deep learning models from `save_model/` folder in Hugging Face repository
    - Download model files from [Hugging Face repository](https://huggingface.co/bong9513/visibility_prediction/tree/main/save_model) and use

- Ensemble/Final Evaluation
  - `Analysis_code/model_voting_test_best_sample/ensemble__voting_best_sample.ipynb`
  - `Analysis_code/final_test/final.ipynb`

---

### Model I/O Specifications (Summary)

- Numerical input `x_num`: `float32` tensor `[batch, num_numeric_features]`
- Categorical input `x_cat`: integer index tensor `[batch, num_categorical_features]`
- Output: Binary (1 logit) or multi-class (K logits). Loss/threshold settings refer to notebooks

---

### Reproducibility/Seeds

- **Random State Configuration**: All models and sampling processes use `random_state=42` for reproducibility
  - SMOTENC: `random_state=42`
  - CTGAN: `random_state=42`
  - GBDT models (LightGBM, XGBoost): `random_state=42`
  - Deep learning models (FT-Transformer, ResNet-Like, DeepGBM): `random_state=42` (including PyTorch seed)
  - QuantileTransformer: `random_state` not specified (default value used)
- Reproducibility may vary due to data/hardware differences, so explicit fold/seed configuration is recommended
- For PyTorch, reproducibility is enhanced with `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `torch.backends.cudnn.deterministic=True` settings

---

### Notes/Troubleshooting

- **Docker Environment**: This project assumes use of the `teddylee777/deepko:preview` Docker image. All required libraries are pre-installed, no additional installation needed.
- **Path Configuration**: Scripts assume relative paths. Verify current working directory is correct before execution.
- **Data Preprocessing**: Missing `wind_dir` `'정온'` (calm) value substitution/type conversion may cause errors in GBDT/XGB.
- **Memory Management**: `dataon/` is very large. Ensure sufficient memory or process by year/region.
- **GPU Usage**: When using GPU, always include `--gpus all` option when running Docker container.

---

### Dependencies

This project uses the `teddylee777/deepko:preview` Docker image with the following libraries pre-installed:

- **Python**: 3.8+
- **Deep Learning**: PyTorch (with CUDA support)
- **Data Processing**: pandas, numpy, scikit-learn
- **Machine Learning**: LightGBM (with GPU support), XGBoost (with GPU support)
- **Sampling**: imbalanced-learn (SMOTENC), CTGAN
- **Optimization**: Optuna, hyperopt
- **Visualization**: matplotlib, seaborn
- **Utilities**: joblib

No additional package installation required; ready to run directly within Docker container.

