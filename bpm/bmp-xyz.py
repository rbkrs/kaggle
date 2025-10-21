# %%
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================
# General utilities
# =====================
import json
import os
import pickle
import time
from collections import Counter

# =====================
# Data handling & processing
# =====================
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================
# Visualization
# =====================
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Machine Learning - Core scikit-learn
# =====================
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
    root_mean_squared_error, roc_auc_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.svm import SVC, SVR

# =====================
# Machine Learning - Tree Boosting & advanced
# =====================
import xgboost as xg
import lightgbm as lgb
import catboost as cb

# =====================
# Deep Learning - TensorFlow / Keras
# =====================
import tensorflow as tf
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# =====================
# Imbalanced data handling
# =====================
from imblearn.over_sampling import SMOTE

# =====================
# Optimization / AutoML
# =====================
import optuna

# =====================
# Feature importance & explainability
# =====================
import shap

# =====================
# Self Made Utilities
# =====================
from utils import *

# =====================
# Settings & reproducibility
# =====================
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

print("Libraries successfully loaded. Ready to go!")

# %%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

feature_cols = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
target_col = 'BeatsPerMinute'

# %%
print("\n" + "="*50)
print("🔗 CORRELATION ANALYSIS")
print("="*50)

# Calculate correlation matrix
correlation_matrix = train[feature_cols + [target_col]].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
            center=0, square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('🔗 Feature Correlation Matrix', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# Feature importance based on correlation with target
target_correlations = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
print("\n🎯 Features ranked by correlation with BPM:")
for i, (feature, corr) in enumerate(target_correlations.items(), 1):
    print(f"{i:2d}. {feature:<25} | Correlation: {corr:.4f}")

# %%
print("\n" + "="*50)
print("🧮 FEATURE ENGINEERING")
print("="*50)

def create_features(df):
    """Create additional features that might help predict BPM"""
    df = df.copy()
    
    # 1. Rhythm and Energy interactions
    df['RhythmEnergyProduct'] = df['RhythmScore'] * df['Energy']
    df['RhythmEnergyRatio'] = df['RhythmScore'] / (df['Energy'] + 1e-8)
    
    # 2. Audio characteristics
    df['LoudnessEnergyProduct'] = df['AudioLoudness'] * df['Energy']
    df['VocalInstrumentalRatio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-8)
    
    # 3. Track duration features
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000  # Convert to minutes
    df['DurationMoodProduct'] = df['TrackDurationMin'] * df['MoodScore']
    
    # 4. Performance and quality features
    df['QualityPerformanceProduct'] = df['AcousticQuality'] * df['LivePerformanceLikelihood']
    
    # 5. Polynomial features for top correlated features
    top_3_features = target_correlations.head(3).index.tolist()
    for feature in top_3_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
    
    # 6. Binned features
    df['EnergyBin'] = pd.cut(df['Energy'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    df['RhythmBin'] = pd.cut(df['RhythmScore'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    
    # 7. Interaction between rhythm and tempo-related features
    df['RhythmDurationInteraction'] = df['RhythmScore'] * df['TrackDurationMin']

    # 8. Log transformations
    df['LogTrackDuration'] = np.log1p(df['TrackDurationMs'])
    df['LogLoudness'] = np.log1p(np.abs(df['AudioLoudness']) + 1e-8)  # Avoid log(0)

    # Statistical aggregations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df['FeatureMean'] = df[numeric_cols].mean(axis=1)
    df['FeatureStd'] = df[numeric_cols].std(axis=1)
    df['FeatureSkew'] = df[numeric_cols].skew(axis=1)
    
    # Ratios and interactions
    df['Energy_Loudness_Ratio'] = df['Energy'] / (np.abs(df['AudioLoudness']) + 1e-8)
    df['Rhythm_Duration_Normalized'] = df['RhythmScore'] / (df['TrackDurationMin'] + 1e-8)
    
    # Clustering features (add musical genre-like groupings)
    from sklearn.cluster import KMeans
    cluster_features = ['Energy', 'RhythmScore', 'AudioLoudness', 'VocalContent']
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['MusicCluster'] = kmeans.fit_predict(df[cluster_features])
    
    return df
    
# Apply feature engineering
train_engineered = create_features(train)
test_engineered = create_features(test)

# Get new feature columns
new_features = [col for col in train_engineered.columns if col not in train.columns]
print(f"✨ Created {len(new_features)} new features:")
for feature in new_features:
    print(f"   • {feature}")

# %%
numerical_features = train_engineered.select_dtypes(include=[np.number]).columns.tolist()
feature_columns = [col for col in numerical_features if col not in ['id', 'BeatsPerMinute']]

# %%
# Prepare features for modeling
feature_columns = [col for col in numerical_features if col != target_col]
X = train_engineered[feature_columns]
y = train_engineered[target_col]

print(f"📊 Training with {len(feature_columns)} features")
print(f"🎯 Target variable: {target_col}")

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# %%
def objective_xgboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'random_state': 42,
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100,
        'verbosity': 0,

        # GPU acceleration parameters
        'tree_method': 'gpu_hist',  # Use GPU histogram method
        'device': 'cuda',           # Specify CUDA device
    }

    # 5-fold CV for faster tuning
    cv_scores = []
    kf_tune = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf_tune.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model = xg.XGBRegressor(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)

        pred = model.predict(X_val_fold)
        score = root_mean_squared_error(y_val_fold, pred)
        cv_scores.append(score)

    return np.mean(cv_scores)

print("➗ Tuning XGBoost parameters...")
study_xgboost = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))       
study_xgboost.optimize(objective_xgboost, n_trials=50)

best_xgboost_params = study_xgboost.best_params
print(f"Best XGBoost CV RMSE: {study_xgboost.best_value:.4f}")
print(f"Best XGBoost params: {best_xgboost_params}")

# %%
def objective_lightgbm(trial):
    # LightGBM-specific parameter space for regression
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),       
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),   
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'random_state': 42,
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'early_stopping_rounds': 100,

        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    }

    # 5-fold CV for faster tuning
    cv_scores = []
    kf_tune = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf_tune.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])

        pred = model.predict(X_val_fold)
        score = root_mean_squared_error(y_val_fold, pred)
        cv_scores.append(score)

    return np.mean(cv_scores)

print("Tuning LightGBM parameters...")
study_lightgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_lightgbm.optimize(objective_lightgbm, n_trials=50)

best_lightgbm_params = study_lightgbm.best_params
print(f"Best LightGBM CV RMSE: {study_lightgbm.best_value:.4f}")
print(f"Best LightGBM params: {best_lightgbm_params}")

# %%
def objective_catboost(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 100,
        'verbose': False,
        
        'task_type': 'GPU',  # Use GPU for training
        'devices': '0:1'     # Specify GPU devices
    }

    # 5-fold CV for faster tuning
    cv_scores = []
    kf_tune = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf_tune.split(X, y):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold))

        pred = model.predict(X_val_fold)
        score = root_mean_squared_error(y_val_fold, pred)
        cv_scores.append(score)

    return np.mean(cv_scores)

print("➗ Tuning CatBoost parameters...")
study_catboost = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_catboost.optimize(objective_catboost, n_trials=50)

best_catboost_params = study_catboost.best_params
print(f"Best CatBoost params: {best_catboost_params}")
print(f"Best CatBoost CV RMSE: {study_catboost.best_value:.4f}")

# %%
# Apply same feature engineering
test_engineered = create_features(test)

# Prepare test features
X_test = test_engineered[feature_columns]

# Define the ensemble models with Optuna-optimized parameters
# Replace these with your actual best parameters from Optuna studies
best_models = {
    'XGBoost': xg.XGBRegressor(**best_xgboost_params, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(**best_lightgbm_params, random_state=42, n_jobs=-1), 
    'CatBoost': cb.CatBoostRegressor(**best_catboost_params, random_state=42),
    'Ridge': Ridge(alpha=1.0),  
}

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# =========================
# 1. Train base models
# =========================
print("Training ensemble base models...")

# Use the already fitted scaler from earlier (lines 210-212)
X_train_scaled = scaler.transform(X_train)

# Train XGBoost (doesn't need scaling, use original)
print("Training XGBoost...")
best_models['XGBoost'].fit(X_train, y_train)

# Train Ridge (needs scaling)
print("Training Ridge...")
best_models['Ridge'].fit(X_train_scaled, y_train)

# Train LightGBM (doesn't need scaling, use original)
print("Training LightGBM...")
best_models['LightGBM'].fit(X_train, y_train)

# Train CatBoost (doesn't need scaling, use original)
print("Training CatBoost...")
best_models['CatBoost'].fit(X_train, y_train)

# =========================
# 2. Generate meta-features (stacking inputs)
# =========================
print("Generating stacking features...")

# Validation predictions (meta-train)
val_meta = []
for model_name, model in best_models.items():
    if model_name == 'Ridge':
        pred = model.predict(X_val_scaled)
    else:
        pred = model.predict(X_val)
    val_meta.append(pred)

val_meta = np.array(val_meta).T   # shape (n_samples, n_models)

# Test predictions (meta-test)
test_meta = []
for model_name, model in best_models.items():
    if model_name == 'Ridge':
        X_test_scaled = scaler.transform(X_test)
        pred = model.predict(X_test_scaled)
    else:
        pred = model.predict(X_test)
    test_meta.append(pred)

test_meta = np.array(test_meta).T  # shape (n_samples, n_models)

# =========================
# 3. Train meta-model (stacker)
# =========================
print("Training meta-model...")

# RidgeCV automatically picks best regularization
stacker = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
stacker.fit(val_meta, y_val)

# Evaluate stacking performance
val_pred = stacker.predict(val_meta)
stack_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Stacking Validation RMSE: {stack_rmse:.4f}")
print("Meta-model coefficients (weights):", stacker.coef_)


# =========================
# 4. Final test predictions
# =========================
print("Making final stacked predictions...")
test_predictions = stacker.predict(test_meta)

# Submission
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': test_predictions
})

print(f"\n✅ Submission created with {len(submission)} predictions")
print(f"Prediction range: {test_predictions.min():.2f} - {test_predictions.max():.2f}")

# %%
submission.head()


