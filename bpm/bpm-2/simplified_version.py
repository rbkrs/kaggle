#!/usr/bin/env python
# coding: utf-8

# <a id="2"></a>
# <h1 style="background-color:#cee2f5; font-family: 'New Timeroman', cursive; color:#263842; font-size:200%; text-align:center; border-radius:30px; padding:10px;">
#   Predicting Beats-per-Minute of Songs: A Complete Solution
# </h1>
# ## Introduction
# 
# This notebook presents a comprehensive solution for the "Predicting the Beats-per-Minute of Songs" Kaggle competition. We'll walk through every step of the machine learning pipeline, from data exploration to submission, with detailed explanations designed for beginners.
# 
# ### What is BPM?
# 
# Beats Per Minute (BPM) is a measure of tempo in music that indicates the number of beats in one minute. For example:
# - Slow ballads might have 60-80 BPM
# - Pop songs often range from 100-130 BPM
# - Dance and electronic music typically has 120-140 BPM
# - Fast-paced genres like drum and bass can exceed 160 BPM
# 
# ### Competition Objective
# 
# Our goal is to predict the BPM of songs based on various audio features. Success is measured by the Root Mean Square Error (RMSE) between our predictions and the actual BPM values.
# 
# ### Important Note on This Dataset
# 
# Recent analysis has shown that this competition's original dataset contained randomly assigned BPM values, but the synthetic data generation process likely added some signal. This means our goal is to create a robust model that can find whatever patterns exist in the competition data.
# 
# Let's begin!

# ## 1. Setting Up Our Environment
# 
# First, we'll import the necessary libraries for our analysis and modeling.

# In[ ]:


# SIMPLIFIED VERSION - No Feature Engineering, Basic Parameters

# Import standard data manipulation and visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler  # Changed from RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor, RandomForestRegressor

# Import gradient boosting libraries
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Settings for better visuals and to suppress warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# For reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Helper function for measuring model performance
def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ## 2. Loading and Exploring the Data
# 
# Now we'll load the dataset and perform some initial exploration to understand what we're working with.

# In[2]:


# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Display information about the datasets
print(f"Train set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# IMPORTANT: Check the sample submission format
print("\nSample submission columns:")
print(sample_submission.columns.tolist())
print("\nSample submission preview:")
print(sample_submission.head())

# Check for missing values
print("\nMissing values in train set:")
print(train.isnull().sum())

print("\nMissing values in test set:")
print(test.isnull().sum())

# Save test IDs separately (using the correct case from sample submission)
if 'ID' in sample_submission.columns:
    id_column_name = 'ID'  # Use uppercase ID
else:
    id_column_name = 'id'  # Fallback to lowercase id

test_ids = test['id'].copy()  # Store for later use in submission

# Display the first few rows of the training data
print("\nTraining data preview:")
train.head()


# ### 2.1 Understanding the Features
# 
# Let's take a moment to understand what each feature represents in the context of music:
# 
# - **RhythmScore**: Represents the rhythmic intensity and pattern clarity of the track
# - **AudioLoudness**: The overall volume level of the track (often in negative dB values)
# - **VocalContent**: Measures the presence and prominence of vocals
# - **AcousticQuality**: How much acoustic (non-electronic) instrumentation is present
# - **InstrumentalScore**: Measures how instrumental (vs. vocal) the track is
# - **LivePerformanceLikelihood**: How likely the track sounds like it was recorded live
# - **MoodScore**: Represents the emotional character of the music
# - **TrackDurationMs**: Length of the track in milliseconds
# - **Energy**: The perceived energy level of the track
# - **BeatsPerMinute**: Our target variable - the tempo of the track
# 
# Now, let's explore the distributions of our features and target.

# In[3]:


# Examine the distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(train['BeatsPerMinute'], kde=True, bins=50)
plt.title('Distribution of Beats Per Minute', fontsize=16)
plt.xlabel('BPM', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(train['BeatsPerMinute'].mean(), color='red', linestyle='--', 
            label=f'Mean: {train["BeatsPerMinute"].mean():.2f}')
plt.axvline(train['BeatsPerMinute'].median(), color='green', linestyle='--', 
            label=f'Median: {train["BeatsPerMinute"].median():.2f}')
plt.legend()
plt.show()

# Calculate and print descriptive statistics for the target
target_stats = train['BeatsPerMinute'].describe()
print("Target Statistics:")
print(target_stats)


# In[4]:


# Feature distributions
plt.figure(figsize=(20, 16))

features = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}', fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel('Frequency', fontsize=10)

plt.tight_layout()
plt.show()


# ### 2.2 Correlation Analysis
# 
# Let's check how the features correlate with each other and with our target variable.

# In[5]:


# Create a correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = train.drop('id', axis=1, errors='ignore').corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            mask=mask, vmin=-1, vmax=1, annot_kws={"size": 8})
plt.title('Correlation Matrix of Features', fontsize=16)
plt.show()

# Sort features by correlation with target
target_correlation = correlation_matrix['BeatsPerMinute'].sort_values(ascending=False)
print("Feature Correlation with BPM:")
print(target_correlation)


# ### 2.3 Exploring Relationships with Scatter Plots
# 
# Let's visualize the relationships between our top features and the target.

# In[6]:


# Get top 4 features correlated with target
top_features = target_correlation.index[1:5]  # Excluding BeatsPerMinute itself

# Create scatter plots for each top feature vs target
plt.figure(figsize=(16, 12))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 2, i)
    plt.scatter(train[feature], train['BeatsPerMinute'], alpha=0.3, s=10)
    plt.title(f'{feature} vs BeatsPerMinute', fontsize=14)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('BeatsPerMinute', fontsize=12)
    
    # Add trend line
    z = np.polyfit(train[feature], train['BeatsPerMinute'], 1)
    p = np.poly1d(z)
    plt.plot(train[feature], p(train[feature]), "r--", alpha=0.8)
    
    # Add correlation coefficient to plot
    corr = train[feature].corr(train['BeatsPerMinute'])
    plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=12, ha='left', va='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

plt.tight_layout()
plt.show()


# ## 3. Feature Engineering
# 
# Now that we understand our data better, let's create new features that might help our model make better predictions. Feature engineering is a critical step in machine learning that can significantly improve performance.

# In[ ]:


# SIMPLIFIED VERSION - Skip feature engineering, use original features only

# Use only original features - no feature engineering
print("Using original features only (no feature engineering)")

# Use the original dataframes directly
train_simple = train.copy()
test_simple = test.copy()

print(f"Original features: {[col for col in train_simple.columns if col not in ['id', 'BeatsPerMinute']]}")
print(f"Number of features: {len([col for col in train_simple.columns if col not in ['id', 'BeatsPerMinute']])}")


# ### 3.1 Feature Importance Analysis
# 
# Let's do a preliminary check of which features might be most important for our model using a simple LightGBM model.

# In[8]:


def get_feature_importance(X, y):
    """Get feature importance from a LightGBM model"""
    model = lgb.LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)
    model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# Get feature importance
features_to_use = [col for col in train_engineered.columns if col not in ['id', 'BeatsPerMinute']]
X = train_engineered[features_to_use]
y = train_engineered['BeatsPerMinute']

importance_df = get_feature_importance(X, y)

# Visualize feature importance
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
plt.title('Top 20 Feature Importance from LightGBM', fontsize=16)
plt.tight_layout()
plt.show()

print("Top 10 most important features:")
print(importance_df.head(10))


# ## 4. Data Preprocessing
# 
# Now let's prepare our data for modeling. This includes scaling numerical features and setting up cross-validation.

# In[ ]:


# SIMPLIFIED VERSION - Basic preprocessing with StandardScaler

def preprocess_data_simple(train_df, test_df, target_col='BeatsPerMinute'):
    """Simplified preprocessing - no feature engineering, StandardScaler"""
    # Drop ID column for modeling
    if 'id' in train_df.columns:
        train_df = train_df.drop('id', axis=1)
    if 'id' in test_df.columns:
        test_df = test_df.drop('id', axis=1)
    
    # Split features and target
    y_train = train_df[target_col]
    X_train = train_df.drop(target_col, axis=1)
    X_test = test_df.copy()
    
    # Ensure X_train and X_test have the same columns
    feature_names = X_train.columns.tolist()
    
    # Use StandardScaler instead of RobustScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=feature_names
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=feature_names
    )
    
    return X_train_scaled, y_train, X_test_scaled, feature_names

# Apply simplified preprocessing
X_train_simple, y_train_simple, X_test_simple, feature_names_simple = preprocess_data_simple(train_simple, test_simple)

# Set up cross-validation strategy (same as before)
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
folds = list(kf.split(X_train_simple))

# Check data shapes after preprocessing
print(f"X_train shape: {X_train_simple.shape}")
print(f"y_train shape: {y_train_simple.shape}")
print(f"X_test shape: {X_test_simple.shape}")
print(f"Features: {feature_names_simple}")


# ## 5. Model Training and Evaluation
# 
# Now we'll train multiple models using cross-validation and combine them into an ensemble for better performance.

# In[ ]:


# SIMPLIFIED VERSION - Basic model parameters (more common defaults)

def train_lightgbm_simple(X_train, y_train, X_val, y_val):
    """Train a LightGBM model with simpler parameters"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,  # Higher learning rate
        'num_leaves': 31,
        'max_depth': -1,  # No limit
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': RANDOM_SEED,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]  # Earlier stopping
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,  # Fewer rounds
        callbacks=callbacks
    )
    
    return model

def train_xgboost_simple(X_train, y_train, X_val, y_val):
    """Train an XGBoost model with simpler parameters"""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.1,  # Higher learning rate
        'max_depth': 6,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'n_estimators': 500,  # Fewer estimators
        'random_state': RANDOM_SEED,
        'verbosity': 0
    }
    
    model = xgb.XGBRegressor(**params, early_stopping_rounds=50)  # Earlier stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def train_catboost_simple(X_train, y_train, X_val, y_val):
    """Train a CatBoost model with simpler parameters"""
    params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.1,  # Higher learning rate
        'depth': 6,
        'iterations': 500,  # Fewer iterations
        'random_seed': RANDOM_SEED,
        'verbose': False
    }
    
    model = cb.CatBoost(params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,  # Earlier stopping
        verbose=False
    )
    
    return model


# ### 5.1 Training with Cross-Validation
# 
# We'll train our models using 5-fold cross-validation to ensure they generalize well.

# In[ ]:


# SIMPLIFIED VERSION - Train models with equal weights

print("Starting simplified model training with cross-validation...")

# Initialize lists to store models and predictions
models_simple = []
oof_predictions_simple = np.zeros(len(X_train_simple))
test_predictions_simple = np.zeros(len(X_test_simple))

# Train and evaluate models across folds
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"\nTraining fold {fold_idx + 1}/{n_folds}")
    
    # Split data for this fold
    X_fold_train, X_fold_val = X_train_simple.iloc[train_idx], X_train_simple.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_simple.iloc[train_idx], y_train_simple.iloc[val_idx]
    
    # Train models with simplified parameters
    print("Training LightGBM...")
    lgb_model = train_lightgbm_simple(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    print("Training XGBoost...")
    xgb_model = train_xgboost_simple(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    print("Training CatBoost...")
    cb_model = train_catboost_simple(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
    
    # Simpler Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model.fit(X_fold_train, y_fold_train)
    
    # Make predictions on validation fold
    lgb_preds = lgb_model.predict(X_fold_val)
    xgb_preds = xgb_model.predict(X_fold_val)
    cb_preds = cb_model.predict(X_fold_val)
    rf_preds = rf_model.predict(X_fold_val)
    
    # EQUAL WEIGHTS - Simple average instead of weighted
    blend_preds = (lgb_preds + xgb_preds + cb_preds + rf_preds) / 4.0
    
    # Store out-of-fold predictions
    oof_predictions_simple[val_idx] = blend_preds
    
    # Make predictions on test set
    lgb_test_preds = lgb_model.predict(X_test_simple)
    xgb_test_preds = xgb_model.predict(X_test_simple)
    cb_test_preds = cb_model.predict(X_test_simple)
    rf_test_preds = rf_model.predict(X_test_simple)
    
    # Equal weights for test predictions
    fold_test_preds = (lgb_test_preds + xgb_test_preds + cb_test_preds + rf_test_preds) / 4.0
    test_predictions_simple += fold_test_preds / n_folds
    
    # Calculate and display fold metrics
    lgb_rmse = rmse(y_fold_val, lgb_preds)
    xgb_rmse = rmse(y_fold_val, xgb_preds)
    cb_rmse = rmse(y_fold_val, cb_preds)
    rf_rmse = rmse(y_fold_val, rf_preds)
    blend_rmse = rmse(y_fold_val, blend_preds)
    
    print(f"Fold {fold_idx + 1} Results:")
    print(f"LightGBM RMSE: {lgb_rmse:.5f}")
    print(f"XGBoost RMSE: {xgb_rmse:.5f}")
    print(f"CatBoost RMSE: {cb_rmse:.5f}")
    print(f"Random Forest RMSE: {rf_rmse:.5f}")
    print(f"Equal Weight Blend RMSE: {blend_rmse:.5f}")
    
    # Store models for this fold
    models_simple.append({
        'fold': fold_idx,
        'lgb_model': lgb_model,
        'xgb_model': xgb_model,
        'cb_model': cb_model,
        'rf_model': rf_model
    })

# Calculate overall cross-validation score
cv_score_simple = rmse(y_train_simple, oof_predictions_simple)
print(f"\nSimplified Approach CV RMSE: {cv_score_simple:.5f}")


# ### 5.2 Visualizing Model Performance
# 
# Let's visualize how well our model's predictions match the actual values.

# In[12]:


# Visualize the distribution of predictions vs actual
plt.figure(figsize=(12, 6))
plt.hist(y_train, alpha=0.5, label='Actual BPM', bins=50)
plt.hist(oof_predictions, alpha=0.5, label='Predicted BPM', bins=50)
plt.title('Distribution of Actual vs Predicted BPM', fontsize=16)
plt.legend()
plt.show()

# Scatter plot of predictions vs actual
plt.figure(figsize=(12, 8))
plt.scatter(y_train, oof_predictions, alpha=0.3, s=10)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual BPM', fontsize=14)
plt.ylabel('Predicted BPM', fontsize=14)
plt.title('Actual vs Predicted BPM', fontsize=16)

# Add correlation annotation
corr = np.corrcoef(y_train, oof_predictions)[0, 1]
plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=14, ha='left', va='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
plt.show()

# Residual plot
residuals = y_train - oof_predictions
plt.figure(figsize=(12, 6))
plt.scatter(oof_predictions, residuals, alpha=0.3, s=10)
plt.hlines(y=0, xmin=oof_predictions.min(), xmax=oof_predictions.max(), colors='r', linestyles='--')
plt.xlabel('Predicted BPM', fontsize=14)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
plt.title('Residual Plot', fontsize=16)
plt.show()


# ## 6. Advanced Stacking Ensemble
# 
# To potentially improve our predictions further, we'll implement a stacking ensemble that trains a meta-model on the predictions of our base models.

# In[13]:


# Build an advanced stacking ensemble
print("\nBuilding advanced stacking ensemble...")

# Filter models if any failed to train
valid_models = []
for model in models:
    if all(m is not None for m in [model['lgb_model'], model['xgb_model'], model['cb_model'], model['rf_model']]):
        valid_models.append(model)

if len(valid_models) > 0:
    # Create meta-features for stacking
    X_meta_train = np.column_stack([
        np.array([model['lgb_model'].predict(X_train) for model in valid_models]).mean(axis=0),
        np.array([model['xgb_model'].predict(X_train) for model in valid_models]).mean(axis=0),
        np.array([model['cb_model'].predict(X_train) for model in valid_models]).mean(axis=0),
        np.array([model['rf_model'].predict(X_train) for model in valid_models]).mean(axis=0)
    ])

    # Create meta-features for test set
    X_meta_test = np.column_stack([
        np.array([model['lgb_model'].predict(X_test) for model in valid_models]).mean(axis=0),
        np.array([model['xgb_model'].predict(X_test) for model in valid_models]).mean(axis=0),
        np.array([model['cb_model'].predict(X_test) for model in valid_models]).mean(axis=0),
        np.array([model['rf_model'].predict(X_test) for model in valid_models]).mean(axis=0)
    ])

    # Train a Ridge meta-model
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_meta_train, y_train)

    # Make final predictions
    stacking_predictions = meta_model.predict(X_meta_test)

    # Analyze the performance of the stacking model
    stacking_oof_preds = meta_model.predict(X_meta_train)
    stacking_cv_score = rmse(y_train, stacking_oof_preds)
    print(f"Stacking Ensemble CV RMSE: {stacking_cv_score:.5f}")

    # Compare with the simple average approach
    print(f"Simple Average Ensemble CV RMSE: {cv_score:.5f}")

    # Select the better performing approach for final predictions
    if stacking_cv_score < cv_score:
        print("Using stacking ensemble for final predictions")
        final_predictions = stacking_predictions
    else:
        print("Using simple average ensemble for final predictions")
        final_predictions = test_predictions
else:
    print("Not enough valid models for stacking. Using simple average ensemble.")
    final_predictions = test_predictions


# ## 7. Preparing the Submission
# 
# Finally, we'll prepare our submission file with the correct format.

# In[14]:


# Check sample submission again
print("Sample submission columns:", sample_submission.columns.tolist())
print("Sample submission format:")
print(sample_submission.head())


# In[ ]:


# SIMPLIFIED VERSION - Create submission with simpler approach

# Apply reasonable bounds to predictions (same as before)
min_bpm = max(60, y_train_simple.min())
max_bpm = min(200, y_train_simple.max())
bounded_predictions_simple = np.clip(test_predictions_simple, min_bpm, max_bpm)

# Create submission file
submission_simple = pd.DataFrame()
submission_simple['id'] = test_ids
submission_simple['BeatsPerMinute'] = bounded_predictions_simple

# Show comparison
print("COMPARISON OF APPROACHES:")
print(f"Original Complex Approach CV RMSE: {cv_score:.5f}")
print(f"Simplified Approach CV RMSE: {cv_score_simple:.5f}")
print(f"Difference: {cv_score_simple - cv_score:.5f}")

print("\nSimplified submission preview:")
print(submission_simple.head())

# Save simplified submission
submission_simple.to_csv('submission_simplified.csv', index=False)
print(f"\nSimplified submission saved to submission_simplified.csv")

# Compare prediction distributions
print(f"\nPrediction Statistics:")
print(f"Original approach - Mean: {final_predictions.mean():.3f}, Std: {final_predictions.std():.3f}")
print(f"Simplified approach - Mean: {test_predictions_simple.mean():.3f}, Std: {test_predictions_simple.std():.3f}")

# Show the differences made
print("\nSIMPLIFIED CHANGES MADE:")
print("1. ❌ Removed all 15 engineered features")
print("2. ❌ Changed from RobustScaler to StandardScaler") 
print("3. ❌ Simplified hyperparameters (higher learning rates, fewer iterations)")
print("4. ❌ Changed from weighted ensemble (35%/35%/20%/10%) to equal weights (25% each)")
print("5. ❌ Reduced model complexity (fewer estimators, earlier stopping)")


# ## 8. Final Verification and Submission
# 
# Let's do a final check to make absolutely sure our submission format is correct.

# In[16]:


# Final verification and format check
def verify_submission(submission_file, sample_submission):
    """Perform comprehensive verification of submission format"""
    verification = pd.read_csv(submission_file)
    
    print("SUBMISSION VERIFICATION:")
    print(f"1. Submission shape: {verification.shape}")
    print(f"   Expected shape: {sample_submission.shape}")
    
    print(f"\n2. Submission columns: {verification.columns.tolist()}")
    print(f"   Expected columns: {sample_submission.columns.tolist()}")
    
    # Check if columns match exactly (including case)
    columns_match = verification.columns.tolist() == sample_submission.columns.tolist()
    print(f"\n3. Columns match exactly: {'✅ YES' if columns_match else '❌ NO'}")
    
    # Check ID column values
    id_col = sample_submission.columns[0]
    id_match = set(verification[id_col]) == set(sample_submission[id_col])
    print(f"\n4. ID values match sample: {'✅ YES' if id_match else '❌ NO'}")
    
    # Check target column statistics
    target_col = sample_submission.columns[1]
    print(f"\n5. Target column statistics:")
    print(f"   Min: {verification[target_col].min():.2f}")
    print(f"   Max: {verification[target_col].max():.2f}")
    print(f"   Mean: {verification[target_col].mean():.2f}")
    print(f"   Std: {verification[target_col].std():.2f}")
    
    # Final verdict
    if columns_match and id_match:
        print("\n✅ SUBMISSION FORMAT LOOKS CORRECT! Ready to upload.")
    else:
        print("\n❌ SUBMISSION FORMAT HAS ISSUES! Please fix before uploading.")
    
    return verification

# Run the verification
final_verification = verify_submission(submission_file, sample_submission)

# If there's a problem with column names, fix it one last time
if final_verification.columns.tolist() != sample_submission.columns.tolist():
    print("\nAttempting to fix column names one last time...")
    final_verification.columns = sample_submission.columns
    final_verification.to_csv(submission_file, index=False)
    print(f"Fixed submission saved to {submission_file}")
    
    # Verify again
    verify_submission(submission_file, sample_submission)


# ## 9. Conclusion and Summary
# 
# In this notebook, we've created a comprehensive solution for predicting the Beats-per-Minute (BPM) of songs. Here's a summary of what we did:
# 
# 1. **Data Exploration**: We examined the features and their relationships with BPM.
# 2. **Feature Engineering**: We created additional features to help our models better understand the data.
# 3. **Model Training**: We trained multiple models using 5-fold cross-validation.
# 4. **Ensemble Learning**: We combined our models to improve prediction accuracy.
# 5. **Submission Preparation**: We carefully ensured our submission format was correct.
# 
# ### Key Insights:
# 
# - The competition dataset appears to be based on randomly assigned BPM values in the original dataset
# - Despite this, our ensemble approach helps find whatever patterns exist in the synthetic data
# - The most important feature, according to our models, was RhythmScore, which makes musical sense
# - Proper submission formatting with the uppercase 'ID' column was critical
# 
# ### Potential Improvements:
# 
# - Experiment with more sophisticated feature engineering
# - Try neural network approaches for potentially capturing more complex patterns
# - Further tune the hyperparameters of our models
# - Explore additional ensemble techniques
