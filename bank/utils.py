from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import shap
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    explained_variance_score,
    median_absolute_error,
    max_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
    mean_absolute_percentage_error,
    mean_squared_log_error,
    log_loss
)

def generate_stack(
    base_models,
    X, y,
    X_test,
    meta_model=None,
    folds=5,
    seed=42,
    eval_metric=root_mean_squared_error
):
    """
    Train a stacking ensemble.

    Parameters:
    - base_models: list of sklearn-like models (must implement fit & predict)
    - X, y: training data
    - X_test: test data (for generating meta features)
    - meta_model: sklearn-like model for meta learning (default: LinearRegression)
    - folds: number of KFold splits
    - seed: random seed for reproducibility
    - eval_metric: function to evaluate final stacked predictions (default: RMSE)

    Returns:
    - meta_model: trained meta-model
    - meta_features_train: level-one train predictions
    - meta_features_test: level-one test predictions
    - final_score: score computed by eval_metric
    """

    n_models = len(base_models)
    meta_features_train = np.zeros((len(X), n_models))
    meta_features_test = np.zeros((len(X_test), n_models))
    
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    print(f"Starting stacking with {n_models} base models and {folds}-fold CV...")
    
    for i, model in enumerate(base_models):
        print(f"\nTraining model {i+1}/{n_models}: {model.__class__.__name__}")
        
        oof = np.zeros(len(X))
        preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            oof[val_idx] = model.predict(X_val)
            preds += model.predict(X_test) / folds

            fold_score = eval_metric(y_val, oof[val_idx])
            print(f"  Fold {fold+1} {eval_metric.__name__}: {fold_score:.4f}")

        meta_features_train[:, i] = oof
        meta_features_test[:, i] = preds
    
    # Use default meta model if none provided
    if meta_model is None:
        meta_model = LinearRegression()

    print(f"\nFitting meta-model: {meta_model.__class__.__name__}")
    meta_model.fit(meta_features_train, y)
    
    stacked_preds = meta_model.predict(meta_features_train)
    final_score = eval_metric(y, stacked_preds)
    print(f"\nFinal cross-validation {eval_metric.__name__}: {final_score:.4f}")

    return meta_model, meta_features_train, meta_features_test, final_score


def plot_nums(dataframe):
    float_cols = [col for col in dataframe.columns if dataframe[col].dtype == "float64" or dataframe[col].dtype == "int64"]

    cols_per_row = 3
    num_plots = len(float_cols)
    rows = (num_plots // cols_per_row) + (num_plots % cols_per_row > 0) 

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows)) 
    axes = axes.flatten()  

    for idx, col in enumerate(float_cols):
        sns.histplot(dataframe[col], bins=50, kde=True, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {col}")

    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_cats(dataframe):
    categorical_features = dataframe.select_dtypes(include=['object']).columns

    num_features = len(categorical_features)
    cols = 3 
    rows = (num_features // cols) + (num_features % cols > 0) 

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5)) 
    axes = axes.flatten()  

    for i, feature in enumerate(categorical_features):
        dataframe[feature].value_counts().plot.pie(
            autopct='%1.1f%%', ax=axes[i], startangle=90, cmap="viridis"
        )
        axes[i].set_title(feature)
        axes[i].set_ylabel("") 

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def heatmap_nums(dataframe): 
    heatmap_train = dataframe.select_dtypes(include=["float64", "int64"])

    corr_matrix = heatmap_train.corr()

    threshold = 0.8

    high_corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
        .stack()  
        .reset_index()
    )

    high_corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    high_corr_pairs = high_corr_pairs[high_corr_pairs["Correlation"].abs() > threshold]  

    plt.figure(figsize=(30, 12))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.show()

    print("Highly correlated feature pairs (above threshold):")
    print(high_corr_pairs)


def create_combination_features(df, features):
    combinations = itertools.combinations(features, 2)

    for comb in combinations:
        feature_name = "_".join(comb)
        df[feature_name] = df[list(comb)].mean(axis=1)
    
    return df


def plot_feature_importance(model, X_train, y_train):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not have feature importances or coefficients.")

    feature_names = X_train.columns
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()


def plot_model_performance(y_true, y_pred, model_name):
    """
    Plot the performance of a regression model.

    Parameters:
    - y_true: true target values
    - y_pred: predicted target values
    - model_name: name of the model for labeling the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f"{model_name} Predictions vs True Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    plt.show()
    print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
    print(f"R^2: {model_name.score(y_true.reshape(-1, 1), y_pred.reshape(-1, 1)):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Explained Variance: {explained_variance_score(y_true, y_pred):.4f}")
    print(f"Median Absolute Error: {median_absolute_error(y_true, y_pred):.4f}")
    print(f"Max Error: {max_error(y_true, y_pred):.4f}")
    print(f"Mean Poisson Deviance: {mean_poisson_deviance(y_true, y_pred):.4f}")
    print(f"Mean Gamma Deviance: {mean_gamma_deviance(y_true, y_pred):.4f}")
    print(f"Mean Tweedie Deviance: {mean_tweedie_deviance(y_true, y_pred):.4f}")
    print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_true, y_pred):.4f}")
    print(f"Mean Squared Logarithmic Error: {mean_squared_log_error(y_true, y_pred):.4f}")
    print(f"Log Loss: {log_loss(y_true, y_pred):.4f}")


def oof_cross_val(
    model_class,
    X_train, y_train,
    X_test,
    folds=5,
    model_params=None,
    eval_metric=root_mean_squared_error,
    seed=42
):
    """
    Perform out-of-fold (OOF) cross-validation for a single model.

    Parameters:
    - model_class: class of the model (e.g., xgboost.XGBRegressor)
    - X_train, y_train: training data
    - X_test: test data to predict on each fold
    - folds: number of KFold splits
    - model_params: dict of hyperparameters for the model
    - eval_metric: function to evaluate fold performance (default: root_mean_squared_error)
    - seed: random seed for reproducibility

    Returns:
    - final_score: evaluation metric on full OOF predictions
    - test_pred_avg: averaged test set predictions across folds
    - oof_pred: full OOF predictions on training data
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))

    print(f"Starting OOF CV with model: {model_class.__name__} and {folds}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = model_class(**model_params) if model_params else model_class()
        
        model.fit(X_tr, y_tr)
        
        y_val_pred = model.predict(X_val)
        oof_pred[val_idx] = y_val_pred
        
        fold_score = eval_metric(y_val, y_val_pred)
        print(f"  Fold {fold+1} {eval_metric.__name__}: {fold_score:.4f}")
        
        test_pred += model.predict(X_test) / folds

    final_score = eval_metric(y_train, oof_pred)
    print(f"\nFinal OOF {eval_metric.__name__}: {final_score:.4f}")

    return final_score, test_pred, oof_pred


def plot_shap_values(model, X_train, y_train):
    """
    Plot SHAP values for a model.

    Parameters:
    - model: trained model (must support SHAP)
    - X_train: training data
    - y_train: target values
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.title("SHAP Feature Importance")
    plt.show()