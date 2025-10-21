from utils import *

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import shap
from typing import List, Dict, Any, Tuple, Optional, Callable

# Import all the original functions (assumed to be in a module called ml_functions)
# from ml_functions import *

class EDAIPipeline:
    """Pipeline for Exploratory Data Analysis"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize EDA Pipeline
        
        Parameters:
        - dataframe: pandas DataFrame to analyze
        """
        self.df = dataframe
        self.numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns
        
    def run_basic_info(self):
        """Display basic information about the dataset"""
        print("="*50)
        print("DATASET OVERVIEW")
        print("="*50)
        print(f"\nShape: {self.df.shape}")
        print(f"Numerical columns: {len(self.numerical_cols)}")
        print(f"Categorical columns: {len(self.categorical_cols)}")
        
        print("\n" + "="*50)
        print("DATA TYPES")
        print("="*50)
        print(self.df.dtypes)
        
        print("\n" + "="*50)
        print("MISSING VALUES")
        print("="*50)
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
            
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        print(self.df.describe())
        
    def run_visual_analysis(self):
        """Run all visual analysis functions"""
        print("\n" + "="*50)
        print("VISUAL ANALYSIS")
        print("="*50)
        
        if len(self.numerical_cols) > 0:
            print("\nNumerical Features Distribution:")
            plot_nums(self.df)
            
            print("\nCorrelation Analysis:")
            heatmap_nums(self.df)
        
        if len(self.categorical_cols) > 0:
            print("\nCategorical Features Distribution:")
            plot_cats(self.df)
    
    def run_full_eda(self):
        """Execute complete EDA pipeline"""
        self.run_basic_info()
        self.run_visual_analysis()
        
        return {
            'numerical_cols': list(self.numerical_cols),
            'categorical_cols': list(self.categorical_cols),
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict()
        }


class FeatureEngineeringPipeline:
    """Pipeline for Feature Engineering"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize Feature Engineering Pipeline
        
        Parameters:
        - dataframe: pandas DataFrame
        """
        self.df = dataframe.copy()
        self.new_features = []
        
    def create_combinations(self, features: List[str] = None):
        """
        Create combination features
        
        Parameters:
        - features: list of features to combine (default: all numerical)
        """
        if features is None:
            features = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        print(f"Creating combinations for {len(features)} features...")
        initial_cols = len(self.df.columns)
        
        self.df = create_combination_features(self.df, features)
        
        new_cols = len(self.df.columns) - initial_cols
        self.new_features.extend(self.df.columns[-new_cols:].tolist())
        print(f"Created {new_cols} new combination features")
        
        return self
    
    def remove_high_correlation(self, threshold: float = 0.95):
        """
        Remove highly correlated features
        
        Parameters:
        - threshold: correlation threshold for removal
        """
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df[numerical_cols].corr().abs()
        
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                   if any(upper_triangle[column] > threshold)]
        
        print(f"Removing {len(to_drop)} highly correlated features (threshold: {threshold})")
        self.df = self.df.drop(columns=to_drop)
        
        return self
    
    def get_engineered_data(self):
        """Return the engineered dataframe"""
        print(f"\nFinal dataset shape: {self.df.shape}")
        print(f"New features created: {len(self.new_features)}")
        return self.df


class ModelTrainingPipeline:
    """Pipeline for Model Training and Validation"""
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_test: pd.DataFrame = None):
        """
        Initialize Model Training Pipeline
        
        Parameters:
        - X_train: training features
        - y_train: training target
        - X_test: test features (optional)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.models = {}
        self.predictions = {}
        self.scores = {}
        
    def train_single_model(self, model_class, model_name: str, 
                          model_params: Dict = None,
                          folds: int = 5, 
                          eval_metric: Callable = root_mean_squared_error):
        """
        Train a single model with cross-validation
        
        Parameters:
        - model_class: model class to instantiate
        - model_name: name for storing results
        - model_params: model hyperparameters
        - folds: number of CV folds
        - eval_metric: evaluation metric function
        """
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print('='*50)
        
        if self.X_test is not None:
            score, test_pred, oof_pred = oof_cross_val(
                model_class=model_class,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                folds=folds,
                model_params=model_params,
                eval_metric=eval_metric
            )
            
            self.predictions[model_name] = {
                'oof': oof_pred,
                'test': test_pred
            }
        else:
            # Simple CV without test predictions
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            oof_pred = np.zeros(len(self.X_train))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
                X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                model = model_class(**model_params) if model_params else model_class()
                model.fit(X_tr, y_tr)
                
                y_val_pred = model.predict(X_val)
                oof_pred[val_idx] = y_val_pred
                
                fold_score = eval_metric(y_val, y_val_pred)
                print(f"  Fold {fold+1} {eval_metric.__name__}: {fold_score:.4f}")
            
            score = eval_metric(self.y_train, oof_pred)
            print(f"\nFinal OOF {eval_metric.__name__}: {score:.4f}")
            
            self.predictions[model_name] = {'oof': oof_pred}
        
        self.scores[model_name] = score
        
        # Train final model on all data
        final_model = model_class(**model_params) if model_params else model_class()
        final_model.fit(self.X_train, self.y_train)
        self.models[model_name] = final_model
        
        return self
    
    def train_stacking_ensemble(self, base_models: List[Any], 
                               meta_model = None,
                               folds: int = 5,
                               eval_metric: Callable = root_mean_squared_error):
        """
        Train a stacking ensemble
        
        Parameters:
        - base_models: list of base models
        - meta_model: meta-learner model
        - folds: number of CV folds
        - eval_metric: evaluation metric
        """
        print(f"\n{'='*50}")
        print("Training Stacking Ensemble")
        print('='*50)
        
        if self.X_test is None:
            raise ValueError("X_test is required for stacking ensemble")
        
        meta_model, meta_train, meta_test, score = generate_stack(
            base_models=base_models,
            X=self.X_train,
            y=self.y_train,
            X_test=self.X_test,
            meta_model=meta_model,
            folds=folds,
            eval_metric=eval_metric
        )
        
        self.models['stacking_ensemble'] = {
            'meta_model': meta_model,
            'base_models': base_models
        }
        
        self.predictions['stacking_ensemble'] = {
            'oof': meta_model.predict(meta_train),
            'test': meta_model.predict(meta_test)
        }
        
        self.scores['stacking_ensemble'] = score
        
        return self
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        summary = pd.DataFrame({
            'Model': list(self.scores.keys()),
            'Score': list(self.scores.values())
        }).sort_values('Score')
        
        print(f"\n{'='*50}")
        print("MODEL PERFORMANCE SUMMARY")
        print('='*50)
        print(summary.to_string(index=False))
        
        return summary


class ModelEvaluationPipeline:
    """Pipeline for Model Evaluation and Interpretation"""
    
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                 y_pred: np.ndarray = None):
        """
        Initialize Model Evaluation Pipeline
        
        Parameters:
        - model: trained model
        - X_train: training features
        - y_train: true target values
        - y_pred: predictions (optional)
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.y_pred = y_pred if y_pred is not None else model.predict(X_train)
        
    def plot_performance(self, model_name: str = "Model"):
        """Plot model performance"""
        plot_model_performance(self.y_train, self.y_pred, model_name)
        return self
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        try:
            plot_feature_importance(self.model, self.X_train, self.y_train)
        except ValueError as e:
            print(f"Cannot plot feature importance: {e}")
        return self
    
    def plot_shap_analysis(self):
        """Plot SHAP values"""
        try:
            plot_shap_values(self.model, self.X_train, self.y_train)
        except Exception as e:
            print(f"Cannot perform SHAP analysis: {e}")
        return self
    
    def run_full_evaluation(self, model_name: str = "Model"):
        """Run complete evaluation pipeline"""
        print(f"\n{'='*50}")
        print(f"EVALUATING: {model_name}")
        print('='*50)
        
        self.plot_performance(model_name)
        self.plot_feature_importance()
        self.plot_shap_analysis()
        
        return self


class CompletMLPipeline:
    """Complete end-to-end ML Pipeline"""
    
    def __init__(self, train_data: pd.DataFrame, target_column: str,
                 test_data: pd.DataFrame = None):
        """
        Initialize Complete ML Pipeline
        
        Parameters:
        - train_data: training dataframe
        - target_column: name of target column
        - test_data: test dataframe (optional)
        """
        self.train_data = train_data
        self.target_column = target_column
        self.test_data = test_data
        
        # Separate features and target
        self.X_train = train_data.drop(columns=[target_column])
        self.y_train = train_data[target_column]
        self.X_test = test_data.drop(columns=[target_column]) if test_data is not None else None
        
        # Pipeline components
        self.eda_pipeline = None
        self.fe_pipeline = None
        self.training_pipeline = None
        self.evaluation_pipeline = None
        
    def run_eda(self):
        """Run EDA pipeline"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        self.eda_pipeline = EDAIPipeline(self.train_data)
        eda_results = self.eda_pipeline.run_full_eda()
        return eda_results
    
    def run_feature_engineering(self, create_combinations: bool = False,
                                remove_correlated: bool = True,
                                correlation_threshold: float = 0.95):
        """Run feature engineering pipeline"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        self.fe_pipeline = FeatureEngineeringPipeline(self.X_train)
        
        if create_combinations:
            self.fe_pipeline.create_combinations()
        
        if remove_correlated:
            self.fe_pipeline.remove_high_correlation(correlation_threshold)
        
        self.X_train = self.fe_pipeline.get_engineered_data()
        
        # Apply same transformations to test data if available
        if self.X_test is not None:
            # This is simplified - in practice you'd need to track and apply the same operations
            pass
        
        return self.X_train
    
    def train_models(self, models_config: Dict[str, Dict]):
        """
        Train multiple models
        
        Parameters:
        - models_config: dictionary with model configurations
          Example: {
              'model_name': {
                  'class': ModelClass,
                  'params': {...},
                  'folds': 5
              }
          }
        """
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        self.training_pipeline = ModelTrainingPipeline(
            self.X_train, self.y_train, self.X_test
        )
        
        for model_name, config in models_config.items():
            self.training_pipeline.train_single_model(
                model_class=config['class'],
                model_name=model_name,
                model_params=config.get('params'),
                folds=config.get('folds', 5)
            )
        
        return self.training_pipeline.get_model_summary()
    
    def evaluate_best_model(self):
        """Evaluate the best performing model"""
        print("\n" + "="*60)
        print("BEST MODEL EVALUATION")
        print("="*60)
        
        # Find best model
        best_model_name = min(self.training_pipeline.scores, 
                              key=self.training_pipeline.scores.get)
        best_model = self.training_pipeline.models[best_model_name]
        best_predictions = self.training_pipeline.predictions[best_model_name]['oof']
        
        print(f"Best model: {best_model_name}")
        print(f"Score: {self.training_pipeline.scores[best_model_name]:.4f}")
        
        self.evaluation_pipeline = ModelEvaluationPipeline(
            best_model, self.X_train, self.y_train, best_predictions
        )
        self.evaluation_pipeline.run_full_evaluation(best_model_name)
        
        return best_model
    
    def run_complete_pipeline(self, models_config: Dict[str, Dict],
                             run_eda: bool = True,
                             run_fe: bool = True,
                             evaluate_best: bool = True):
        """
        Run the complete ML pipeline
        
        Parameters:
        - models_config: model configurations
        - run_eda: whether to run EDA
        - run_fe: whether to run feature engineering
        - evaluate_best: whether to evaluate best model
        """
        results = {}
        
        if run_eda:
            results['eda'] = self.run_eda()
        
        if run_fe:
            results['engineered_features'] = self.run_feature_engineering()
        
        results['model_summary'] = self.train_models(models_config)
        
        if evaluate_best:
            results['best_model'] = self.evaluate_best_model()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return results


# Example usage
if __name__ == "__main__":
    # Example with dummy data
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    import xgboost as xgb
    
    # Generate dummy data
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    
    # Split into train/test
    train_df = df.iloc[:800]
    test_df = df.iloc[800:]
    
    # Define models to train
    models_config = {
        'ridge': {
            'class': Ridge,
            'params': {'alpha': 1.0},
            'folds': 5
        },
        'lasso': {
            'class': Lasso,
            'params': {'alpha': 0.1},
            'folds': 5
        },
        'random_forest': {
            'class': RandomForestRegressor,
            'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
            'folds': 5
        },
        'gradient_boosting': {
            'class': GradientBoostingRegressor,
            'params': {'n_estimators': 100, 'max_depth': 3, 'random_state': 42},
            'folds': 5
        },
        'xgboost': {
            'class': xgb.XGBRegressor,
            'params': {'n_estimators': 100, 'max_depth': 3, 'random_state': 42},
            'folds': 5
        }
    }
    
    # Run complete pipeline
    pipeline = CompletMLPipeline(train_df, 'target', test_df)
    results = pipeline.run_complete_pipeline(
        models_config=models_config,
        run_eda=True,
        run_fe=True,
        evaluate_best=True
    )
    
    print("\nPipeline execution completed successfully!")