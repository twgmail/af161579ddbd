"""
Enhanced Diabetes Prediction Pipeline

This script provides a robust machine learning pipeline for diabetes prediction
with comprehensive error handling, data validation, and model evaluation.

Author: OpenHands AI Assistant
Date: 2025-07-01
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve
)
import joblib

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('diabetes_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration with validation
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'max_iter': 1000,
    'cv_folds': 5,
    'target_column': 'Diabetes_binary',
    'dataset_file': 'c99d9bc33649_b.csv',
    'min_samples': 100,  # Minimum samples required
    'max_features': 1000,  # Maximum features allowed
}

def validate_config() -> None:
    """Validate configuration parameters."""
    if not 0 < CONFIG['test_size'] < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {CONFIG['test_size']}")
    
    if CONFIG['cv_folds'] < 2:
        raise ValueError(f"cv_folds must be >= 2, got {CONFIG['cv_folds']}")
    
    if CONFIG['max_iter'] < 1:
        raise ValueError(f"max_iter must be >= 1, got {CONFIG['max_iter']}")

def get_project_root() -> Path:
    """
    Get project root directory more reliably.
    
    Returns:
        Path: Project root directory
        
    Raises:
        FileNotFoundError: If project root cannot be determined
    """
    current_file = Path(__file__).resolve()
    
    # Go up directories until we find the project root (contains datasets folder)
    for parent in current_file.parents:
        if (parent / "datasets").exists():
            logger.info(f"Found project root: {parent}")
            return parent
    
    # Fallback to current working directory
    cwd = Path.cwd()
    if (cwd / "datasets").exists():
        logger.warning(f"Using current working directory as project root: {cwd}")
        return cwd
    
    raise FileNotFoundError(
        "Could not determine project root. Please ensure the script is run from "
        "within the project directory or that 'datasets' folder exists."
    )

def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """
    Load and validate the dataset.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded and validated dataset
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset is invalid
    """
    try:
        logger.info(f"Loading data from {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        # Check file size (basic validation)
        file_size = data_path.stat().st_size
        if file_size == 0:
            raise ValueError("Dataset file is empty")
        
        logger.info(f"Dataset file size: {file_size / (1024*1024):.2f} MB")
        
        data = pd.read_csv(data_path)
        
        # Basic validation
        if data.empty:
            raise ValueError("Dataset contains no data")
        
        if len(data) < CONFIG['min_samples']:
            raise ValueError(f"Dataset too small: {len(data)} < {CONFIG['min_samples']} samples")
        
        if len(data.columns) > CONFIG['max_features']:
            logger.warning(f"Large number of features: {len(data.columns)}")
        
        if CONFIG['target_column'] not in data.columns:
            available_cols = ', '.join(data.columns[:10])  # Show first 10 columns
            raise ValueError(
                f"Target column '{CONFIG['target_column']}' not found. "
                f"Available columns: {available_cols}..."
            )
        
        # Check for duplicate rows
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows ({duplicates/len(data)*100:.2f}%)")
        
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        logger.info(f"Missing values: {data.isnull().sum().sum()}")
        logger.info(f"Data types: {data.dtypes.value_counts().to_dict()}")
        
        return data
    
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset file is empty or corrupted")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        raise

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data for machine learning.
    
    Args:
        data: Raw dataset
        
    Returns:
        Tuple of (features, target)
        
    Raises:
        ValueError: If preprocessing fails
    """
    logger.info("Preprocessing data...")
    
    try:
        # Handle missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Missing values detected: {missing_count}")
            
            # For numeric columns, fill with median
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if data[col].isnull().sum() > 0:
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled {col} missing values with median: {median_val}")
            
            # For categorical columns, fill with mode
            categorical_columns = data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if data[col].isnull().sum() > 0:
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                    data[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled {col} missing values with mode: {mode_val}")
        
        # Separate features and target
        if CONFIG['target_column'] not in data.columns:
            raise ValueError(f"Target column '{CONFIG['target_column']}' not found after preprocessing")
        
        X = data.drop(CONFIG['target_column'], axis=1)
        y = data[CONFIG['target_column']]
        
        # Validate target variable
        unique_targets = y.nunique()
        if unique_targets < 2:
            raise ValueError(f"Target variable must have at least 2 classes, found {unique_targets}")
        
        if unique_targets > 10:
            logger.warning(f"High number of target classes: {unique_targets}")
        
        # Check for class imbalance
        class_distribution = y.value_counts(normalize=True).sort_index()
        logger.info(f"Class distribution: {class_distribution.to_dict()}")
        
        # Warn about severe class imbalance
        min_class_ratio = class_distribution.min()
        if min_class_ratio < 0.05:
            logger.warning(f"Severe class imbalance detected. Minority class: {min_class_ratio:.3f}")
        
        # Validate features
        if X.empty:
            raise ValueError("No features available after preprocessing")
        
        # Check for constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            logger.warning(f"Constant features detected (will be removed): {constant_features}")
            X = X.drop(columns=constant_features)
        
        # Ensure all features are numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning(f"Non-numeric features detected: {non_numeric}")
            # Simple encoding for categorical variables
            for col in non_numeric:
                X[col] = pd.Categorical(X[col]).codes
        
        logger.info(f"Preprocessing completed. Features: {X.shape[1]}, Samples: {len(X)}")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """
    Train and evaluate the logistic regression model.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of (model, scaler, results_dict)
    """
    logger.info("Splitting data and training model...")
    
    try:
        # Validate inputs
        if len(X) != len(y):
            raise ValueError(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
        
        if len(X) < CONFIG['min_samples']:
            raise ValueError(f"Insufficient samples for training: {len(X)}")
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state'],
            stratify=y  # Maintain class distribution
        )
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Scale features (important for logistic regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = LogisticRegression(
            max_iter=CONFIG['max_iter'], 
            random_state=CONFIG['random_state'],
            solver='lbfgs'  # Good default solver
        )
        
        logger.info("Training logistic regression model...")
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation with stratification
        cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_state'])
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"Cross-validation ROC-AUC scores: {cv_scores}")
        logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Comprehensive evaluation
        logger.info("Model Evaluation Results:")
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(y_test, y_pred))
        
        print("\n" + "="*50)
        print("CONFUSION MATRIX")
        print("="*50)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Calculate additional metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        # Feature importance (coefficients for logistic regression)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\n" + "="*50)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("="*50)
        print(feature_importance.head(10).to_string(index=False))
        
        # Prepare results dictionary
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return model, scaler, results
    
    except Exception as e:
        logger.error(f"Error during model training/evaluation: {str(e)}")
        raise

def save_model_and_artifacts(model: LogisticRegression, scaler: StandardScaler, 
                           results: Dict[str, Any], project_root: Path) -> None:
    """
    Save the trained model, scaler, and results.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        results: Evaluation results
        project_root: Project root directory
    """
    try:
        model_dir = project_root / "models"
        model_dir.mkdir(exist_ok=True)
        
        # Save model and scaler
        model_path = model_dir / "diabetes_model.joblib"
        scaler_path = model_dir / "scaler.joblib"
        results_path = model_dir / "model_results.joblib"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save results (excluding non-serializable items)
        serializable_results = {
            'cv_scores': results['cv_scores'].tolist(),
            'cv_mean': float(results['cv_mean']),
            'cv_std': float(results['cv_std']),
            'roc_auc': float(results['roc_auc']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'train_size': results['train_size'],
            'test_size': results['test_size']
        }
        joblib.dump(serializable_results, results_path)
        
        # Save feature importance as CSV
        feature_importance_path = model_dir / "feature_importance.csv"
        results['feature_importance'].to_csv(feature_importance_path, index=False)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Scaler saved to: {scaler_path}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Feature importance saved to: {feature_importance_path}")
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise

def main() -> None:
    """Main execution function with comprehensive error handling."""
    try:
        logger.info("Starting Diabetes Prediction Pipeline")
        logger.info(f"Configuration: {CONFIG}")
        
        # Validate configuration
        validate_config()
        
        # Get project root and data path
        project_root = get_project_root()
        data_path = project_root / "datasets" / "c99d9bc33649" / CONFIG['dataset_file']
        
        # Load and preprocess data
        data = load_and_validate_data(data_path)
        X, y = preprocess_data(data)
        
        # Train and evaluate model
        model, scaler, results = train_and_evaluate_model(X, y)
        
        # Save model and artifacts
        save_model_and_artifacts(model, scaler, results, project_root)
        
        logger.info("Pipeline completed successfully!")
        print(f"\n{'='*50}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Final ROC-AUC Score: {results['roc_auc']:.4f}")
        print(f"Cross-validation Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()