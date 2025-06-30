"""
Improved diabetes prediction pipeline with better error handling,
data preprocessing, and model evaluation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_auc_score
)

# Configuration constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000
CV_FOLDS = 5
TARGET_COLUMN = "Diabetes_binary"
DATASET_PATH = "datasets/c99d9bc33649/c99d9bc33649_b.csv"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory by looking for specific marker files.
    
    Returns:
        Path: Project root directory
    """
    current_path = Path(__file__).resolve()
    
    # Look for common project markers
    for parent in current_path.parents:
        if any((parent / marker).exists() for marker in ['.git', 'requirements.txt', 'setup.py']):
            return parent
    
    # Fallback to current working directory
    return Path.cwd()


def load_and_validate_data(file_path: Path) -> pd.DataFrame:
    """
    Load and validate the dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and validated dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the data is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with shape: {data.shape}")
        
        # Validate target column exists
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Dataset contains {missing_values} missing values")
        
        # Basic data validation
        if data.empty:
            raise ValueError("Dataset is empty")
        
        logger.info(f"Dataset validation passed. Columns: {list(data.columns)}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the data for machine learning.
    
    Args:
        data: Raw dataset
        
    Returns:
        Tuple of (features, target) arrays
    """
    # Handle missing values (if any)
    if data.isnull().any().any():
        logger.info("Handling missing values...")
        # For numerical columns, fill with median
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Separate features and target
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    
    # Validate target values
    unique_targets = y.unique()
    logger.info(f"Target variable unique values: {unique_targets}")
    
    if len(unique_targets) != 2:
        logger.warning(f"Expected binary classification, found {len(unique_targets)} classes")
    
    return X.values, y.values


def train_and_evaluate_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Train and evaluate the logistic regression model.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        LogisticRegression: Trained model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set size: {X_train_scaled.shape[0]}")
    logger.info(f"Test set size: {X_test_scaled.shape[0]}")
    
    # Train the model
    model = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    
    try:
        model.fit(X_train_scaled, y_train)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Model Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=CV_FOLDS, scoring='accuracy')
    print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model


def main():
    """Main execution function."""
    try:
        logger.info("Starting diabetes prediction pipeline...")
        
        # Get project root and construct data path
        project_root = get_project_root()
        data_file = project_root / DATASET_PATH
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Looking for dataset at: {data_file}")
        
        # Load and validate data
        data = load_and_validate_data(data_file)
        
        # Preprocess data
        X, y = preprocess_data(data)
        
        # Train and evaluate model
        model = train_and_evaluate_model(X, y)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()