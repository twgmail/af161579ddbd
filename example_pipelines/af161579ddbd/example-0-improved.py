import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'max_iter': 1000,
    'cv_folds': 5,
    'target_column': 'Diabetes_binary',
    'dataset_file': 'c99d9bc33649_b.csv'
}

def get_project_root():
    """Get project root directory more reliably."""
    current_file = Path(__file__).resolve()
    # Go up directories until we find the project root (contains datasets folder)
    for parent in current_file.parents:
        if (parent / "datasets").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()

def load_and_validate_data(data_path):
    """Load and validate the dataset."""
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Basic validation
        if data.empty:
            raise ValueError("Dataset is empty")
        
        if CONFIG['target_column'] not in data.columns:
            raise ValueError(f"Target column '{CONFIG['target_column']}' not found in dataset")
        
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        logger.info(f"Missing values: {data.isnull().sum().sum()}")
        
        return data
    
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data):
    """Preprocess the data for machine learning."""
    logger.info("Preprocessing data...")
    
    # Handle missing values
    if data.isnull().sum().sum() > 0:
        logger.warning("Missing values detected. Filling with median for numeric columns.")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Separate features and target
    X = data.drop(CONFIG['target_column'], axis=1)
    y = data[CONFIG['target_column']]
    
    # Check for class imbalance
    class_distribution = y.value_counts(normalize=True)
    logger.info(f"Class distribution: {class_distribution.to_dict()}")
    
    return X, y

def train_and_evaluate_model(X, y):
    """Train and evaluate the logistic regression model."""
    logger.info("Splitting data and training model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'],
        stratify=y  # Maintain class distribution
    )
    
    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(
        max_iter=CONFIG['max_iter'], 
        random_state=CONFIG['random_state']
    )
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=CONFIG['cv_folds'])
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluation
    logger.info("Model Evaluation Results:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Feature importance (coefficients for logistic regression)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, {
        'cv_scores': cv_scores,
        'roc_auc': roc_auc,
        'feature_importance': feature_importance
    }

def save_model(model, scaler, project_root):
    """Save the trained model and scaler."""
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "diabetes_model.joblib"
    scaler_path = model_dir / "scaler.joblib"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")

def main():
    """Main execution function."""
    try:
        # Get project root and data path
        project_root = get_project_root()
        logger.info(f"Project root: {project_root}")
        
        data_path = project_root / "datasets" / "c99d9bc33649" / CONFIG['dataset_file']
        
        # Load and preprocess data
        data = load_and_validate_data(data_path)
        X, y = preprocess_data(data)
        
        # Train and evaluate model
        model, scaler, results = train_and_evaluate_model(X, y)
        
        # Save model
        save_model(model, scaler, project_root)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()