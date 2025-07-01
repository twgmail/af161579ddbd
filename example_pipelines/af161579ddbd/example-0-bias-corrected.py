"""
Bias-Corrected Diabetes Prediction Pipeline

This script addresses the measurement bias in the CholCheck feature
and provides a more realistic diabetes prediction model.

Key Fix: Removes CholCheck feature to eliminate post-diagnosis bias.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

def load_and_clean_data():
    """Load data and remove biased features."""
    # Getting the project root
    project_root = get_project_root()
    
    # Getting the raw data file
    raw_data_file = os.path.join(
        project_root, "datasets", "c99d9bc33649", "c99d9bc33649_b.csv"
    )
    data = pd.read_csv(raw_data_file)
    
    print("🔍 ADDRESSING MEASUREMENT BIAS")
    print("="*50)
    
    # Identify and remove biased features
    biased_features = ['CholCheck']  # Features that represent post-diagnosis care
    
    print(f"Original features: {len(data.columns)}")
    print(f"Removing biased features: {biased_features}")
    
    # Show the bias before removal
    for feature in biased_features:
        if feature in data.columns:
            bias_analysis = data.groupby('Diabetes_binary')[feature].mean()
            print(f"\n{feature} bias analysis:")
            print(f"  Non-diabetic rate: {bias_analysis[0.0]:.3f} ({bias_analysis[0.0]*100:.1f}%)")
            print(f"  Diabetic rate: {bias_analysis[1.0]:.3f} ({bias_analysis[1.0]*100:.1f}%)")
            print(f"  Bias: {bias_analysis[1.0] - bias_analysis[0.0]:.3f} ({(bias_analysis[1.0] - bias_analysis[0.0])*100:.1f} pp)")
            
            # Remove the biased feature
            data = data.drop(columns=[feature])
    
    print(f"\nFeatures after bias correction: {len(data.columns)}")
    print("Remaining features:", list(data.columns))
    
    return data

def main():
    """Main execution with bias correction."""
    print("Diabetes Prediction Pipeline - Bias Corrected Version")
    print("="*60)
    
    # Load and clean data
    data = load_and_clean_data()
    
    # Prepare features and target
    X = data.drop("Diabetes_binary", axis=1)
    y = data["Diabetes_binary"]
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Diabetes prevalence: {y.mean():.3f} ({y.mean()*100:.1f}%)")
    
    # Split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("\n🤖 Training bias-corrected model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate the model
    print("\n📊 BIAS-CORRECTED MODEL RESULTS")
    print("="*50)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features (Bias-Corrected):")
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n✅ BIAS CORRECTION SUMMARY")
    print("="*50)
    print("• Removed CholCheck feature to eliminate post-diagnosis bias")
    print("• Model now predicts diabetes based on pre-diagnosis factors")
    print("• More realistic for real-world screening applications")
    print("• Lower performance is expected but more honest/generalizable")
    
    return model, scaler, feature_importance

if __name__ == "__main__":
    model, scaler, importance = main()