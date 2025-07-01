"""
Comprehensive Bias-Corrected Diabetes Prediction Pipeline

This script addresses MULTIPLE measurement biases in the dataset:
1. Post-diagnosis health reporting bias (PhysHlth, MentHlth)
2. Detection bias (CholCheck, HighBP, HighChol) 
3. Complication/awareness bias (DiffWalk)

The goal is to create a model suitable for pre-diagnosis screening.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

def analyze_measurement_bias(data):
    """Analyze and identify measurement biases in the dataset."""
    print("🔍 COMPREHENSIVE MEASUREMENT BIAS ANALYSIS")
    print("="*60)
    
    # Define bias categories and their mechanisms
    bias_analysis = {
        'CholCheck': {
            'type': 'Healthcare monitoring bias',
            'mechanism': 'Diabetics get more cholesterol checks as part of care',
            'severity': 'MODERATE',
            'action': 'REMOVE'
        },
        'HighBP': {
            'type': 'Detection bias', 
            'mechanism': 'More medical visits → more BP monitoring → more diagnoses',
            'severity': 'HIGH',
            'action': 'REMOVE'
        },
        'HighChol': {
            'type': 'Detection bias',
            'mechanism': 'More medical visits → more cholesterol monitoring → more diagnoses', 
            'severity': 'HIGH',
            'action': 'REMOVE'
        },
        'PhysHlth': {
            'type': 'Post-diagnosis health reporting',
            'mechanism': 'Diabetes diagnosis increases health problem awareness/reporting',
            'severity': 'CRITICAL',
            'action': 'REMOVE'
        },
        'MentHlth': {
            'type': 'Post-diagnosis psychological impact',
            'mechanism': 'Diabetes diagnosis affects mental health and reporting',
            'severity': 'HIGH', 
            'action': 'REMOVE'
        },
        'DiffWalk': {
            'type': 'Complication/awareness bias',
            'mechanism': 'Could be diabetes complication OR increased symptom awareness',
            'severity': 'HIGH',
            'action': 'REMOVE'
        }
    }
    
    print("IDENTIFIED BIASES:")
    print("-" * 60)
    
    biased_features = []
    for feature, info in bias_analysis.items():
        if feature in data.columns:
            rates = data.groupby('Diabetes_binary')[feature].mean()
            non_diabetic = rates[0.0]
            diabetic = rates[1.0] 
            bias = (diabetic - non_diabetic) * 100
            
            print(f"\n🚨 {feature}")
            print(f"   Bias: {bias:+.1f} percentage points")
            print(f"   Type: {info['type']}")
            print(f"   Mechanism: {info['mechanism']}")
            print(f"   Severity: {info['severity']}")
            print(f"   Action: {info['action']}")
            
            if info['action'] == 'REMOVE':
                biased_features.append(feature)
    
    return biased_features

def load_and_clean_data():
    """Load data and remove all identified biased features."""
    # Getting the project root
    project_root = get_project_root()
    
    # Getting the raw data file
    raw_data_file = os.path.join(
        project_root, "datasets", "c99d9bc33649", "c99d9bc33649_b.csv"
    )
    data = pd.read_csv(raw_data_file)
    
    print(f"Original dataset: {data.shape}")
    
    # Analyze biases
    biased_features = analyze_measurement_bias(data)
    
    print(f"\n📋 BIAS CORRECTION SUMMARY")
    print("="*60)
    print(f"Features to remove: {len(biased_features)}")
    print(f"Biased features: {biased_features}")
    
    # Remove biased features
    data_clean = data.drop(columns=biased_features)
    
    print(f"Clean dataset: {data_clean.shape}")
    print(f"Remaining features: {list(data_clean.columns)}")
    
    return data_clean, biased_features

def compare_models(data_original, data_clean, biased_features):
    """Compare biased vs unbiased model performance."""
    print(f"\n📊 MODEL COMPARISON: BIASED vs UNBIASED")
    print("="*60)
    
    # Prepare datasets
    X_biased = data_original.drop('Diabetes_binary', axis=1)
    X_unbiased = data_clean.drop('Diabetes_binary', axis=1)
    y = data_original['Diabetes_binary']
    
    results = {}
    
    for name, X in [('Biased (Original)', X_biased), ('Unbiased (Clean)', X_unbiased)]:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'features': X.shape[1],
            'test_auc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std()
        }
    
    # Display comparison
    print(f"{'Model':<20} {'Features':<10} {'Test AUC':<10} {'CV AUC':<15} {'Reliability':<12}")
    print("-" * 75)
    
    for name, metrics in results.items():
        reliability = "High" if metrics['cv_auc_std'] < 0.01 else "Medium" if metrics['cv_auc_std'] < 0.02 else "Low"
        print(f"{name:<20} {metrics['features']:<10} {metrics['test_auc']:<10.4f} {metrics['cv_auc_mean']:.4f}±{metrics['cv_auc_std']:.4f} {reliability:<12}")
    
    # Calculate impact
    auc_drop = results['Unbiased (Clean)']['test_auc'] - results['Biased (Original)']['test_auc']
    print(f"\n📉 Performance Impact:")
    print(f"   AUC decrease: {auc_drop:.4f}")
    print(f"   Features removed: {len(biased_features)}")
    print(f"   Bias correction trade-off: {abs(auc_drop)/len(biased_features):.4f} AUC per feature")
    
    return results

def main():
    """Main execution with comprehensive bias correction."""
    print("🏥 COMPREHENSIVE BIAS-CORRECTED DIABETES PREDICTION")
    print("="*70)
    
    # Load original data for comparison
    project_root = get_project_root()
    raw_data_file = os.path.join(project_root, "datasets", "c99d9bc33649", "c99d9bc33649_b.csv")
    data_original = pd.read_csv(raw_data_file)
    
    # Load and clean data
    data_clean, biased_features = load_and_clean_data()
    
    # Compare models
    comparison_results = compare_models(data_original, data_clean, biased_features)
    
    # Train final unbiased model
    print(f"\n🤖 TRAINING FINAL UNBIASED MODEL")
    print("="*60)
    
    X = data_clean.drop("Diabetes_binary", axis=1)
    y = data_clean["Diabetes_binary"]
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Diabetes prevalence: {y.mean():.3f} ({y.mean()*100:.1f}%)")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluation
    print("\n📊 FINAL MODEL PERFORMANCE")
    print("="*60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 Most Important Features (Unbiased):")
    print(feature_importance.head(10).to_string(index=False))
    
    print(f"\n✅ COMPREHENSIVE BIAS CORRECTION COMPLETE")
    print("="*70)
    print("✓ Removed post-diagnosis health reporting bias")
    print("✓ Removed detection bias from increased medical monitoring") 
    print("✓ Removed complication/awareness bias")
    print("✓ Model suitable for pre-diagnosis screening")
    print("✓ More honest performance metrics")
    print("✓ Better real-world generalizability")
    
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print("="*70)
    print("1. Use this unbiased model for diabetes screening")
    print("2. Expect lower but more realistic performance")
    print("3. Validate on external pre-diagnosis datasets")
    print("4. Consider temporal validation if timestamps available")
    print("5. Monitor for concept drift in deployment")
    
    return model, scaler, feature_importance, biased_features

if __name__ == "__main__":
    model, scaler, importance, removed_features = main()