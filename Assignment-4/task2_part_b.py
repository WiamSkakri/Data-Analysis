"""
Task 2 Part B: Model Training and Evaluation
Author: Wiam Skakri

This script trains two classification models (Logistic Regression and Neural Network)
and evaluates their performance both within and across wine types.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import pickle
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set matplotlib backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

print("="*80)
print("TASK 2 PART B: MODEL TRAINING AND EVALUATION")
print("="*80)

# ============================================================================
# 1. LOAD PROCESSED DATASETS
# ============================================================================
print("\n[1] Loading processed datasets...")

red_wine = pd.read_csv('datasets/wine+quality/red_wine_labeled.csv')
white_wine = pd.read_csv('datasets/wine+quality/white_wine_labeled.csv')

print(f"   ✓ Red wine: {len(red_wine)} samples")
print(f"   ✓ White wine: {len(white_wine)} samples")

# Features (11 physicochemical attributes)
feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                   'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                   'pH', 'sulphates', 'alcohol']

# Target variable
target_column = 'quality_label'  # 0 = Bad, 1 = Good

# ============================================================================
# 2. TRAIN/TEST SPLIT (80/20)
# ============================================================================
print("\n[2] Splitting data into training and testing sets (80/20)...")

# Red wine split
X_red = red_wine[feature_columns]
y_red = red_wine[target_column]

X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(
    X_red, y_red, test_size=0.2, random_state=42, stratify=y_red
)

# White wine split
X_white = white_wine[feature_columns]
y_white = white_wine[target_column]

X_white_train, X_white_test, y_white_train, y_white_test = train_test_split(
    X_white, y_white, test_size=0.2, random_state=42, stratify=y_white
)

print(f"\n   Red wine:")
print(f"      Training: {len(X_red_train)} samples")
print(f"      Testing:  {len(X_red_test)} samples")
print(f"\n   White wine (before downsampling):")
print(f"      Training: {len(X_white_train)} samples")
print(f"      Testing:  {len(X_white_test)} samples")

# ============================================================================
# 3. DOWNSAMPLE WHITE WINE TRAINING SET
# ============================================================================
print("\n[3] Downsampling white wine training set to match red wine size...")

# Target size: match red wine training set (~1280 samples)
target_size = len(X_red_train)

# Stratified sampling to maintain class balance
white_train_combined = pd.concat([X_white_train, y_white_train], axis=1)
white_train_downsampled = white_train_combined.groupby(target_column, group_keys=False).apply(
    lambda x: x.sample(n=int(target_size * len(x) / len(white_train_combined)), random_state=42)
)

X_white_train_downsampled = white_train_downsampled[feature_columns]
y_white_train_downsampled = white_train_downsampled[target_column]

print(f"   ✓ Downsampled white wine training set: {len(X_white_train_downsampled)} samples")
print(f"   ✓ Class distribution: Good={sum(y_white_train_downsampled==1)}, Bad={sum(y_white_train_downsampled==0)}")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
print("\n[4] Standardizing features...")

# Create separate scalers for red and white wine
scaler_red = StandardScaler()
scaler_white = StandardScaler()

# Fit and transform training data
X_red_train_scaled = scaler_red.fit_transform(X_red_train)
X_white_train_scaled = scaler_white.fit_transform(X_white_train_downsampled)

# Transform test data
X_red_test_scaled = scaler_red.transform(X_red_test)
X_white_test_scaled = scaler_white.transform(X_white_test)

print("   ✓ Features standardized (mean=0, std=1)")

# ============================================================================
# 5. MODEL SELECTION
# ============================================================================
print("\n[5] Initializing classification models...")

# Model 1: Logistic Regression (simple, linear)
lr_params = {
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs'
}

# Model 2: Neural Network (complex, non-linear)
nn_params = {
    'hidden_layer_sizes': (64, 32),  # Two hidden layers: 64 and 32 neurons
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,  # L2 regularization
    'batch_size': 32,
    'learning_rate': 'adaptive',
    'max_iter': 500,
    'random_state': 42,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 20
}

print(f"   Model 1: Logistic Regression")
print(f"      Parameters: {lr_params}")
print(f"\n   Model 2: Neural Network (MLP)")
print(f"      Architecture: Input → 64 neurons → 32 neurons → Output")
print(f"      Parameters: {nn_params}")

# ============================================================================
# 6. TRAINING AND EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name, train_type, test_type):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    return {
        'model': model_name,
        'train_on': train_type,
        'test_on': test_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'predictions': y_pred,
        'true_labels': y_test
    }

# Dictionary to store all results
results = []

print("\n[6] Training and evaluating models...")
print("\n" + "="*80)
print("IN-DOMAIN TESTING")
print("="*80)

# ----------------------------------------------------------------------------
# IN-DOMAIN: Red → Red
# ----------------------------------------------------------------------------
print("\n--- Logistic Regression: Red → Red ---")
lr_red = LogisticRegression(**lr_params)
lr_red.fit(X_red_train_scaled, y_red_train)
result = evaluate_model(lr_red, X_red_test_scaled, y_red_test,
                        'Logistic Regression', 'Red', 'Red')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

print("\n--- Neural Network: Red → Red ---")
nn_red = MLPClassifier(**nn_params)
nn_red.fit(X_red_train_scaled, y_red_train)
result = evaluate_model(nn_red, X_red_test_scaled, y_red_test,
                        'Neural Network', 'Red', 'Red')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

# ----------------------------------------------------------------------------
# IN-DOMAIN: White → White
# ----------------------------------------------------------------------------
print("\n--- Logistic Regression: White → White ---")
lr_white = LogisticRegression(**lr_params)
lr_white.fit(X_white_train_scaled, y_white_train_downsampled)
result = evaluate_model(lr_white, X_white_test_scaled, y_white_test,
                        'Logistic Regression', 'White', 'White')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

print("\n--- Neural Network: White → White ---")
nn_white = MLPClassifier(**nn_params)
nn_white.fit(X_white_train_scaled, y_white_train_downsampled)
result = evaluate_model(nn_white, X_white_test_scaled, y_white_test,
                        'Neural Network', 'White', 'White')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

# ----------------------------------------------------------------------------
# CROSS-DOMAIN: Red → White
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("CROSS-DOMAIN TESTING")
print("="*80)

print("\n--- Logistic Regression: Red → White ---")
# Scale white test data using red scaler for fair comparison
X_white_test_scaled_red = scaler_red.transform(X_white_test)
result = evaluate_model(lr_red, X_white_test_scaled_red, y_white_test,
                        'Logistic Regression', 'Red', 'White')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

print("\n--- Neural Network: Red → White ---")
result = evaluate_model(nn_red, X_white_test_scaled_red, y_white_test,
                        'Neural Network', 'Red', 'White')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

# ----------------------------------------------------------------------------
# CROSS-DOMAIN: White → Red
# ----------------------------------------------------------------------------
print("\n--- Logistic Regression: White → Red ---")
# Scale red test data using white scaler for fair comparison
X_red_test_scaled_white = scaler_white.transform(X_red_test)
result = evaluate_model(lr_white, X_red_test_scaled_white, y_red_test,
                        'Logistic Regression', 'White', 'Red')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

print("\n--- Neural Network: White → Red ---")
result = evaluate_model(nn_white, X_red_test_scaled_white, y_red_test,
                        'Neural Network', 'White', 'Red')
results.append(result)
print(f"   Accuracy:  {result['accuracy']:.4f}")
print(f"   Precision: {result['precision']:.4f}")
print(f"   Recall:    {result['recall']:.4f}")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n[7] Saving results...")

# Convert results to DataFrame
results_df = pd.DataFrame([{
    'Model': r['model'],
    'Train On': r['train_on'],
    'Test On': r['test_on'],
    'Accuracy': r['accuracy'],
    'Precision': r['precision'],
    'Recall': r['recall']
} for r in results])

results_df.to_csv('task2_part_b_results.csv', index=False)
print("   ✓ Saved: task2_part_b_results.csv")

# Save full results with predictions for Part C
with open('task2_part_b_full_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("   ✓ Saved: task2_part_b_full_results.pkl")

# Save models
models = {
    'lr_red': lr_red,
    'lr_white': lr_white,
    'nn_red': nn_red,
    'nn_white': nn_white,
    'scaler_red': scaler_red,
    'scaler_white': scaler_white
}

with open('task2_part_b_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("   ✓ Saved: task2_part_b_models.pkl")

# ============================================================================
# 8. RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print("\n" + results_df.to_string(index=False))

print("\n" + "="*80)
print("PART B COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nNext steps:")
print("  - Review results in task2_part_b_results.csv")
print("  - Proceed to Part C: Analysis and Interpretation")
