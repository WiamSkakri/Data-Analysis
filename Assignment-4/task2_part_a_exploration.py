"""
Task 2 Part A: Wine Quality Classification - Data Exploration and Label Design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('seaborn-v0_8-whitegrid')

# Load datasets
red_wine = pd.read_csv('datasets/wine+quality/winequality-red.csv', sep=';')
white_wine = pd.read_csv('datasets/wine+quality/winequality-white.csv', sep=';')

print("="*80)
print("WINE QUALITY DATASET EXPLORATION")
print("="*80)

# Dataset sizes
print(f"\nRed Wine Dataset: {red_wine.shape[0]} samples, {red_wine.shape[1]} features")
print(f"White Wine Dataset: {white_wine.shape[0]} samples, {white_wine.shape[1]} features")

# Quality score statistics
print("\n" + "="*80)
print("QUALITY SCORE STATISTICS")
print("="*80)

print("\nRed Wine Quality:")
print(red_wine['quality'].describe())
print(f"Value counts:\n{red_wine['quality'].value_counts().sort_index()}")

print("\nWhite Wine Quality:")
print(white_wine['quality'].describe())
print(f"Value counts:\n{white_wine['quality'].value_counts().sort_index()}")

# Visualize original quality distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Red wine histogram
axes[0].hist(red_wine['quality'], bins=np.arange(2.5, 10.5, 1),
             color='darkred', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Quality Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Red Wine Quality Distribution\n(n={len(red_wine)})', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# White wine histogram
axes[1].hist(white_wine['quality'], bins=np.arange(2.5, 10.5, 1),
             color='gold', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Quality Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'White Wine Quality Distribution\n(n={len(white_wine)})', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Combined comparison
axes[2].hist(red_wine['quality'], bins=np.arange(2.5, 10.5, 1),
             color='darkred', alpha=0.6, label='Red Wine', edgecolor='black')
axes[2].hist(white_wine['quality'], bins=np.arange(2.5, 10.5, 1),
             color='gold', alpha=0.6, label='White Wine', edgecolor='black')
axes[2].set_xlabel('Quality Score', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].set_title('Quality Distribution Comparison', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('task2_original_quality_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: task2_original_quality_distributions.png")
plt.close()

# Analyze potential discretization thresholds
print("\n" + "="*80)
print("DISCRETIZATION ANALYSIS")
print("="*80)

# Common threshold options
thresholds = [5, 6, 7]

for threshold in thresholds:
    print(f"\n--- Threshold: Quality > {threshold} = 'Good', ≤ {threshold} = 'Bad' ---")

    # Red wine
    red_good = (red_wine['quality'] > threshold).sum()
    red_bad = (red_wine['quality'] <= threshold).sum()
    red_good_pct = red_good / len(red_wine) * 100

    # White wine
    white_good = (white_wine['quality'] > threshold).sum()
    white_bad = (white_wine['quality'] <= threshold).sum()
    white_good_pct = white_good / len(white_wine) * 100

    print(f"Red Wine:   Good={red_good:4d} ({red_good_pct:5.1f}%), Bad={red_bad:4d} ({100-red_good_pct:5.1f}%)")
    print(f"White Wine: Good={white_good:4d} ({white_good_pct:5.1f}%), Bad={white_bad:4d} ({100-white_good_pct:5.1f}%)")
    print(f"Balance difference: {abs(red_good_pct - white_good_pct):.1f}%")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
Based on the analysis above, consider:
1. Threshold = 6 (quality > 6 = Good, ≤ 6 = Bad)
   - This is a common split that separates above-average wines
   - Provides reasonable class balance

2. Threshold = 5 (quality > 5 = Good, ≤ 5 = Bad)
   - Separates below-average from average+ wines
   - May have class imbalance

Choose the threshold that best balances:
- Interpretability (what defines "good" wine?)
- Class balance (avoid extreme imbalance)
- Consistent proportions across red and white wines
""")

print("\nScript completed successfully!")
