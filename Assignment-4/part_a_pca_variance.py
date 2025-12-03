"""
Assignment 4 - Part A: PCA Cumulative Variance Analysis
Task: Apply PCA to congressional votes and plot cumulative variance explained
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Load the votes data
print("Loading congressional votes data...")
votes = pd.read_csv(
    'datasets/congress/p1_congress_1984_votes.csv', header=None).values
print(f"Data shape: {votes.shape}")  # Should be (435, 16)

# Step 2: Apply PCA
print("\nApplying PCA...")
pca = PCA()
pca.fit(votes)

# Step 3: Get explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_
print(f"\nVariance explained by each principal component:")
print("=" * 50)
for i, var in enumerate(explained_variance):
    print(f"PC{i+1:2d}: {var:.6f} ({var*100:6.2f}%)")

# Step 4: Calculate cumulative variance
cumulative_variance = np.cumsum(explained_variance)
print(f"\nCumulative variance explained:")
print("=" * 50)
for i, cum_var in enumerate(cumulative_variance):
    print(f"First {i+1:2d} PCs: {cum_var:.6f} ({cum_var*100:6.2f}%)")

# Step 5: Plot cumulative variance
print("\nCreating cumulative variance plot...")
plt.figure(figsize=(10, 6))
k_values = np.arange(1, len(cumulative_variance) + 1)
plt.plot(k_values, cumulative_variance, 'bo-', linewidth=2, markersize=8)

# Add reference lines for common thresholds
plt.axhline(y=0.90, color='r', linestyle='--',
            linewidth=1.5, label='90% variance', alpha=0.7)
plt.axhline(y=0.95, color='g', linestyle='--',
            linewidth=1.5, label='95% variance', alpha=0.7)

plt.xlabel('Number of Principal Components (k)', fontsize=12)
plt.ylabel('Cumulative Variance Explained', fontsize=12)
plt.title('Cumulative Variance Explained by Principal Components\nCongressional Votes 1984', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xticks(k_values)
plt.ylim([0, 1.05])
plt.tight_layout()

# Save the figure
output_file = 'pca_cumulative_variance.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {output_file}")
# plt.show()  # Commented out for non-interactive use

# Step 6: Determine how many components are sufficient
print("\n" + "=" * 50)
print("ANALYSIS: How many components are sufficient?")
print("=" * 50)

# Check different thresholds
thresholds = [0.80, 0.85, 0.90, 0.95]
for threshold in thresholds:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    actual_variance = cumulative_variance[n_components - 1]
    print(
        f"To explain ≥{threshold*100:.0f}% variance: {n_components} components (actual: {actual_variance*100:.2f}%)")

print("\n" + "=" * 50)
print("RECOMMENDATION:")
print("=" * 50)
# Use 90% as the standard threshold
recommended_threshold = 0.90
n_recommended = np.argmax(cumulative_variance >= recommended_threshold) + 1
actual_var = cumulative_variance[n_recommended - 1]
print(f"Based on the 90% variance threshold (common standard),")
print(f"{n_recommended} principal components are sufficient to summarize the data.")
print(f"These {n_recommended} components explain {actual_var*100:.2f}% of the total variance.")
print(
    f"\nThis means voting behavior can be represented by {n_recommended} underlying")
print(f"dimensions instead of the original 16 votes, with minimal information loss.")
