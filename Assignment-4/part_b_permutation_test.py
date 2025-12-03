"""
Assignment 4 - Part B: Permutation Test for Clustering Significance
Task: Assess statistical significance of clustering using permutation tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Step 1: Load the data
print("Loading congressional votes data...")
votes = pd.read_csv('datasets/congress/p1_congress_1984_votes.csv', header=None).values
print(f"Data shape: {votes.shape}")

# Step 2: Define clustering quality score
# We'll use the silhouette score as our quality metric
# Higher silhouette score = better clustering
# Range: -1 to 1, where 1 = perfect clustering

def compute_clustering_score(data):
    """
    Compute clustering quality score using silhouette coefficient.

    Args:
        data: voting data (435 x 16)

    Returns:
        silhouette score (float)
    """
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    return score

# Step 3: Compute score on original dataset
print("\n" + "="*70)
print("COMPUTING CLUSTERING SCORE ON ORIGINAL DATA")
print("="*70)

original_score = compute_clustering_score(votes)
print(f"\nClustering Quality Score (Silhouette):")
print(f"  Original dataset: {original_score:.4f}")
print(f"  (Range: -1 to 1, where 1 = perfect clustering)")

# Step 4: Permutation test
print("\n" + "="*70)
print("PERMUTATION TEST")
print("="*70)

n_permutations = 1000
print(f"\nRunning {n_permutations} permutations...")
print("This will take a few minutes...\n")

permuted_scores = []

for i in tqdm(range(n_permutations), desc="Permutations"):
    # Create permuted dataset
    # For each congress member, randomly shuffle their votes across the 16 issues
    # This destroys the correlation structure while preserving vote distributions
    permuted_data = votes.copy()

    # Permute each member's votes independently
    for member_idx in range(votes.shape[0]):
        np.random.shuffle(permuted_data[member_idx, :])

    # Cluster the permuted data
    score = compute_clustering_score(permuted_data)
    permuted_scores.append(score)

permuted_scores = np.array(permuted_scores)

# Step 5: Statistical analysis
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# Calculate p-value (one-tailed test)
# How many permuted scores are >= original score?
p_value = np.sum(permuted_scores >= original_score) / n_permutations

print(f"\nPermuted data statistics:")
print(f"  Mean:   {permuted_scores.mean():.4f}")
print(f"  Std:    {permuted_scores.std():.4f}")
print(f"  Min:    {permuted_scores.min():.4f}")
print(f"  Max:    {permuted_scores.max():.4f}")
print(f"  Median: {np.median(permuted_scores):.4f}")

print(f"\nOriginal vs Permuted:")
print(f"  Original score:     {original_score:.4f}")
print(f"  Mean permuted:      {permuted_scores.mean():.4f}")
print(f"  Difference:         {original_score - permuted_scores.mean():.4f}")
print(f"  Z-score:            {(original_score - permuted_scores.mean()) / permuted_scores.std():.2f}")

print(f"\n" + "="*70)
print(f"P-VALUE: {p_value:.4f}")
print(f"  (Probability of obtaining a score >= {original_score:.4f} by chance)")
print("="*70)

# Significance level
alpha = 0.05
if p_value < alpha:
    print(f"\n✓ CONCLUSION: Clustering is STATISTICALLY SIGNIFICANT (p < {alpha})")
    print(f"  The original dataset has significantly better clustering than random data.")
    print(f"  The voting structure is real, not due to chance.")
else:
    print(f"\n✗ CONCLUSION: Clustering is NOT statistically significant (p >= {alpha})")
    print(f"  Cannot reject the null hypothesis of random structure.")

# Step 6: Visualization
print("\n" + "="*70)
print("CREATING VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Histogram of permuted scores
ax1 = axes[0]
ax1.hist(permuted_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(original_score, color='red', linestyle='--', linewidth=3,
            label=f'Original score = {original_score:.4f}')
ax1.axvline(permuted_scores.mean(), color='orange', linestyle='-', linewidth=2,
            label=f'Mean permuted = {permuted_scores.mean():.4f}')
ax1.set_xlabel('Silhouette Score', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title(f'Distribution of Clustering Scores\n({n_permutations} Permutations)',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add text box with p-value
textstr = f'p-value = {p_value:.4f}\nα = {alpha}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# Plot 2: Cumulative distribution
ax2 = axes[1]
sorted_scores = np.sort(permuted_scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
ax2.plot(sorted_scores, cumulative, linewidth=2, color='skyblue', label='Permuted data CDF')
ax2.axvline(original_score, color='red', linestyle='--', linewidth=3,
            label=f'Original score = {original_score:.4f}')

# Mark the percentile of the original score
percentile = np.sum(permuted_scores < original_score) / n_permutations * 100
ax2.axhline(percentile/100, color='green', linestyle=':', linewidth=2, alpha=0.7,
            label=f'Percentile = {percentile:.1f}%')

ax2.set_xlabel('Silhouette Score', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.suptitle('Permutation Test for Clustering Significance',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

output_file = 'permutation_test_results.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Step 7: Additional analysis - effect size
print("\n" + "="*70)
print("EFFECT SIZE ANALYSIS")
print("="*70)

# Cohen's d effect size
cohens_d = (original_score - permuted_scores.mean()) / permuted_scores.std()
print(f"\nCohen's d: {cohens_d:.2f}")
if abs(cohens_d) < 0.2:
    effect_size_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_size_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_size_interpretation = "medium"
else:
    effect_size_interpretation = "large"
print(f"Effect size: {effect_size_interpretation}")
print(f"  (|d| < 0.2: negligible, < 0.5: small, < 0.8: medium, >= 0.8: large)")

# Probability of superiority
prob_superiority = np.sum(permuted_scores < original_score) / n_permutations
print(f"\nProbability of superiority: {prob_superiority:.4f}")
print(f"  ({prob_superiority*100:.1f}% of permuted datasets have worse clustering)")

print("\n" + "="*70)
print("FINAL INTERPRETATION")
print("="*70)
print(f"""
The permutation test reveals that:

1. Original clustering score: {original_score:.4f}
2. Mean permuted score:      {permuted_scores.mean():.4f}
3. P-value:                  {p_value:.4f}
4. Effect size (Cohen's d):  {cohens_d:.2f} ({effect_size_interpretation})

CONCLUSION:
The clustering found in the original congressional voting data is
{'STATISTICALLY SIGNIFICANT' if p_value < alpha else 'NOT statistically significant'}.

EXPLANATION:
When we randomly shuffle each congress member's votes (destroying any real
voting patterns), the resulting clusters have {'much worse' if cohens_d > 0.8 else 'worse'} quality scores
({prob_superiority*100:.1f}% of permutations). This demonstrates that the original
voting patterns contain real structure that meaningfully separates congress
members into two distinct groups (which correspond to party affiliations).

The {'very large' if cohens_d > 0.8 else 'large' if cohens_d > 0.5 else 'moderate'} effect size (d = {cohens_d:.2f}) indicates that this difference is not
only statistically significant but also practically meaningful.
""")
