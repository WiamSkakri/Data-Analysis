"""
Assignment 4 - Part A: PCA Projection and Visualization
Task: Project data onto first 3 PCs and visualize with party affiliation colors
"""

from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Load the data
print("Loading data...")
votes = pd.read_csv(
    'datasets/congress/p1_congress_1984_votes.csv', header=None).values
parties = pd.read_csv(
    'datasets/congress/p1_congress_1984_party_affiliations.csv', header=None).values.ravel()

print(f"Votes shape: {votes.shape}")
print(f"Parties shape: {parties.shape}")
print(f"Party distribution: {np.unique(parties, return_counts=True)}")

# Step 2: Apply PCA and project onto first 3 components
print("\nApplying PCA...")
pca = PCA(n_components=3)  # We only need the first 3 components
votes_pca = pca.fit_transform(votes)

print(f"Transformed data shape: {votes_pca.shape}")
print(f"\nVariance explained by first 3 PCs:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f} ({pca.explained_variance_ratio_.sum()*100:.2f}%)")

# Step 3: Create color mapping for parties
# Democrats = blue, Republicans = red (traditional US political colors)
colors = np.where(parties == 'Democrat', 'blue', 'red')
color_labels = {'Democrat': 'blue', 'Republican': 'red'}

# Step 4: Create scatter plots for each PC pair
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: PC1 vs PC2
ax1 = axes[0]
for party, color in color_labels.items():
    mask = parties == party
    ax1.scatter(votes_pca[mask, 0], votes_pca[mask, 1],
                c=color, label=party, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
ax1.set_xlabel(
    f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax1.set_ylabel(
    f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax1.set_title('PC1 vs PC2', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: PC1 vs PC3
ax2 = axes[1]
for party, color in color_labels.items():
    mask = parties == party
    ax2.scatter(votes_pca[mask, 0], votes_pca[mask, 2],
                c=color, label=party, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
ax2.set_xlabel(
    f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax2.set_ylabel(
    f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)', fontsize=11)
ax2.set_title('PC1 vs PC3', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: PC2 vs PC3
ax3 = axes[2]
for party, color in color_labels.items():
    mask = parties == party
    ax3.scatter(votes_pca[mask, 1], votes_pca[mask, 2],
                c=color, label=party, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
ax3.set_xlabel(
    f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax3.set_ylabel(
    f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)', fontsize=11)
ax3.set_title('PC2 vs PC3', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle('Congressional Votes Projected onto First 3 Principal Components\nColored by Party Affiliation',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Save the figure
output_file = 'pca_party_scatter_plots.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Step 5: Analyze separation and clustering
print("\n" + "="*70)
print("ANALYSIS: Separation and Clustering")
print("="*70)

# Calculate separation for each PC pair using the mean distance between party centroids
for i, (pc1, pc2, name) in enumerate([(0, 1, 'PC1-PC2'), (0, 2, 'PC1-PC3'), (1, 2, 'PC2-PC3')]):
    dem_mask = parties == 'Democrat'
    rep_mask = parties == 'Republican'

    # Calculate centroids
    dem_centroid = np.mean(votes_pca[dem_mask][:, [pc1, pc2]], axis=0)
    rep_centroid = np.mean(votes_pca[rep_mask][:, [pc1, pc2]], axis=0)

    # Calculate distance between centroids
    centroid_distance = np.linalg.norm(dem_centroid - rep_centroid)

    # Calculate average within-party variance
    dem_variance = np.mean(np.var(votes_pca[dem_mask][:, [pc1, pc2]], axis=0))
    rep_variance = np.mean(np.var(votes_pca[rep_mask][:, [pc1, pc2]], axis=0))
    avg_within_variance = (dem_variance + rep_variance) / 2

    # Separation score (higher = better separation)
    separation_score = centroid_distance / np.sqrt(avg_within_variance)

    print(f"\n{name}:")
    print(f"  Centroid distance: {centroid_distance:.3f}")
    print(f"  Avg within-party variance: {avg_within_variance:.3f}")
    print(f"  Separation score: {separation_score:.3f}")

# Calculate overall clustering quality (silhouette-like metric)
print("\n" + "="*70)
print("CLUSTERING QUALITY:")
print("="*70)


# Convert party labels to numeric
party_numeric = np.where(parties == 'Democrat', 0, 1)

# Calculate silhouette score for first 3 PCs
silhouette = silhouette_score(votes_pca, party_numeric)
print(f"\nSilhouette score (using first 3 PCs): {silhouette:.4f}")
print(f"  (Range: -1 to 1, where 1 = perfect clustering)")

# Analyze overlap
print("\n" + "="*70)
print("KEY OBSERVATIONS:")
print("="*70)

# Find PC1 threshold that best separates parties
pc1_dem_mean = np.mean(votes_pca[parties == 'Democrat', 0])
pc1_rep_mean = np.mean(votes_pca[parties == 'Republican', 0])

print(f"\nPC1 (primary axis of variation):")
print(f"  Democrat mean: {pc1_dem_mean:.3f}")
print(f"  Republican mean: {pc1_rep_mean:.3f}")
print(f"  Difference: {abs(pc1_dem_mean - pc1_rep_mean):.3f}")

if pc1_dem_mean < pc1_rep_mean:
    print(f"  → Democrats tend to have lower PC1 scores")
    print(f"  → Republicans tend to have higher PC1 scores")
else:
    print(f"  → Democrats tend to have higher PC1 scores")
    print(f"  → Republicans tend to have lower PC1 scores")
