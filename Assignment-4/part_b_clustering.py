"""
Assignment 4 - Part B: Clustering Analysis
Task: Cluster congress members into 2 groups and compare with party affiliations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_rand_score

# Step 1: Load the data
print("Loading data...")
votes = pd.read_csv('datasets/congress/p1_congress_1984_votes.csv', header=None).values
parties = pd.read_csv('datasets/congress/p1_congress_1984_party_affiliations.csv', header=None).values.ravel()

print(f"Votes shape: {votes.shape}")
print(f"Party distribution: {np.unique(parties, return_counts=True)}")

# Step 2: Apply K-means clustering with k=2
print("\n" + "="*70)
print("CLUSTERING ALGORITHM: K-Means")
print("="*70)
print("\nAlgorithm Details:")
print("- Algorithm: K-Means clustering")
print("- Number of clusters (k): 2")
print("- Distance metric: Euclidean distance")
print("- Initialization: k-means++ (smart initialization)")
print("- Random state: 42 (for reproducibility)")

print("\nHow K-Means works:")
print("1. Initialize 2 cluster centers randomly (using k-means++ strategy)")
print("2. Assign each congress member to the nearest cluster center")
print("3. Update cluster centers to be the mean of assigned members")
print("4. Repeat steps 2-3 until convergence")
print("\nDistance function: Euclidean distance in 16-dimensional vote space")
print("  d(x, y) = sqrt(sum((x_i - y_i)^2)) for all 16 voting dimensions")

# Perform clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(votes)

print(f"\nClustering complete!")
print(f"Cluster distribution: {np.unique(cluster_labels, return_counts=True)}")
print(f"Cluster 0: {np.sum(cluster_labels == 0)} members")
print(f"Cluster 1: {np.sum(cluster_labels == 1)} members")

# Step 3: Project onto first 2 PCs for visualization
print("\nProjecting data onto first 2 principal components for visualization...")
pca = PCA(n_components=2)
votes_pca = pca.fit_transform(votes)
print(f"Variance explained by first 2 PCs: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Step 4: Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Clusters (unsupervised result)
ax1 = axes[0]
colors_cluster = ['purple', 'orange']
for cluster in [0, 1]:
    mask = cluster_labels == cluster
    ax1.scatter(votes_pca[mask, 0], votes_pca[mask, 1],
               c=colors_cluster[cluster], label=f'Cluster {cluster}',
               alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax1.set_title('K-Means Clustering Result\n(Unsupervised)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
ax1.scatter(centers_pca[:, 0], centers_pca[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidths=2,
           label='Cluster Centers', zorder=5)
ax1.legend(fontsize=11)

# Plot 2: Actual party affiliations (ground truth)
ax2 = axes[1]
colors_party = {'Democrat': 'blue', 'Republican': 'red'}
for party, color in colors_party.items():
    mask = parties == party
    ax2.scatter(votes_pca[mask, 0], votes_pca[mask, 1],
               c=color, label=party, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax2.set_title('Actual Party Affiliations\n(Ground Truth)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.suptitle('Clustering vs Party Affiliation Comparison', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()

output_file = 'clustering_vs_party.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {output_file}")

# Step 5: Analyze agreement with party affiliations
print("\n" + "="*70)
print("ANALYSIS: Comparison with Party Affiliations")
print("="*70)

# Convert party labels to numeric for confusion matrix
party_numeric = np.where(parties == 'Democrat', 0, 1)
party_names = ['Democrat', 'Republican']

# Create confusion matrix
# Note: Need to check which cluster corresponds to which party
conf_matrix = confusion_matrix(party_numeric, cluster_labels)
print("\nConfusion Matrix:")
print("                Cluster 0  Cluster 1")
print(f"Democrat:       {conf_matrix[0,0]:6d}     {conf_matrix[0,1]:6d}")
print(f"Republican:     {conf_matrix[1,0]:6d}     {conf_matrix[1,1]:6d}")

# Determine which cluster corresponds to which party
# The cluster with more Democrats is the "Democrat cluster"
if conf_matrix[0,0] > conf_matrix[0,1]:
    dem_cluster, rep_cluster = 0, 1
else:
    dem_cluster, rep_cluster = 1, 0

print(f"\nCluster {dem_cluster} predominantly contains Democrats")
print(f"Cluster {rep_cluster} predominantly contains Republicans")

# Calculate accuracy (best possible alignment)
correct_assignments = max(
    conf_matrix[0,0] + conf_matrix[1,1],  # Cluster 0=Dem, 1=Rep
    conf_matrix[0,1] + conf_matrix[1,0]   # Cluster 0=Rep, 1=Dem
)
accuracy = correct_assignments / len(parties)

print(f"\n" + "="*70)
print(f"CLUSTERING ACCURACY: {accuracy*100:.2f}%")
print(f"Correctly clustered: {correct_assignments}/{len(parties)} members")
print(f"Incorrectly clustered: {len(parties) - correct_assignments}/{len(parties)} members")
print("="*70)

# Calculate Adjusted Rand Index (measures agreement, accounting for chance)
ari = adjusted_rand_score(party_numeric, cluster_labels)
print(f"\nAdjusted Rand Index: {ari:.4f}")
print(f"  (Range: -1 to 1, where 1 = perfect agreement, 0 = random)")

# Detailed breakdown
print("\n" + "="*70)
print("DETAILED BREAKDOWN:")
print("="*70)

dem_in_dem_cluster = conf_matrix[0, dem_cluster]
dem_in_rep_cluster = conf_matrix[0, rep_cluster]
rep_in_rep_cluster = conf_matrix[1, rep_cluster]
rep_in_dem_cluster = conf_matrix[1, dem_cluster]

total_dems = dem_in_dem_cluster + dem_in_rep_cluster
total_reps = rep_in_rep_cluster + rep_in_dem_cluster

print(f"\nDemocrats (total: {total_dems}):")
print(f"  Correctly clustered: {dem_in_dem_cluster} ({dem_in_dem_cluster/total_dems*100:.1f}%)")
print(f"  Incorrectly clustered: {dem_in_rep_cluster} ({dem_in_rep_cluster/total_dems*100:.1f}%)")

print(f"\nRepublicans (total: {total_reps}):")
print(f"  Correctly clustered: {rep_in_rep_cluster} ({rep_in_rep_cluster/total_reps*100:.1f}%)")
print(f"  Incorrectly clustered: {rep_in_dem_cluster} ({rep_in_dem_cluster/total_reps*100:.1f}%)")

# Visual separation analysis
print("\n" + "="*70)
print("VISUAL SEPARATION ANALYSIS:")
print("="*70)

print("\nAre the groups visually separated in PC1-PC2 space?")
# Calculate distance between cluster centers in PC space
center_distance = np.linalg.norm(centers_pca[0] - centers_pca[1])
print(f"  Distance between cluster centers (in PC space): {center_distance:.3f}")

# Calculate average within-cluster spread
cluster0_spread = np.mean(np.std(votes_pca[cluster_labels == 0], axis=0))
cluster1_spread = np.mean(np.std(votes_pca[cluster_labels == 1], axis=0))
avg_spread = (cluster0_spread + cluster1_spread) / 2
print(f"  Average within-cluster spread: {avg_spread:.3f}")
print(f"  Separation ratio: {center_distance/avg_spread:.3f}")

if center_distance/avg_spread > 2:
    print("\n  → YES: Groups are well-separated (separation ratio > 2)")
elif center_distance/avg_spread > 1:
    print("\n  → MODERATE: Groups show some separation but with overlap")
else:
    print("\n  → NO: Groups are poorly separated")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print(f"The K-means clustering achieves {accuracy*100:.1f}% agreement with party affiliations.")
print(f"This demonstrates that voting patterns strongly correlate with party membership,")
print(f"confirming the partisan nature of congressional voting in 1984.")
