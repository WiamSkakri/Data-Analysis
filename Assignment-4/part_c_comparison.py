"""
Assignment 4 - Part C: Clustering Comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (mutual_info_score, normalized_mutual_info_score,
                            adjusted_mutual_info_score, adjusted_rand_score,
                            silhouette_score)

# Load data
votes = pd.read_csv('datasets/congress/p1_congress_1984_votes.csv', header=None).values
parties = pd.read_csv('datasets/congress/p1_congress_1984_party_affiliations.csv', header=None).values.ravel()
party_numeric = np.where(parties == 'Democrat', 0, 1)

print("Part C: Clustering Comparison\n")

# ============================================================================
# Task 1: Quantify agreement using Mutual Information (all 16 votes)
# ============================================================================

print("Task 1: Mutual Information (using all 16 votes)")
kmeans_full = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_full = kmeans_full.fit_predict(votes)

mi_full = mutual_info_score(parties, clusters_full)
nmi_full = normalized_mutual_info_score(parties, clusters_full)
ami_full = adjusted_mutual_info_score(parties, clusters_full)

print(f"  MI:  {mi_full:.4f}")
print(f"  NMI: {nmi_full:.4f}")
print(f"  AMI: {ami_full:.4f}\n")

# ============================================================================
# Task 2: Clustering on first 2 PCs
# ============================================================================

print("Task 2: Clustering on first 2 principal components")
pca = PCA(n_components=2)
votes_pca = pca.fit_transform(votes)

print(f"  PC1 variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"  PC2 variance: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"  Total: {pca.explained_variance_ratio_.sum()*100:.2f}%\n")

kmeans_pca = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_pca = kmeans_pca.fit_predict(votes_pca)

mi_pca = mutual_info_score(parties, clusters_pca)
nmi_pca = normalized_mutual_info_score(parties, clusters_pca)
ami_pca = adjusted_mutual_info_score(parties, clusters_pca)

print(f"  MI:  {mi_pca:.4f}")
print(f"  NMI: {nmi_pca:.4f}")
print(f"  AMI: {ami_pca:.4f}\n")

# ============================================================================
# Comparison
# ============================================================================

print("Comparison:")
print(f"{'Metric':<20} {'All 16 Votes':>15} {'First 2 PCs':>15} {'Difference':>15}")
print("-" * 65)

metrics = [
    ('MI', mi_full, mi_pca),
    ('NMI', nmi_full, nmi_pca),
    ('AMI', ami_full, ami_pca),
    ('ARI', adjusted_rand_score(party_numeric, clusters_full),
           adjusted_rand_score(party_numeric, clusters_pca))
]

for name, val_full, val_pca in metrics:
    diff = val_full - val_pca
    print(f"{name:<20} {val_full:>15.4f} {val_pca:>15.4f} {diff:>+15.4f}")

print(f"\nAnswer: {'All 16 votes' if nmi_full > nmi_pca else 'First 2 PCs'} agrees more with party affiliations")
print(f"  (NMI: {max(nmi_full, nmi_pca):.4f} vs {min(nmi_full, nmi_pca):.4f})\n")

# Why?
if nmi_full > nmi_pca:
    print("Why? Using all 16 votes captures more information about voting patterns.")
    print(f"  PC1-PC2 only captures {pca.explained_variance_ratio_.sum()*100:.1f}% of variance.")
    print(f"  Missing {100 - pca.explained_variance_ratio_.sum()*100:.1f}% may contain party-distinguishing information.")
else:
    print("Why? PCA reduces noise and focuses on main party-separating dimensions.")
    print("  PC1 strongly aligns with party differences.")
    print("  Lower dimensions avoid curse of dimensionality with 435 samples.")

# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: All 16 votes (projected to PC space)
ax1 = axes[0, 0]
colors = ['purple', 'orange']
for c in [0, 1]:
    mask = clusters_full == c
    ax1.scatter(votes_pca[mask, 0], votes_pca[mask, 1], c=colors[c],
               label=f'Cluster {c}', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax1.set_title(f'Clustering on All 16 Votes\nNMI = {nmi_full:.4f}', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: First 2 PCs
ax2 = axes[0, 1]
for c in [0, 1]:
    mask = clusters_pca == c
    ax2.scatter(votes_pca[mask, 0], votes_pca[mask, 1], c=colors[c],
               label=f'Cluster {c}', alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax2.scatter(kmeans_pca.cluster_centers_[:, 0], kmeans_pca.cluster_centers_[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidths=2, zorder=5)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax2.set_title(f'Clustering on First 2 PCs\nNMI = {nmi_pca:.4f}', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Ground truth
ax3 = axes[1, 0]
for party, color in [('Democrat', 'blue'), ('Republican', 'red')]:
    mask = parties == party
    ax3.scatter(votes_pca[mask, 0], votes_pca[mask, 1], c=color, label=party,
               alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax3.set_title('Actual Party Affiliations', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison bars
ax4 = axes[1, 1]
x = np.arange(len(metrics))
width = 0.35
vals_full = [m[1] for m in metrics]
vals_pca = [m[2] for m in metrics]

bars1 = ax4.bar(x - width/2, vals_full, width, label='All 16 Votes', color='steelblue')
bars2 = ax4.bar(x + width/2, vals_pca, width, label='First 2 PCs', color='coral')

ax4.set_ylabel('Score')
ax4.set_title('Agreement Metrics Comparison', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([m[0] for m in metrics])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1])

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.suptitle('Part C: Clustering Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('part_c_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved: part_c_comparison.png")
