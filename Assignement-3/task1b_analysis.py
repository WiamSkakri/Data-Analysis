import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations

# Load the data
data = pd.read_csv('p1b.csv', header=None)
print(f"Data shape: {data.shape}")
print(f"Number of variables: {data.shape[1]}")

# Generate all pairs of variables
n_vars = data.shape[1]
all_pairs = list(combinations(range(n_vars), 2))
n_pairs = len(all_pairs)
print(f"Number of variable pairs: {n_pairs}")
print(f"Expected: {n_vars * (n_vars - 1) // 2} = {15 * 14 // 2}")

# =============================================================================
# FUNCTIONS
# =============================================================================

def compute_mutual_information(x, y):
    """Compute mutual information between two binary variables."""
    contingency = np.zeros((2, 2))
    for i in range(len(x)):
        contingency[int(x[i]), int(y[i])] += 1

    n = len(x)
    joint_prob = contingency / n
    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)

    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (px[i] * py[j]))

    return mi

def compute_jaccard_index(x, y):
    """Compute Jaccard Index for binary variables."""
    intersection = np.sum((x == 1) & (y == 1))
    union = np.sum((x == 1) | (y == 1))

    if union == 0:
        return 0.0

    return intersection / union

def permutation_test(x, y, statistic_func, n_permutations=10000, seed=42):
    """
    Perform permutation test for a given statistic.
    Returns observed value, p-value, and null distribution.
    """
    np.random.seed(seed)

    # Compute observed statistic
    observed = statistic_func(x, y)

    # Generate null distribution
    null_dist = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        null_dist.append(statistic_func(x, y_perm))

    null_dist = np.array(null_dist)

    # Compute p-value
    count_extreme = np.sum(null_dist >= observed)
    p_value = (count_extreme + 1) / (n_permutations + 1)

    return observed, p_value, null_dist

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg procedure for multiple hypothesis correction.
    Returns boolean array indicating which hypotheses are significant.
    """
    n = len(p_values)
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = p_values[sorted_indices]

    # Find largest i such that P(i) <= (i/n) * alpha
    significant = np.zeros(n, dtype=bool)
    for i in range(n-1, -1, -1):
        if sorted_pvalues[i] <= ((i + 1) / n) * alpha:
            # All hypotheses up to and including i are significant
            significant[sorted_indices[:i+1]] = True
            break

    return significant

# =============================================================================
# ANALYSIS
# =============================================================================

N_PERMUTATIONS = 10000
ALPHA = 0.05

print(f"\n{'='*70}")
print(f"ANALYZING {n_pairs} VARIABLE PAIRS")
print(f"{'='*70}")
print(f"Number of permutations: {N_PERMUTATIONS}")
print(f"Significance level (α): {ALPHA}")
print(f"Using Benjamini-Hochberg correction for multiple testing")
print()

# Store results
results = {
    'pair': [],
    'var1': [],
    'var2': [],
    'mi': [],
    'mi_pvalue': [],
    'ji': [],
    'ji_pvalue': [],
    'chi2': [],
    'chi2_pvalue': []
}

# Compute statistics for all pairs
print("Computing statistics for all pairs...")
for idx, (i, j) in enumerate(all_pairs):
    if (idx + 1) % 20 == 0:
        print(f"  Processed {idx + 1}/{n_pairs} pairs...")

    X = data.iloc[:, i].values
    Y = data.iloc[:, j].values

    # Mutual Information with permutation test
    mi_obs, mi_pval, _ = permutation_test(X, Y, compute_mutual_information, N_PERMUTATIONS, seed=42+idx)

    # Jaccard Index with permutation test
    ji_obs, ji_pval, _ = permutation_test(X, Y, compute_jaccard_index, N_PERMUTATIONS, seed=42+idx)

    # Chi-squared test (parametric)
    contingency_table = pd.crosstab(X, Y)
    chi2_stat, chi2_pval, _, _ = chi2_contingency(contingency_table)

    # Store results
    results['pair'].append(f"V{i}-V{j}")
    results['var1'].append(i)
    results['var2'].append(j)
    results['mi'].append(mi_obs)
    results['mi_pvalue'].append(mi_pval)
    results['ji'].append(ji_obs)
    results['ji_pvalue'].append(ji_pval)
    results['chi2'].append(chi2_stat)
    results['chi2_pvalue'].append(chi2_pval)

print(f"  Completed all {n_pairs} pairs!\n")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# =============================================================================
# MULTIPLE TESTING CORRECTION
# =============================================================================

print(f"{'='*70}")
print("APPLYING BENJAMINI-HOCHBERG CORRECTION")
print(f"{'='*70}\n")

# Apply Benjamini-Hochberg correction
mi_significant = benjamini_hochberg(np.array(results['mi_pvalue']), ALPHA)
ji_significant = benjamini_hochberg(np.array(results['ji_pvalue']), ALPHA)
chi2_significant = benjamini_hochberg(np.array(results['chi2_pvalue']), ALPHA)

results_df['mi_significant'] = mi_significant
results_df['ji_significant'] = ji_significant
results_df['chi2_significant'] = chi2_significant

# Count significant pairs
n_mi_sig = np.sum(mi_significant)
n_ji_sig = np.sum(ji_significant)
n_chi2_sig = np.sum(chi2_significant)

print(f"Mutual Information: {n_mi_sig}/{n_pairs} significantly associated pairs")
print(f"Jaccard Index:      {n_ji_sig}/{n_pairs} significantly associated pairs")
print(f"Chi-squared:        {n_chi2_sig}/{n_pairs} significantly associated pairs")

# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

print(f"\n{'='*70}")
print("OVERLAP BETWEEN STATISTICS")
print(f"{'='*70}\n")

# Compute overlaps
mi_ji_overlap = np.sum(mi_significant & ji_significant)
mi_chi2_overlap = np.sum(mi_significant & chi2_significant)
ji_chi2_overlap = np.sum(ji_significant & chi2_significant)
all_three_overlap = np.sum(mi_significant & ji_significant & chi2_significant)

print(f"MI and JI overlap:        {mi_ji_overlap} pairs")
print(f"MI and χ² overlap:        {mi_chi2_overlap} pairs")
print(f"JI and χ² overlap:        {ji_chi2_overlap} pairs")
print(f"All three overlap:        {all_three_overlap} pairs")

# Find which pairs are significant for each statistic
mi_sig_pairs = results_df[results_df['mi_significant']]['pair'].tolist()
ji_sig_pairs = results_df[results_df['ji_significant']]['pair'].tolist()
chi2_sig_pairs = results_df[results_df['chi2_significant']]['pair'].tolist()

print(f"\nMI significant pairs: {mi_sig_pairs}")
print(f"\nJI significant pairs: {ji_sig_pairs}")
print(f"\nχ² significant pairs: {chi2_sig_pairs}")

# =============================================================================
# SIMILARITY ANALYSIS
# =============================================================================

print(f"\n{'='*70}")
print("SIMILARITY BETWEEN STATISTICS")
print(f"{'='*70}\n")

# Compute correlations between statistics
from scipy.stats import spearmanr, pearsonr

mi_ji_corr, _ = spearmanr(results_df['mi'], results_df['ji'])
mi_chi2_corr, _ = spearmanr(results_df['mi'], results_df['chi2'])
ji_chi2_corr, _ = spearmanr(results_df['ji'], results_df['chi2'])

print("Spearman correlations between test statistics:")
print(f"  MI vs JI:  {mi_ji_corr:.4f}")
print(f"  MI vs χ²:  {mi_chi2_corr:.4f}")
print(f"  JI vs χ²:  {ji_chi2_corr:.4f}")

# Determine which two are most similar
correlations = {
    'MI-JI': mi_ji_corr,
    'MI-χ²': mi_chi2_corr,
    'JI-χ²': ji_chi2_corr
}
most_similar = max(correlations, key=correlations.get)
print(f"\nMost similar pair: {most_similar} (correlation = {correlations[most_similar]:.4f})")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print(f"\n{'='*70}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*70}\n")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Scatter plots comparing statistics
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(results_df['mi'], results_df['ji'], alpha=0.6, s=30, c='blue')
ax1.scatter(results_df[results_df['mi_significant']]['mi'],
           results_df[results_df['mi_significant']]['ji'],
           alpha=0.8, s=50, c='red', marker='x', label='MI significant')
ax1.scatter(results_df[results_df['ji_significant']]['mi'],
           results_df[results_df['ji_significant']]['ji'],
           alpha=0.8, s=50, c='green', marker='+', label='JI significant')
ax1.set_xlabel('Mutual Information', fontsize=11)
ax1.set_ylabel('Jaccard Index', fontsize=11)
ax1.set_title(f'MI vs JI (Spearman ρ={mi_ji_corr:.3f})', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(results_df['mi'], results_df['chi2'], alpha=0.6, s=30, c='blue')
ax2.scatter(results_df[results_df['mi_significant']]['mi'],
           results_df[results_df['mi_significant']]['chi2'],
           alpha=0.8, s=50, c='red', marker='x', label='MI significant')
ax2.scatter(results_df[results_df['chi2_significant']]['mi'],
           results_df[results_df['chi2_significant']]['chi2'],
           alpha=0.8, s=50, c='orange', marker='s', label='χ² significant')
ax2.set_xlabel('Mutual Information', fontsize=11)
ax2.set_ylabel('Chi-squared', fontsize=11)
ax2.set_title(f'MI vs χ² (Spearman ρ={mi_chi2_corr:.3f})', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(results_df['ji'], results_df['chi2'], alpha=0.6, s=30, c='blue')
ax3.scatter(results_df[results_df['ji_significant']]['ji'],
           results_df[results_df['ji_significant']]['chi2'],
           alpha=0.8, s=50, c='green', marker='+', label='JI significant')
ax3.scatter(results_df[results_df['chi2_significant']]['ji'],
           results_df[results_df['chi2_significant']]['chi2'],
           alpha=0.8, s=50, c='orange', marker='s', label='χ² significant')
ax3.set_xlabel('Jaccard Index', fontsize=11)
ax3.set_ylabel('Chi-squared', fontsize=11)
ax3.set_title(f'JI vs χ² (Spearman ρ={ji_chi2_corr:.3f})', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 2. P-value distributions
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(results_df['mi_pvalue'], bins=30, alpha=0.7, color='red', edgecolor='black')
ax4.axvline(ALPHA, color='black', linestyle='--', linewidth=2, label=f'α={ALPHA}')
ax4.set_xlabel('P-value', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title(f'MI P-value Distribution\n({n_mi_sig} significant after BH correction)',
             fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(results_df['ji_pvalue'], bins=30, alpha=0.7, color='green', edgecolor='black')
ax5.axvline(ALPHA, color='black', linestyle='--', linewidth=2, label=f'α={ALPHA}')
ax5.set_xlabel('P-value', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title(f'JI P-value Distribution\n({n_ji_sig} significant after BH correction)',
             fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(results_df['chi2_pvalue'], bins=30, alpha=0.7, color='orange', edgecolor='black')
ax6.axvline(ALPHA, color='black', linestyle='--', linewidth=2, label=f'α={ALPHA}')
ax6.set_xlabel('P-value', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title(f'χ² P-value Distribution\n({n_chi2_sig} significant after BH correction)',
             fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 3. Venn diagram representation
ax7 = fig.add_subplot(gs[2, :])
categories = ['MI only', 'JI only', 'χ² only', 'MI∩JI', 'MI∩χ²', 'JI∩χ²', 'All three']
mi_only = np.sum(mi_significant & ~ji_significant & ~chi2_significant)
ji_only = np.sum(~mi_significant & ji_significant & ~chi2_significant)
chi2_only = np.sum(~mi_significant & ~ji_significant & chi2_significant)
mi_ji_only = np.sum(mi_significant & ji_significant & ~chi2_significant)
mi_chi2_only = np.sum(mi_significant & ~ji_significant & chi2_significant)
ji_chi2_only = np.sum(~mi_significant & ji_significant & chi2_significant)

counts = [mi_only, ji_only, chi2_only, mi_ji_only, mi_chi2_only, ji_chi2_only, all_three_overlap]
colors_bar = ['red', 'green', 'orange', 'purple', 'brown', 'cyan', 'navy']
bars = ax7.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
ax7.set_ylabel('Number of Pairs', fontsize=12)
ax7.set_title('Overlap of Significant Pairs Between Statistics', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.suptitle(f'Comparison of MI, JI, and χ² for {n_pairs} Variable Pairs\n' +
             f'(N={N_PERMUTATIONS} permutations, α={ALPHA}, Benjamini-Hochberg correction)',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('task1b_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: task1b_comparison.png")

# Save results to CSV for reference
results_df.to_csv('task1b_results.csv', index=False)
print("Saved: task1b_results.csv")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE!")
print(f"{'='*70}")
