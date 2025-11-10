import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chi2
from scipy.special import rel_entr

# Load the data
data = pd.read_csv('p1a.csv', header=None)
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

print("Data loaded successfully!")
print(f"Shape: {data.shape}")
print(f"X distribution: 0s={np.sum(X==0)}, 1s={np.sum(X==1)}")
print(f"Y distribution: 0s={np.sum(Y==0)}, 1s={np.sum(Y==1)}")

# =============================================================================
# 1. MUTUAL INFORMATION
# =============================================================================

def compute_mutual_information(x, y):
    """
    Compute mutual information between two binary variables.
    MI(X;Y) = sum_x sum_y P(x,y) log(P(x,y) / (P(x)P(y)))
    """
    # Create contingency table
    contingency = np.zeros((2, 2))
    for i in range(len(x)):
        contingency[int(x[i]), int(y[i])] += 1

    # Normalize to get probabilities
    n = len(x)
    joint_prob = contingency / n

    # Marginal probabilities
    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)

    # Compute mutual information
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (px[i] * py[j]))

    return mi

# Observed MI
observed_mi = compute_mutual_information(X, Y)
print(f"\n{'='*60}")
print("MUTUAL INFORMATION")
print(f"{'='*60}")
print(f"Observed MI: {observed_mi:.6f}")

# Permutation test for MI
N_PERMUTATIONS = 10000
np.random.seed(42)

mi_null_distribution = []
for i in range(N_PERMUTATIONS):
    # Permute Y to break association
    Y_perm = np.random.permutation(Y)
    mi_perm = compute_mutual_information(X, Y_perm)
    mi_null_distribution.append(mi_perm)

mi_null_distribution = np.array(mi_null_distribution)

# Compute p-value
count_extreme = np.sum(mi_null_distribution >= observed_mi)
p_value_mi = (count_extreme + 1) / (N_PERMUTATIONS + 1)

print(f"Number of permutations (N): {N_PERMUTATIONS}")
print(f"Count of permutations >= observed: {count_extreme}")
print(f"P-value: {p_value_mi:.6f}")

# =============================================================================
# 2. JACCARD INDEX
# =============================================================================

def compute_jaccard_index(x, y):
    """
    Compute Jaccard Index for binary variables.
    JI = |X=1 AND Y=1| / |X=1 OR Y=1|
    """
    intersection = np.sum((x == 1) & (y == 1))
    union = np.sum((x == 1) | (y == 1))

    if union == 0:
        return 0.0

    return intersection / union

# Observed Jaccard Index
observed_ji = compute_jaccard_index(X, Y)
print(f"\n{'='*60}")
print("JACCARD INDEX")
print(f"{'='*60}")
print(f"Observed JI: {observed_ji:.6f}")

# Permutation test for JI
ji_null_distribution = []
for i in range(N_PERMUTATIONS):
    Y_perm = np.random.permutation(Y)
    ji_perm = compute_jaccard_index(X, Y_perm)
    ji_null_distribution.append(ji_perm)

ji_null_distribution = np.array(ji_null_distribution)

# Compute p-value
count_extreme_ji = np.sum(ji_null_distribution >= observed_ji)
p_value_ji = (count_extreme_ji + 1) / (N_PERMUTATIONS + 1)

print(f"Number of permutations (N): {N_PERMUTATIONS}")
print(f"Count of permutations >= observed: {count_extreme_ji}")
print(f"P-value: {p_value_ji:.6f}")

# =============================================================================
# 3. PEARSON'S CHI-SQUARED TEST
# =============================================================================

# Create contingency table
contingency_table = pd.crosstab(X, Y)
print(f"\n{'='*60}")
print("PEARSON'S CHI-SQUARED TEST")
print(f"{'='*60}")
print("Contingency Table:")
print(contingency_table)

# Perform chi-squared test
chi2_stat, p_value_chi2, dof, expected = chi2_contingency(contingency_table)

print(f"\nObserved χ² value: {chi2_stat:.6f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value_chi2:.6e}")
print(f"Expected frequencies:\n{expected}")

# =============================================================================
# SUMMARY AND INTERPRETATION
# =============================================================================

alpha = 0.05

print(f"\n{'='*60}")
print("SUMMARY OF RESULTS")
print(f"{'='*60}")
print(f"Significance level (α): {alpha}")
print()
print("Statistic            | Value      | P-value    | Significant?")
print("-" * 60)
print(f"Mutual Information   | {observed_mi:.6f}   | {p_value_mi:.6f}   | {'Yes' if p_value_mi < alpha else 'No'}")
print(f"Jaccard Index        | {observed_ji:.6f}   | {p_value_ji:.6f}   | {'Yes' if p_value_ji < alpha else 'No'}")
print(f"Chi-squared (χ²)     | {chi2_stat:.6f}   | {p_value_chi2:.6e} | {'Yes' if p_value_chi2 < alpha else 'No'}")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: MI permutation test
axes[0].hist(mi_null_distribution, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
axes[0].axvline(observed_mi, color='red', linestyle='--', linewidth=2, label=f'Observed MI = {observed_mi:.4f}')
axes[0].set_xlabel('Mutual Information', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Mutual Information Permutation Test', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: JI permutation test
axes[1].hist(ji_null_distribution, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1].axvline(observed_ji, color='red', linestyle='--', linewidth=2, label=f'Observed JI = {observed_ji:.4f}')
axes[1].set_xlabel('Jaccard Index', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Jaccard Index Permutation Test', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Chi-squared distribution
x_chi2 = np.linspace(0, 15, 1000)
y_chi2 = chi2.pdf(x_chi2, df=dof)
axes[2].plot(x_chi2, y_chi2, 'b-', linewidth=2, label=f'χ² distribution (df={dof})')
axes[2].axvline(chi2_stat, color='red', linestyle='--', linewidth=2, label=f'Observed χ² = {chi2_stat:.4f}')
axes[2].fill_between(x_chi2[x_chi2 >= chi2_stat], 0, y_chi2[x_chi2 >= chi2_stat],
                      color='red', alpha=0.3, label=f'p-value = {p_value_chi2:.2e}')
axes[2].set_xlabel('χ² value', fontsize=12)
axes[2].set_ylabel('Probability Density', fontsize=12)
axes[2].set_title("Pearson's Chi-squared Test", fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1a_permutation.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as 'task1a_permutation.png'")
plt.close()

# =============================================================================
# DETAILED INTERPRETATION
# =============================================================================

print(f"\n{'='*60}")
print("DETAILED INTERPRETATION")
print(f"{'='*60}")

print("\n1. STATISTICAL SIGNIFICANCE:")
if p_value_mi < alpha and p_value_ji < alpha and p_value_chi2 < alpha:
    print("   All three statistics indicate a statistically significant association")
    print("   between the two genomic variants at α = 0.05 level.")
elif p_value_mi >= alpha and p_value_ji >= alpha and p_value_chi2 >= alpha:
    print("   None of the statistics show a statistically significant association")
    print("   between the two genomic variants at α = 0.05 level.")
else:
    print("   The statistics show conflicting results regarding statistical significance.")

print("\n2. STRENGTH OF ASSOCIATION:")
print(f"   - Mutual Information: {observed_mi:.6f}")
print(f"     MI ranges from 0 (no association) to log2(min(|X|,|Y|)) = 1 for binary.")
if observed_mi < 0.05:
    print("     This indicates very weak or no association.")
elif observed_mi < 0.15:
    print("     This indicates weak association.")
elif observed_mi < 0.30:
    print("     This indicates moderate association.")
else:
    print("     This indicates strong association.")

print(f"\n   - Jaccard Index: {observed_ji:.6f}")
print(f"     JI ranges from 0 (no overlap) to 1 (complete overlap).")
if observed_ji < 0.2:
    print("     This indicates very low overlap/similarity.")
elif observed_ji < 0.4:
    print("     This indicates low to moderate overlap.")
elif observed_ji < 0.6:
    print("     This indicates moderate overlap.")
else:
    print("     This indicates strong overlap.")

print(f"\n   - Chi-squared: {chi2_stat:.6f}")
print(f"     The magnitude itself depends on sample size.")
print(f"     The p-value ({p_value_chi2:.2e}) provides better interpretation.")

print("\n3. AGREEMENT BETWEEN STATISTICS:")
significant_count = sum([p_value_mi < alpha, p_value_ji < alpha, p_value_chi2 < alpha])
print(f"   {significant_count} out of 3 statistics show significance at α = {alpha}.")

if significant_count == 3:
    print("   All statistics agree: there IS a statistically significant association.")
elif significant_count == 0:
    print("   All statistics agree: there is NO statistically significant association.")
else:
    print("   The statistics do NOT fully agree on statistical significance.")

print("\n4. EXPLANATION OF DISCREPANCIES (if any):")
if significant_count == 1 or significant_count == 2:
    print("   Possible reasons for disagreement:")
    print("   - Different sensitivity to types of association:")
    print("     * MI captures general dependence (information-theoretic)")
    print("     * JI focuses on co-occurrence of 1s (set overlap)")
    print("     * χ² tests independence based on contingency table")
    print("   - Sample size effects may differ across statistics")
    print("   - JI can be more sensitive to sparse positive cases")
else:
    print("   The statistics show strong agreement, suggesting a consistent")
    print("   pattern of association (or lack thereof) in the data.")

print(f"\n{'='*60}")
print("Analysis complete!")
print(f"{'='*60}")
