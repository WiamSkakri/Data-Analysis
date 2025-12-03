"""
Task 2 Part A: Wine Quality Classification - Label Design and Visualization
Author: Wiam Skakri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

print("="*80)
print("TASK 2 PART A: LABEL DESIGN AND DISCRETIZATION")
print("="*80)

# Load datasets
print("\n[1] Loading datasets...")
red_wine = pd.read_csv('datasets/wine+quality/winequality-red.csv', sep=';')
white_wine = pd.read_csv('datasets/wine+quality/winequality-white.csv', sep=';')

print(f"   ✓ Red wine: {red_wine.shape[0]} samples")
print(f"   ✓ White wine: {white_wine.shape[0]} samples")

# Define discretization threshold
QUALITY_THRESHOLD = 5  # Quality > 5 = Good, ≤ 5 = Bad

print(f"\n[2] Discretization Strategy: Quality > {QUALITY_THRESHOLD} = 'Good', ≤ {QUALITY_THRESHOLD} = 'Bad'")

# Apply discretization consistently to both datasets
red_wine['quality_label'] = (red_wine['quality'] > QUALITY_THRESHOLD).astype(int)
white_wine['quality_label'] = (white_wine['quality'] > QUALITY_THRESHOLD).astype(int)

# Map to readable labels
label_map = {0: 'Bad', 1: 'Good'}
red_wine['quality_class'] = red_wine['quality_label'].map(label_map)
white_wine['quality_class'] = white_wine['quality_label'].map(label_map)

# Statistics
red_good_count = (red_wine['quality_label'] == 1).sum()
red_bad_count = (red_wine['quality_label'] == 0).sum()
white_good_count = (white_wine['quality_label'] == 1).sum()
white_bad_count = (white_wine['quality_label'] == 0).sum()

print(f"\n   Red Wine:   Good = {red_good_count} ({red_good_count/len(red_wine)*100:.1f}%), "
      f"Bad = {red_bad_count} ({red_bad_count/len(red_wine)*100:.1f}%)")
print(f"   White Wine: Good = {white_good_count} ({white_good_count/len(white_wine)*100:.1f}%), "
      f"Bad = {white_bad_count} ({white_bad_count/len(white_wine)*100:.1f}%)")

# ============================================================================
# VISUALIZATION 1: Original Quality Score Distributions
# ============================================================================
print("\n[3] Creating visualizations...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Row 1: Original quality distributions
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# Red wine original distribution
red_quality_counts = red_wine['quality'].value_counts().sort_index()
ax1.bar(red_quality_counts.index, red_quality_counts.values,
        color='darkred', alpha=0.7, edgecolor='black', width=0.7)
ax1.axvline(x=QUALITY_THRESHOLD + 0.5, color='red', linestyle='--', linewidth=2,
            label=f'Threshold = {QUALITY_THRESHOLD}')
ax1.set_xlabel('Quality Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title(f'Red Wine: Original Quality Distribution\n(n={len(red_wine)}, mean={red_wine["quality"].mean():.2f})',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(range(int(red_wine['quality'].min()), int(red_wine['quality'].max())+1))

# White wine original distribution
white_quality_counts = white_wine['quality'].value_counts().sort_index()
ax2.bar(white_quality_counts.index, white_quality_counts.values,
        color='gold', alpha=0.7, edgecolor='black', width=0.7)
ax2.axvline(x=QUALITY_THRESHOLD + 0.5, color='red', linestyle='--', linewidth=2,
            label=f'Threshold = {QUALITY_THRESHOLD}')
ax2.set_xlabel('Quality Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title(f'White Wine: Original Quality Distribution\n(n={len(white_wine)}, mean={white_wine["quality"].mean():.2f})',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(range(int(white_wine['quality'].min()), int(white_wine['quality'].max())+1))

# Combined comparison
bins = np.arange(2.5, 10.5, 1)
ax3.hist(red_wine['quality'], bins=bins, color='darkred', alpha=0.6,
         label='Red Wine', edgecolor='black')
ax3.hist(white_wine['quality'], bins=bins, color='gold', alpha=0.6,
         label='White Wine', edgecolor='black')
ax3.axvline(x=QUALITY_THRESHOLD + 0.5, color='red', linestyle='--', linewidth=2,
            label=f'Threshold = {QUALITY_THRESHOLD}')
ax3.set_xlabel('Quality Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Combined Quality Distribution Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Row 2: Discretized class distributions
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# Red wine discretized
red_class_counts = red_wine['quality_class'].value_counts()
colors_red = ['#8B0000', '#90EE90']  # dark red for bad, light green for good
bars1 = ax4.bar(red_class_counts.index, red_class_counts.values,
                color=[colors_red[1] if x == 'Good' else colors_red[0] for x in red_class_counts.index],
                alpha=0.8, edgecolor='black', width=0.5)
ax4.set_xlabel('Quality Class', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title(f'Red Wine: After Discretization\n(Bad={red_bad_count}, Good={red_good_count})',
              fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
# Add percentage labels on bars
for i, (idx, val) in enumerate(red_class_counts.items()):
    percentage = val / len(red_wine) * 100
    ax4.text(i, val, f'{val}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

# White wine discretized
white_class_counts = white_wine['quality_class'].value_counts()
colors_white = ['#DAA520', '#90EE90']  # gold for bad, light green for good
bars2 = ax5.bar(white_class_counts.index, white_class_counts.values,
                color=[colors_white[1] if x == 'Good' else colors_white[0] for x in white_class_counts.index],
                alpha=0.8, edgecolor='black', width=0.5)
ax5.set_xlabel('Quality Class', fontsize=12, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax5.set_title(f'White Wine: After Discretization\n(Bad={white_bad_count}, Good={white_good_count})',
              fontsize=13, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
# Add percentage labels on bars
for i, (idx, val) in enumerate(white_class_counts.items()):
    percentage = val / len(white_wine) * 100
    ax5.text(i, val, f'{val}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

# Comparison of class proportions
wine_types = ['Red Wine', 'White Wine']
bad_counts = [red_bad_count, white_bad_count]
good_counts = [red_good_count, white_good_count]

x = np.arange(len(wine_types))
width = 0.35

bars_bad = ax6.bar(x - width/2, bad_counts, width, label='Bad (≤5)',
                   color='#CD5C5C', alpha=0.8, edgecolor='black')
bars_good = ax6.bar(x + width/2, good_counts, width, label='Good (>5)',
                    color='#90EE90', alpha=0.8, edgecolor='black')

ax6.set_xlabel('Wine Type', fontsize=12, fontweight='bold')
ax6.set_ylabel('Count', fontsize=12, fontweight='bold')
ax6.set_title('Class Distribution Comparison', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(wine_types)
ax6.legend(fontsize=11)
ax6.grid(axis='y', alpha=0.3)

# Add count labels on bars
for bar in bars_bad:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')
for bar in bars_good:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Main title
fig.suptitle('Task 2 Part A: Wine Quality Label Design and Discretization',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('task2_part_a_label_design.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: task2_part_a_label_design.png")
plt.close()

# ============================================================================
# Save processed datasets for Part B
# ============================================================================
print("\n[4] Saving processed datasets...")

# Save complete datasets with labels
red_wine.to_csv('datasets/wine+quality/red_wine_labeled.csv', index=False)
white_wine.to_csv('datasets/wine+quality/white_wine_labeled.csv', index=False)

print("   ✓ Saved: red_wine_labeled.csv")
print("   ✓ Saved: white_wine_labeled.csv")

# ============================================================================
# Print Rationale (for LaTeX document)
# ============================================================================
print("\n" + "="*80)
print("RATIONALE FOR LABEL DESIGN (≤150 words)")
print("="*80)

rationale = """
I discretized wine quality using a threshold of 5 (quality > 5 = Good, ≤ 5 = Bad)
for the following reasons:

1. INTERPRETABILITY: This threshold separates above-average wines (quality 6-9)
   from average and below-average wines (quality 3-5), which aligns with the
   intuitive notion of "good" wine based on the 0-10 quality scale.

2. CLASS BALANCE: This approach yields relatively balanced classes - Red wine:
   53.5% good, 46.5% bad; White wine: 66.5% good, 33.5% bad. This balance helps
   prevent models from being biased toward the majority class.

3. CONSISTENT APPLICATION: The threshold is applied identically to both red and
   white wine datasets, ensuring models trained on one type can be meaningfully
   evaluated on the other during cross-domain testing.

4. SUFFICIENT SAMPLES: Both classes have adequate sample sizes in each dataset
   for reliable model training and evaluation.
"""

print(rationale)

print("\n" + "="*80)
print("PART A COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nNext steps:")
print("  - Add visualizations to LaTeX document")
print("  - Include rationale in write-up")
print("  - Proceed to Part B: Model Training & Evaluation")
