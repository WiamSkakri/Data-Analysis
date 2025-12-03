"""
Task 2 Part C: Analysis and Interpretation with Visualizations
Author: Wiam Skakri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set matplotlib backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

print("="*80)
print("TASK 2 PART C: ANALYSIS AND INTERPRETATION")
print("="*80)

# Load results from Part B
print("\n[1] Loading Part B results...")
with open('task2_part_b_full_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Convert to DataFrame for easier manipulation
results_df = pd.DataFrame([{
    'Model': r['model'],
    'Train On': r['train_on'],
    'Test On': r['test_on'],
    'Accuracy': r['accuracy'],
    'Precision': r['precision'],
    'Recall': r['recall']
} for r in results])

print("   ✓ Loaded results for analysis")

# ============================================================================
# VISUALIZATION 1: Performance Comparison Across All Conditions
# ============================================================================
print("\n[2] Creating performance comparison visualization...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Extract data
lr_results = results_df[results_df['Model'] == 'Logistic Regression']
nn_results = results_df[results_df['Model'] == 'Neural Network']

scenarios = ['Red→Red', 'White→White', 'Red→White', 'White→Red']
lr_acc = [
    lr_results[(lr_results['Train On']=='Red') & (lr_results['Test On']=='Red')]['Accuracy'].values[0],
    lr_results[(lr_results['Train On']=='White') & (lr_results['Test On']=='White')]['Accuracy'].values[0],
    lr_results[(lr_results['Train On']=='Red') & (lr_results['Test On']=='White')]['Accuracy'].values[0],
    lr_results[(lr_results['Train On']=='White') & (lr_results['Test On']=='Red')]['Accuracy'].values[0]
]

nn_acc = [
    nn_results[(nn_results['Train On']=='Red') & (nn_results['Test On']=='Red')]['Accuracy'].values[0],
    nn_results[(nn_results['Train On']=='White') & (nn_results['Test On']=='White')]['Accuracy'].values[0],
    nn_results[(nn_results['Train On']=='Red') & (nn_results['Test On']=='White')]['Accuracy'].values[0],
    nn_results[(nn_results['Train On']=='White') & (nn_results['Test On']=='Red')]['Accuracy'].values[0]
]

# Plot 1: Accuracy comparison across all scenarios
ax1 = fig.add_subplot(gs[0, :])
x = np.arange(len(scenarios))
width = 0.35

bars1 = ax1.bar(x - width/2, lr_acc, width, label='Logistic Regression',
                color='#4C72B0', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, nn_acc, width, label='Neural Network',
                color='#DD8452', alpha=0.8, edgecolor='black')

ax1.set_xlabel('Evaluation Scenario', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax1.set_title('Model Performance Comparison Across All Scenarios', fontsize=15, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add dividing line between in-domain and cross-domain
ax1.axvline(x=1.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax1.text(0.75, 0.95, 'In-Domain', ha='center', va='top', fontsize=12,
         fontweight='bold', color='green', transform=ax1.transAxes)
ax1.text(0.25, 0.95, '←', ha='center', va='top', fontsize=16,
         color='green', transform=ax1.transAxes)
ax1.text(0.75, 0.05, 'Cross-Domain →', ha='center', va='bottom', fontsize=12,
         fontweight='bold', color='red', transform=ax1.transAxes)

# ============================================================================
# VISUALIZATION 2: Performance Degradation Analysis
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

# Calculate degradation for each model
lr_red_indomain = lr_acc[0]
lr_white_indomain = lr_acc[1]
lr_red_to_white = lr_acc[2]
lr_white_to_red = lr_acc[3]

nn_red_indomain = nn_acc[0]
nn_white_indomain = nn_acc[1]
nn_red_to_white = nn_acc[2]
nn_white_to_red = nn_acc[3]

lr_red_deg = (lr_red_indomain - lr_red_to_white) * 100
lr_white_deg = (lr_white_indomain - lr_white_to_red) * 100
nn_red_deg = (nn_red_indomain - nn_red_to_white) * 100
nn_white_deg = (nn_white_indomain - nn_white_to_red) * 100

degradation_scenarios = ['Red→White\nDegradation', 'White→Red\nDegradation']
lr_degs = [lr_red_deg, lr_white_deg]
nn_degs = [nn_red_deg, nn_white_deg]

x2 = np.arange(len(degradation_scenarios))
bars3 = ax2.bar(x2 - width/2, lr_degs, width, label='Logistic Regression',
                color='#4C72B0', alpha=0.8, edgecolor='black')
bars4 = ax2.bar(x2 + width/2, nn_degs, width, label='Neural Network',
                color='#DD8452', alpha=0.8, edgecolor='black')

ax2.set_xlabel('Cross-Domain Scenario', fontsize=12, fontweight='bold')
ax2.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cross-Domain Performance Degradation', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(degradation_scenarios, fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# VISUALIZATION 3: Precision-Recall Trade-off
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

# Extract precision and recall for all scenarios
lr_prec = [
    lr_results[(lr_results['Train On']=='Red') & (lr_results['Test On']=='Red')]['Precision'].values[0],
    lr_results[(lr_results['Train On']=='White') & (lr_results['Test On']=='White')]['Precision'].values[0],
    lr_results[(lr_results['Train On']=='Red') & (lr_results['Test On']=='White')]['Precision'].values[0],
    lr_results[(lr_results['Train On']=='White') & (lr_results['Test On']=='Red')]['Precision'].values[0]
]

lr_rec = [
    lr_results[(lr_results['Train On']=='Red') & (lr_results['Test On']=='Red')]['Recall'].values[0],
    lr_results[(lr_results['Train On']=='White') & (lr_results['Test On']=='White')]['Recall'].values[0],
    lr_results[(lr_results['Train On']=='Red') & (lr_results['Test On']=='White')]['Recall'].values[0],
    lr_results[(lr_results['Train On']=='White') & (lr_results['Test On']=='Red')]['Recall'].values[0]
]

nn_prec = [
    nn_results[(nn_results['Train On']=='Red') & (nn_results['Test On']=='Red')]['Precision'].values[0],
    nn_results[(nn_results['Train On']=='White') & (nn_results['Test On']=='White')]['Precision'].values[0],
    nn_results[(nn_results['Train On']=='Red') & (nn_results['Test On']=='White')]['Precision'].values[0],
    nn_results[(nn_results['Train On']=='White') & (nn_results['Test On']=='Red')]['Precision'].values[0]
]

nn_rec = [
    nn_results[(nn_results['Train On']=='Red') & (nn_results['Test On']=='Red')]['Recall'].values[0],
    nn_results[(nn_results['Train On']=='White') & (nn_results['Test On']=='White')]['Recall'].values[0],
    nn_results[(nn_results['Train On']=='Red') & (nn_results['Test On']=='White')]['Recall'].values[0],
    nn_results[(nn_results['Train On']=='White') & (nn_results['Test On']=='Red')]['Recall'].values[0]
]

# Scatter plot with labels
colors = ['green', 'green', 'red', 'red']
markers = ['o', 's', '^', 'D']
for i, scenario in enumerate(scenarios):
    ax3.scatter(lr_rec[i], lr_prec[i], s=200, marker=markers[i],
               color=colors[i], alpha=0.6, edgecolor='black', linewidth=2,
               label=f'LR: {scenario}')
    ax3.scatter(nn_rec[i], nn_prec[i], s=200, marker=markers[i],
               color=colors[i], alpha=0.3, edgecolor='black', linewidth=2,
               label=f'NN: {scenario}')

ax3.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax3.set_title('Precision-Recall Trade-off Analysis', fontsize=14, fontweight='bold')
ax3.legend(fontsize=8, loc='lower left', ncol=2)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# ============================================================================
# VISUALIZATION 4: Model Consistency (Variance) Analysis
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

# Calculate variance/consistency metrics
lr_indomain_mean = np.mean([lr_acc[0], lr_acc[1]])
lr_indomain_std = np.std([lr_acc[0], lr_acc[1]])
nn_indomain_mean = np.mean([nn_acc[0], nn_acc[1]])
nn_indomain_std = np.std([nn_acc[0], nn_acc[1]])

lr_crossdomain_mean = np.mean([lr_acc[2], lr_acc[3]])
lr_crossdomain_std = np.std([lr_acc[2], lr_acc[3]])
nn_crossdomain_mean = np.mean([nn_acc[2], nn_acc[3]])
nn_crossdomain_std = np.std([nn_acc[2], nn_acc[3]])

categories = ['In-Domain\nMean', 'Cross-Domain\nMean']
lr_means = [lr_indomain_mean, lr_crossdomain_mean]
lr_stds = [lr_indomain_std, lr_crossdomain_std]
nn_means = [nn_indomain_mean, nn_crossdomain_mean]
nn_stds = [nn_indomain_std, nn_crossdomain_std]

x3 = np.arange(len(categories))
bars5 = ax4.bar(x3 - width/2, lr_means, width, yerr=lr_stds,
                label='Logistic Regression', color='#4C72B0', alpha=0.8,
                edgecolor='black', capsize=5, error_kw={'linewidth': 2})
bars6 = ax4.bar(x3 + width/2, nn_means, width, yerr=nn_stds,
                label='Neural Network', color='#DD8452', alpha=0.8,
                edgecolor='black', capsize=5, error_kw={'linewidth': 2})

ax4.set_xlabel('Testing Condition', fontsize=12, fontweight='bold')
ax4.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Model Consistency: Mean Accuracy with Variance', fontsize=14, fontweight='bold')
ax4.set_xticks(x3)
ax4.set_xticklabels(categories, fontsize=11)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([0, 1])

# Add value labels
for i, (bars, means, stds) in enumerate([(bars5, lr_means, lr_stds), (bars6, nn_means, nn_stds)]):
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + stds[j],
                f'{height:.1%}\n±{stds[j]:.1%}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# ============================================================================
# VISUALIZATION 5: Metric Heatmap
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

# Create performance matrix for heatmap
performance_data = []
for scenario in scenarios:
    train, test = scenario.split('→')
    lr_row = lr_results[(lr_results['Train On']==train) & (lr_results['Test On']==test)]
    nn_row = nn_results[(nn_results['Train On']==train) & (nn_results['Test On']==test)]

    performance_data.append([
        lr_row['Accuracy'].values[0],
        lr_row['Precision'].values[0],
        lr_row['Recall'].values[0],
        nn_row['Accuracy'].values[0],
        nn_row['Precision'].values[0],
        nn_row['Recall'].values[0]
    ])

performance_matrix = np.array(performance_data).T

im = ax5.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax5.set_xticks(np.arange(len(scenarios)))
ax5.set_yticks(np.arange(6))
ax5.set_xticklabels(scenarios, fontsize=10)
ax5.set_yticklabels(['LR Acc', 'LR Prec', 'LR Rec', 'NN Acc', 'NN Prec', 'NN Rec'], fontsize=10)

# Add text annotations
for i in range(6):
    for j in range(len(scenarios)):
        text = ax5.text(j, i, f'{performance_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=9)

ax5.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('Performance Score', fontsize=11, fontweight='bold')

# Main title
fig.suptitle('Task 2 Part C: Model Performance Analysis and Comparison',
             fontsize=17, fontweight='bold', y=0.995)

plt.savefig('task2_part_c_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: task2_part_c_analysis.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n[3] Summary Statistics for Analysis:")
print("\n--- In-Domain Performance ---")
print(f"Logistic Regression: Mean={lr_indomain_mean:.1%}, Std={lr_indomain_std:.1%}")
print(f"Neural Network:      Mean={nn_indomain_mean:.1%}, Std={nn_indomain_std:.1%}")
print(f"More consistent: {'Logistic Regression' if lr_indomain_std < nn_indomain_std else 'Neural Network'}")

print("\n--- Cross-Domain Performance ---")
print(f"Logistic Regression: Mean={lr_crossdomain_mean:.1%}, Std={lr_crossdomain_std:.1%}")
print(f"Neural Network:      Mean={nn_crossdomain_mean:.1%}, Std={nn_crossdomain_std:.1%}")
print(f"Better generalization: Logistic Regression")

print("\n--- Performance Degradation ---")
print(f"LR Red→White:  {lr_red_deg:.1f}% drop")
print(f"LR White→Red:  {lr_white_deg:.1f}% drop")
print(f"NN Red→White:  {nn_red_deg:.1f}% drop")
print(f"NN White→Red:  {nn_white_deg:.1f}% drop")

print("\n" + "="*80)
print("PART C COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nVisualization saved: task2_part_c_analysis.png")
print("Ready to write analysis answers!")
