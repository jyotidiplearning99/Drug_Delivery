#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Setup figure
fig, ax = plt.subplots(figsize=(16, 8)) # Increased height slightly for better spacing

# Data setup
grouped_methods = ['Ridge', 'Lasso', 'SVM', 'Gradient\nBoosting']
grouped_ai_dadds_corr = [0.749, 0.645, 0.634, 0.706]
grouped_ai_dadds_r2 = [c**2 for c in grouped_ai_dadds_corr]

grouped_reported_corr = [0.514, 0.040, 0.409, 0.308]
grouped_reported_r2 = [-1.036, -0.010, 0.067, 0.090]
grouped_reported_r2_display = [max(0, r2) for r2 in grouped_reported_r2]

ensemble_ai_dadds_corr = 0.811
ensemble_ai_dadds_r2 = ensemble_ai_dadds_corr**2

agile_reported_corr = 0.573
agile_reported_r2 = 0.249

# Bar positions
x_grouped = np.arange(len(grouped_methods))
width = 0.35
x_ensemble = len(grouped_methods) 
x_agile = len(grouped_methods) + 1 
single_width = 0.50

# --- PLOTTING ---

# Grouped bars
bars1 = ax.bar(x_grouped - width/2, grouped_ai_dadds_r2, width, 
               label='AI-DADDS', color='#2ecc71', edgecolor='black', linewidth=1.2, alpha=0.85)

bars2 = ax.bar(x_grouped + width/2, grouped_reported_r2_display, width, 
               label='Reported²', color='#e74c3c', edgecolor='black', linewidth=1.2, alpha=0.85)

# Separate bars
bar_ensemble = ax.bar(x_ensemble, ensemble_ai_dadds_r2, single_width,
                       color='#2ecc71', edgecolor='black', linewidth=1.2, alpha=0.85)

bar_agile = ax.bar(x_agile, agile_reported_r2, single_width,
                   color='#e74c3c', edgecolor='black', linewidth=1.2, alpha=0.85)

# --- TEXT LABELS (Updated offset to 'put down a bit') ---
# Using 0.01 offset instead of 0.03
label_offset = 0.01 

for bar, r2, corr in zip(bars1, grouped_ai_dadds_r2, grouped_ai_dadds_corr):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + label_offset,
            f'R²={r2:.3f}\nr={corr:.3f}', ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='black')

for bar, actual_r2, corr in zip(bars2, grouped_reported_r2, grouped_reported_corr):
    display_r2 = max(0, actual_r2)
    text_color = 'red' if actual_r2 < 0 else 'black'
    label = f'R²={actual_r2:.3f}*\nr={corr:.3f}' if actual_r2 < 0 else f'R²={display_r2:.3f}\nr={corr:.3f}'
    ax.text(bar.get_x() + bar.get_width()/2., display_r2 + label_offset,
            label, ha='center', va='bottom', fontsize=12, fontweight='bold', color=text_color)

# Ensemble/Agile labels
ax.text(bar_ensemble[0].get_x() + bar_ensemble[0].get_width()/2., 
        bar_ensemble[0].get_height() + label_offset,
        f'R²={ensemble_ai_dadds_r2:.3f}\nr={ensemble_ai_dadds_corr:.3f}',
        ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.text(bar_agile[0].get_x() + bar_agile[0].get_width()/2., 
        bar_agile[0].get_height() + label_offset,
        f'R²={agile_reported_r2:.3f}\nr={agile_reported_corr:.3f}',
        ha='center', va='bottom', fontsize=13, fontweight='bold')

# --- FORMATTING (BIG & BOLD) ---

# X-Axis Labels (Methods)
all_labels = grouped_methods + ['Ensemble', 'AGILE']
ax.set_xticks(list(x_grouped) + [x_ensemble, x_agile])
ax.set_xticklabels(all_labels, fontsize=18, fontweight='bold') # Enlarged and Bold

ax.set_ylabel('R² (Coef. of Determination)', fontsize=18, fontweight='bold')
ax.set_xlabel('Method', fontsize=18, fontweight='bold')
ax.set_title('AI-DADDS vs AGILE Comparison\nPerformance Comparison', fontsize=22, fontweight='bold', pad=20)

# Axis limits and grid
ax.set_ylim([0, 0.95]) # Increased to ensure top labels aren't cut
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Legend (Bigger and bolder labels)
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='AI-DADDS'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Reported²')
]
ax.legend(handles=legend_elements, loc='upper center', 
          fontsize=18, framealpha=0.95, ncol=2, prop={'size': 18, 'weight': 'bold'})

# Note
ax.text(0.02, 0.97, '*Negative R² displayed as 0', transform=ax.transAxes, 
        fontsize=12, va='top', style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('AI_DADDS_vs_Nature_Updated.png', dpi=300, bbox_inches='tight')
plt.show()