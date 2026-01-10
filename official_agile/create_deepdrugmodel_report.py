#!/usr/bin/env python
"""
AI-DADDS Grouped Comparison - Ensemble and AGILE Separated
Grouped bars for Ridge/Lasso/SVM/GB, then separate Ensemble (green) and AGILE (red)
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(16, 7))

# First 4 methods for grouped comparison
grouped_methods = ['Ridge', 'Lasso', 'SVM', 'Gradient\nBoosting']

# AI-DADDS results for grouped methods
grouped_ai_dadds_corr = [0.749, 0.645, 0.634, 0.706]
grouped_ai_dadds_r2 = [c**2 for c in grouped_ai_dadds_corr]

# Nature Paper results for grouped methods
grouped_reported_corr = [0.514, 0.040, 0.409, 0.308]
grouped_reported_r2 = [-1.036, -0.010, 0.067, 0.090]
grouped_reported_r2_display = [max(0, r2) for r2 in grouped_reported_r2]

# Ensemble (AI-DADDS) - separate green bar
ensemble_ai_dadds_corr = 0.811
ensemble_ai_dadds_r2 = ensemble_ai_dadds_corr**2

# AGILE (Reported) - separate red bar
agile_reported_corr = 0.573
agile_reported_r2 = 0.249

# Set up grouped bars for first 4 methods
x_grouped = np.arange(len(grouped_methods))
width = 0.35

# Create grouped bars
bars1 = ax.bar(x_grouped - width/2, grouped_ai_dadds_r2, width, 
               label='AI-DADDS', 
               color='#2ecc71', edgecolor='black', 
               linewidth=1.2, alpha=0.85)

bars2 = ax.bar(x_grouped + width/2, grouped_reported_r2_display, width, 
               label='Reported²',
               color='#e74c3c', edgecolor='black', 
               linewidth=1.2, alpha=0.85)

# Add value labels for grouped methods - 2 DECIMAL PLACES
for bar, r2, corr in zip(bars1, grouped_ai_dadds_r2, grouped_ai_dadds_corr):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
            f'R²={r2:.2f}\nr={corr:.2f}',  # Changed from .3f to .2f
            ha='center', va='bottom', fontsize=16, fontweight='bold',
            color='black')

# Add value labels for red bars - 2 DECIMAL PLACES
for bar, r2_display, corr, actual_r2 in zip(bars2, grouped_reported_r2_display, grouped_reported_corr, grouped_reported_r2):
    height = bar.get_height()
    text_color = 'red' if actual_r2 < 0 else 'black'
    display_text = f'R²={actual_r2:.2f}*\nr={corr:.2f}' if actual_r2 < 0 else f'R²={r2_display:.2f}\nr={corr:.2f}'  # Changed from .3f to .2f
    
    # Use minimum height of 0.08 for text placement to avoid overlap
    text_y_position = max(height + 0.015, 0.08)
    
    # SHIFT TEXT SLIGHTLY RIGHT by 0.05 units (few mm)
    text_x_position = bar.get_x() + bar.get_width()/2. + 0.06
    
    ax.text(text_x_position, text_y_position,
            display_text,
            ha='center', va='bottom', fontsize=16, fontweight='bold',
            color=text_color)

# Add Ensemble and AGILE as separate individual bars (no grouping)
x_ensemble = len(grouped_methods)  # No spacing - right after Gradient Boosting
x_agile = len(grouped_methods) + 1  # Next position

# Single bar width
single_width = 0.50

# Ensemble bar (green - AI-DADDS)
bar_ensemble = ax.bar(x_ensemble, ensemble_ai_dadds_r2, single_width,
                       color='#2ecc71', edgecolor='black', 
                       linewidth=1.2, alpha=0.85)

# AGILE bar (red - Reported)
bar_agile = ax.bar(x_agile, agile_reported_r2, single_width,
                   color='#e74c3c', edgecolor='black', 
                   linewidth=1.2, alpha=0.85)

# Add labels for Ensemble bar - 2 DECIMAL PLACES
ax.text(bar_ensemble[0].get_x() + bar_ensemble[0].get_width()/2., 
        bar_ensemble[0].get_height() + 0.015,
        f'R²={ensemble_ai_dadds_r2:.2f}\nr={ensemble_ai_dadds_corr:.2f}',  # Changed from .3f to .2f
        ha='center', va='bottom', fontsize=17, fontweight='bold',
        color='black')

# Add labels for AGILE bar - 2 DECIMAL PLACES
ax.text(bar_agile[0].get_x() + bar_agile[0].get_width()/2., 
        bar_agile[0].get_height() + 0.015,
        f'R²={agile_reported_r2:.2f}\nr={agile_reported_corr:.2f}',  # Changed from .3f to .2f
        ha='center', va='bottom', fontsize=17, fontweight='bold',
        color='black')

# Formatting
# Y-AXIS LABEL (SIGNIFICANTLY INCREASED FONTSIZE)
ax.set_ylabel('R² (Coe. of Determination)', 
              fontsize=20, fontweight='bold')

# X-AXIS LABEL REMOVED (as requested)
# ax.set_xlabel('Method', fontsize=12, fontweight='bold')  # COMMENTED OUT

#ax.set_title('AI-DADDS vs AGILE Comparison\n' +
#             'Molecular Property Prediction Performance',
#             fontsize=13, fontweight='bold', pad=15)

# Set x-axis labels (SIGNIFICANTLY INCREASED FONTSIZE AND BOLD)
all_labels = grouped_methods + ['Ensemble', 'AGILE']
ax.set_xticks(list(x_grouped) + [x_ensemble, x_agile])
ax.set_xticklabels(all_labels, fontsize=18, fontweight='bold')

ax.set_ylim([0, 0.75])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Legend (SIGNIFICANTLY INCREASED SIZE)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='AI-DADDS'),
    Patch(facecolor='#e74c3c', edgecolor='black', label=r'Reported$^{\mathbf{2}}$')
]
ax.legend(handles=legend_elements, 
          loc='upper center', 
          framealpha=0.95, 
          ncol=2,
          markerscale=1.5,
          prop={'weight': 'bold', 'size': 18})
# Add note (INCREASED SIZE)
note_text = '*Negative R² displayed as 0 (in red)'
ax.text(0.02, 0.98, note_text,
        transform=ax.transAxes, fontsize=14.5, va='top', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.8))

plt.tight_layout()
plt.savefig('AI_DADDS_vs_Nature_Paper_AGILE_SEPARATED.png', dpi=300, bbox_inches='tight')
plt.savefig('AI_DADDS_vs_Nature_Paper_AGILE_SEPARATED.pdf', bbox_inches='tight')
print("✓ Figure saved with AGILE properly separated!")
plt.show()
