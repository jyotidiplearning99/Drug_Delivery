#!/usr/bin/env python
"""
Bar Chart: AI-DADDS vs Nature Paper
ULTRA-THIN bars with MINIMAL gap - Pushed to Origin with Large Text
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14
# Changed to DejaVu Sans which is always available
plt.rcParams['font.family'] = 'DejaVu Sans'

print("="*70)
print("Creating Ultra-Thin Bar Chart - Pushed to Origin")
print("="*70)

# AI-DADDS (5 trials - best MAE)
ai_trials = 5
ai_dadds_mae = 10.11  # %
ai_dadds_std = 1.70   # %
ai_dadds_best = 7.71  # %

# Nature Paper Reported¹ (10 trials)
paper_trials = 10
paper_mae = 9.44  # %

# Create figure with extra bottom space to prevent cropping
fig, ax = plt.subplots(figsize=(10, 7))

# Data
methods = ['AI-DADDS\n(LGBM)', 'Reported\n(LGBM)']
mae_values = [ai_dadds_mae, paper_mae]
errors = [ai_dadds_std, 0]
colors = ['#2ecc71', '#e74c3c']

# ULTRA-THIN bars positioned at origin
bar_width = 0.015
x_pos = np.array([0.02, 0.045])  # Start closer to origin

bars = ax.bar(
    x_pos, mae_values, width=bar_width,
    color=colors, edgecolor='black', 
    linewidth=1.5, alpha=0.85,
    yerr=errors, capsize=8, 
    error_kw={'linewidth': 2}
)

# Value labels with LARGER text - GREEN BAR TEXT MOVED FURTHER DOWN
for i, (bar, mae) in enumerate(zip(bars, mae_values)):
    h = bar.get_height()
    if i == 0:  # Green bar (AI-DADDS)
        label_text = f"MAE = {mae:.2f}%\n±{ai_dadds_std:.2f}%"
        y_text = h + ai_dadds_std - 1.2  # ↓ MOVED FURTHER DOWN
    else:
        label_text = f"MAE = {mae:.2f}%"
        y_text = h + 0.8
    
    ax.text(bar.get_x() + bar.get_width()/2., y_text,
            label_text, ha='center', va='bottom',
            fontsize=18, fontweight='bold')

# Best annotation - LARGER text
ax.text(x_pos[0], 2.5, f'Best: {ai_dadds_best:.2f}%',
        ha='center', fontsize=16, style='italic',
        bbox=dict(boxstyle='round,pad=0.45', 
                 facecolor='lightgreen', alpha=0.7))

# Labels - LARGER text
ax.set_ylabel('Mean Absolute Error (MAE) %', 
              fontsize=18, fontweight='bold')

# Left margin to prevent cropping
ax.set_xlim([-0.01, 0.07])
ax.margins(x=0)

# X-axis setup
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.tick_params(axis='x', pad=12)

# Y axis - increase upper limit slightly to accommodate text
ymax = max(ai_dadds_mae + ai_dadds_std, paper_mae) + 5
ax.set_ylim([0, ymax])
ax.set_ylim(bottom=0)
ax.margins(y=0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.tick_params(axis='y', labelsize=16)

# Move spines to push bars to origin
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))

# Legend - FIXED: Using Unicode superscript ¹ directly
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='AI-DADDS'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Reported¹')
]
ax.legend(handles=legend_elements, 
          loc='upper right', 
          bbox_to_anchor=(1.0, 1.0),
          framealpha=0.95, 
          ncol=1,
          markerscale=1.5,
          prop={'weight': 'bold', 'size': 16})

# ✅ FORCE x-axis labels to be BOLD and SIZE 18 (apply AFTER everything else)
for label in ax.get_xticklabels():
    label.set_fontsize(18)
    label.set_fontweight('bold')

# Extra bottom space for huge x labels
plt.tight_layout(pad=3.0)
fig.subplots_adjust(bottom=0.22)

# Save with increased padding
plt.savefig('trials_comparison_ORIGIN_LARGE_TEXT.png', dpi=300, bbox_inches='tight', pad_inches=0.7)
plt.savefig('trials_comparison_ORIGIN_LARGE_TEXT.pdf', bbox_inches='tight', pad_inches=0.7)
print("✓ Figure saved with bars at origin and large text!")
plt.show()

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"AI-DADDS ({ai_trials} trials):")
print(f"  MAE: {ai_dadds_mae:.2f}% ± {ai_dadds_std:.2f}%")
print(f"  Best trial: {ai_dadds_best:.2f}%")
print(f"\nReported¹ ({paper_trials} trials):")
print(f"  MAE: {paper_mae:.2f}%")
print(f"\nDifference: {ai_dadds_mae - paper_mae:+.2f}%")
print("="*70)
