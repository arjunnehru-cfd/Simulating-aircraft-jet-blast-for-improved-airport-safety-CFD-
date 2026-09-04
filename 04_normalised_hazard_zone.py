# Author: Arjun Nehru
"""
Plot the normalised hazard zone (mean +/- 1 std) from saved data.
Reads hazard_zone_normalized_data.csv produced by 03_hazard_zone_analysis.py.

Produces Figure04_hazard_zone_normalized_meanstd.jpeg
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['lines.linewidth'] = 1

# ---- paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

CSV = os.path.join(DATA_DIR, 'hazard_zone_normalized_data.csv')

MODEL_STYLES = {
    'Spalart-Allmaras':        dict(color='#7FA8D8', linestyle='--', marker='o'),
    'Spalart-Allmaras w/ CFC': dict(color='#E08B8B', linestyle='-.', marker='s'),
    r'RSM $\epsilon$-linear':  dict(color='#8CC08C', linestyle=':',  marker='^'),
}

# ---- read ----
data = np.genfromtxt(CSV, delimiter=',', names=True)
names = list(data.dtype.names)

x_star = data['x_star']
upper_mean, upper_std = data['upper_mean'], data['upper_std']
lower_mean, lower_std = data['lower_mean'], data['lower_std']

model_cols = names[5:]
curves = {}
for i, label in enumerate(MODEL_STYLES):
    curves[label] = (data[model_cols[2 * i]], data[model_cols[2 * i + 1]])

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(x_star, upper_mean - upper_std, upper_mean + upper_std,
                alpha=0.20, color='C0', zorder=0)
ax.fill_between(x_star, lower_mean - lower_std, lower_mean + lower_std,
                alpha=0.20, color='C0', zorder=0)

for label, (z_upper, z_lower) in curves.items():
    s = MODEL_STYLES[label]
    ax.plot(x_star, z_upper, color=s['color'], linestyle=s['linestyle'],
            marker=s['marker'], markevery=20, markersize=4,
            markerfacecolor='white', markeredgecolor=s['color'],
            linewidth=1.0, zorder=2, label=label)
    ax.plot(x_star, z_lower, color=s['color'], linestyle=s['linestyle'],
            marker=s['marker'], markevery=20, markersize=4,
            markerfacecolor='white', markeredgecolor=s['color'],
            linewidth=1.0, zorder=2)

ax.plot(x_star, upper_mean, 'k-', linewidth=2.2, zorder=4, label='Mean')
ax.plot(x_star, lower_mean, 'k-', linewidth=2.2, zorder=4)

ax.set_xlabel(r'$x^* = x / X_h$', fontsize=12)
ax.set_ylabel(r'$z/D$', fontsize=12)
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02),
          ncol=4, fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Figure04_hazard_zone_normalized_meanstd.jpeg'), dpi=1000)
plt.show()
