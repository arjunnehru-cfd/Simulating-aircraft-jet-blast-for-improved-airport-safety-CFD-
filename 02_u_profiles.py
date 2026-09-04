# Author: Arjun Nehru
"""
Turbulence model comparison: centerline velocity decay, radial and
vertical self-similar profiles at x/D = 50.

Produces Figure02_combined_profiles.pdf/.jpeg
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d


matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.dashed_pattern'] = [5, 5]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 3]
matplotlib.rcParams['lines.dashdot_pattern'] = [1, 3]

plt.close('all')

# ---- paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

RADIAL_DIR = os.path.join(DATA_DIR, 'radial_velocity_profiles')
VERTICAL_DIR = os.path.join(DATA_DIR, 'vertical_velocity_profiles')
DECAY_DIR = os.path.join(DATA_DIR, 'Max_u_decay')

# ---- I/O helpers ----

def read_fluent_xy(filepath):
    """Read a Fluent XY plot export, returning x and y arrays."""
    x_vals, y_vals = [], []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(('(', ')', 'title', 'labels',
                                      'cellnumber', '((xy/key/label')):
                continue
            if 'Label' in s:
                continue
            parts = s.split()
            if len(parts) == 2:
                try:
                    x_vals.append(float(parts[0]))
                    y_vals.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(x_vals), np.array(y_vals)


def read_fluent_max_report(filepath):
    """Read a Fluent maximum-velocity report, returning x/D and U_max."""
    with open(filepath) as f:
        text = f.read()
    pattern = re.compile(r'xd(\d+)\s+([\d.eE+-]+)')
    matches = pattern.findall(text)
    rows = np.array([(int(m[0]), float(m[1])) for m in matches])
    if abs(rows[-1, 1] - rows[0, 1]) < 1e-6:
        rows = rows[:-1]
    return rows[:, 0], rows[:, 1]


# ---- normalisation routines ----

def normalize_radial_profile(y, u):
    """Half-width normalisation for the lateral (z) profile."""
    Um = np.nanmax(u)
    urat = u / Um

    idx_max = np.argmax(u)
    z_peak = y[idx_max]
    z_shifted = np.abs(y - z_peak)

    sort_idx = np.argsort(z_shifted)
    z_sym = z_shifted[sort_idx]
    u_sym = urat[sort_idx]

    f = interp1d(u_sym, z_sym, kind='linear', fill_value='extrapolate')
    try:
        y_half = float(f(0.5))
    except Exception:
        y_half = float(z_sym[np.argmin(np.abs(u_sym - 0.5))])

    if y_half == 0:
        y_half = 1e-6

    return z_sym / y_half, u_sym


def verhoff_normalize(y, u):
    """Half-width normalisation for the vertical (y) profile."""
    y_shifted = y - np.min(y)
    idx_max = np.argmax(u)
    Umax = u[idx_max]

    u_norm = u / Umax
    half_u = 0.5 * Umax

    outer_u = u[idx_max:]
    outer_y = y_shifted[idx_max:]
    idx_half_outer = np.argmin(np.abs(outer_u - half_u))
    y_half = outer_y[idx_half_outer]

    if y_half == 0:
        y_half = 1e-6

    return y_shifted / y_half, u_norm


# ---- model definitions ----

D = 1.0
U_J = 1.0

models_Um = {
    'Spalart-Allmaras':      os.path.join(DECAY_DIR, 'velocity-report-sa-i5.txt'),
    'Spalart-Allmaras w CFC': os.path.join(DECAY_DIR, 'velocity-report-sa-cft-i5.txt'),
    r'RSM $\epsilon$-linear': os.path.join(DECAY_DIR, 'velocity-report-rsm-i5.txt'),
}

models_radial = {
    'Spalart-Allmaras':      os.path.join(RADIAL_DIR, 'x50_z_profile_u_SA_i5.xy'),
    'Spalart-Allmaras w CFC': os.path.join(RADIAL_DIR, 'x50_z_profile_u_SA_CFT_i5.xy'),
    r'RSM $\epsilon$-linear': os.path.join(RADIAL_DIR, 'x50_z_profile_u_RSM_i5.xy'),
}

models_vertical = {
    'Spalart-Allmaras':      os.path.join(VERTICAL_DIR, 'x50_y_profile_u_SA_i5.xy'),
    'Spalart-Allmaras w CFC': os.path.join(VERTICAL_DIR, 'x50_y_profile_u_SA_CFT_i5.xy'),
    r'RSM $\epsilon$-linear': os.path.join(VERTICAL_DIR, 'x50_y_profile_u_RSM_i5.xy'),
}

COLORS = {
    'Spalart-Allmaras': 'blue',
    'Spalart-Allmaras w CFC': 'red',
    r'RSM $\epsilon$-linear': 'green',
}

# ---- figure setup ----

fig = plt.figure(num=1, figsize=(15 / 2.54, 19 / 2.54))
gs = GridSpec(2, 4)
gs.update(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.6, hspace=0.35)

graph_lbls = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$']

# (a) centerline velocity decay

ax1 = plt.subplot(gs[0, 0:2])
ax1.text(0.92, 0.92, graph_lbls[0], ha='center', va='center',
         transform=ax1.transAxes)

for model, filepath in models_Um.items():
    if not os.path.exists(filepath):
        continue
    xd, Um = read_fluent_max_report(filepath)
    ax1.plot(xd, U_J / Um, label=model, color=COLORS[model], linewidth=1.5)

# digitised reference data
dw_data = np.loadtxt(os.path.join(DATA_DIR, 'dw_digitized.csv'), delimiter=',')
decay_data = np.loadtxt(os.path.join(DATA_DIR, 'maslov_decay_digitised.csv'), delimiter=',')

ax1.plot(dw_data[:, 0], dw_data[:, 1], 'k^', markersize=4, zorder=5,
         markevery=3, label=r'Davis \& Winarto (1980), $h/D=0.5$')
ax1.plot(decay_data[:, 0], decay_data[:, 1], 'ko', markersize=4, zorder=5,
         label=r'Maslov et al.(2001)')

# correlation extrapolation
xd_ext = np.linspace(64.0, 250.0, 50)
ax1.plot(xd_ext, 0.15 * (xd_ext - 5.6), 'k--', linewidth=1.0)

ax1.set_xlabel(r'$x/D$', fontsize=12)
ax1.set_ylabel(r'$U_j / U_{\mathrm{max}}$', fontsize=12)
ax1.set_xlim(0, 250)
ax1.set_ylim(0, 40)
ax1.grid(True, linestyle='--', alpha=0.5)

# (b) radial velocity profiles at x/D = 50

ax2 = plt.subplot(gs[0, 2:4])
ax2.text(0.92, 0.92, graph_lbls[1], ha='center', va='center',
         transform=ax2.transAxes)

for label, fpath in models_radial.items():
    if not os.path.exists(fpath):
        continue
    y_raw, u_raw = read_fluent_xy(fpath)
    order = np.argsort(y_raw)
    eta, urat = normalize_radial_profile(y_raw[order], u_raw[order])
    ax2.plot(eta, urat, label=label, color=COLORS[label], linewidth=1.5)

radial_data = np.loadtxt(os.path.join(DATA_DIR, 'radial_profile_maslov.csv'),
                         delimiter=',')
ax2.plot(radial_data[:, 0], radial_data[:, 1], 'k--', linewidth=1.5,
         zorder=5, label='Maslov et al. (2001) Radial')

ax2.set_xlabel(r'$\zeta$ = $z / B_z$', fontsize=12)
ax2.set_ylabel(r'$U / U_{\mathrm{max}}$', fontsize=12)
ax2.set_xlim(0.0, 3.0)
ax2.set_ylim(0.0, 1.05)
ax2.grid(True, linestyle='--', alpha=0.5)

# (c) vertical velocity profile at x/D = 50

ax3 = plt.subplot(gs[1, 0:2])
ax3.text(0.92, 0.92, graph_lbls[2], ha='center', va='center',
         transform=ax3.transAxes)

for label, filepath in models_vertical.items():
    if not os.path.exists(filepath):
        continue
    y_coords, u = read_fluent_xy(filepath)
    if len(y_coords) == 0:
        continue

    sort_idx = np.argsort(y_coords)
    y_coords, u = y_coords[sort_idx], u[sort_idx]
    zeta, u_norm = verhoff_normalize(y_coords, u)

    ax3.plot(u_norm, zeta, label=label, color=COLORS[label],
             linestyle='none', marker='o', markersize=4)

vertical_data = np.loadtxt(os.path.join(DATA_DIR, 'vertical_profile_maslov.csv'),
                           delimiter=',')
ax3.plot(vertical_data[:, 0], vertical_data[:, 1], color='k',
         linestyle=(0, (3, 2, 1, 2)), linewidth=1.5, zorder=5,
         label='Maslov et al.(2001) Vertical')

ax3.set_xlabel(r'$U / U_{\mathrm{max}}$', fontsize=12)
ax3.set_ylabel(r'$\eta$ = $y / B_y$', fontsize=12)
ax3.set_xlim(0, 1.05)
ax3.set_ylim(0, 4)
ax3.grid(True, linestyle='--', alpha=0.7)

# shared legend

ax_leg = plt.subplot(gs[1, 2:4])
ax_leg.axis('off')

handles, labels = [], []
for ax in [ax1, ax2, ax3]:
    h, l = ax.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in labels:
            handles.append(hi)
            labels.append(li)

ax_leg.legend(handles, labels, loc='center', fontsize=8, frameon=False)

# RMSD against digitised data

eta_ref = radial_data[:, 0]
u_ref_rad = radial_data[:, 1]
zeta_ref = vertical_data[:, 1]
u_ref_vert = vertical_data[:, 0]

_o = np.argsort(eta_ref)
eta_ref_s, u_ref_rad_s = eta_ref[_o], u_ref_rad[_o]
_o = np.argsort(zeta_ref)
zeta_ref_s, u_ref_vert_s = zeta_ref[_o], u_ref_vert[_o]

print('\n-- Lateral RMSD (%d digitised points) --' % len(eta_ref_s))
for label, fpath in models_radial.items():
    if not os.path.exists(fpath):
        continue
    z, u = read_fluent_xy(fpath)
    o = np.argsort(z)
    eta, urat = normalize_radial_profile(z[o], u[o])
    o2 = np.argsort(eta)
    dev = np.interp(eta_ref_s, eta[o2], urat[o2]) - u_ref_rad_s
    print(f'  {label:38s} RMSD = {np.sqrt(np.mean(dev**2)):.4f}')

print('\n-- Vertical RMSD (%d digitised points) --' % len(zeta_ref_s))
for label, fpath in models_vertical.items():
    if not os.path.exists(fpath):
        continue
    y, u = read_fluent_xy(fpath)
    o = np.argsort(y)
    zv, un = verhoff_normalize(y[o], u[o])
    o2 = np.argsort(zv)
    dev = np.interp(zeta_ref_s, zv[o2], un[o2]) - u_ref_vert_s
    print(f'  {label:38s} RMSD = {np.sqrt(np.mean(dev**2)):.4f}')

print('\nDigitisation resolution:')
print('  lateral  mean |dU| = %.4f' % np.mean(np.abs(np.diff(u_ref_rad_s))))
print('  vertical mean |dU| = %.4f' % np.mean(np.abs(np.diff(u_ref_vert_s))))


#fig.savefig(os.path.join(OUTPUT_DIR, 'Figure02_combined_profiles.pdf'), dpi=1000, bbox_inches='tight')
#fig.savefig(os.path.join(OUTPUT_DIR, 'Figure02_combined_profiles.jpeg'), dpi=1000, bbox_inches='tight')
plt.show()