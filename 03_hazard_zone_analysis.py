# Author: Arjun Nehru
"""
Hazard zone comparison across turbulence models (SA, SA-CFC, RSM).

Reads Tecplot FEPolygon slice exports at twelve heights, projects onto a
common x-z grid, extracts hazard zone boundaries, normalises by each
model's streamwise extent, and computes the mean/std across models.

Produces Figure04_hazard_zones.jpeg and hazard_zone_normalized_data.csv
"""
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import MultipleLocator

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['lines.dashed_pattern'] = [5, 5]
matplotlib.rcParams['lines.dotted_pattern'] = [1, 3]
matplotlib.rcParams['lines.dashdot_pattern'] = [1, 3]

# ---- paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
HAZARD_DIR = os.path.join(DATA_DIR, 'Hazard_zones')

# ---- parameters ----
MODELS = {
    'Spalart-Allmaras': 'sa',
    'Spalart-Allmaras w/ CFC': 'sa-cft',
    r'RSM $\epsilon$-linear': 'rsm',
}

HEIGHT_TAGS = {'005': 0.05, '01': 0.1, '025': 0.25, '04': 0.4, '05': 0.5,
               '06': 0.6, '075': 0.75, '1': 1.0, '125': 1.25, '15': 1.5,
               '175': 1.75, '2': 2.0}

X_RANGE = (0.0, 250.0)
Z_RANGE = (-20.0, 20.0)
NX, NZ = 900, 300

UREF = 1.0
THRESHOLD_RATIO = 0.057
MIN_LOOP_POINTS = 50
N_STATIONS = 250
SMOOTH_WINDOW = 9
BOUNDARY_SMOOTH = 15

COLORS = {
    'Spalart-Allmaras': 'blue',
    'Spalart-Allmaras w/ CFC': 'red',
    r'RSM $\epsilon$-linear': 'green',
}


# ---- I/O and processing ----

def parse_tecplot_slice(filepath):
    """Parse a Tecplot FEPolygon ASCII slice, returning X, Z, U arrays."""
    with open(filepath, 'r', errors='ignore') as f:
        text = f.read()
    lines = text.splitlines()

    nodes_line = next(l for l in lines if 'Nodes=' in l)
    n_nodes = int(re.search(r'Nodes=(\d+)', nodes_line).group(1))
    dt_idx = next(i for i, l in enumerate(lines) if l.strip().startswith('DT='))

    tokens = ' '.join(lines[dt_idx + 1:]).split()
    X = np.array(tokens[0:n_nodes], dtype=float)
    Z = np.array(tokens[n_nodes:2 * n_nodes], dtype=float)
    U = np.array(tokens[2 * n_nodes:3 * n_nodes], dtype=float)
    return X, Z, U


def project_over_height(prefix, xg, zg):
    """Maximum velocity over all slice heights on the common grid."""
    Umax = np.full(xg.shape, np.nan)
    for tag, y in sorted(HEIGHT_TAGS.items(), key=lambda kv: kv[1]):
        filepath = os.path.join(HAZARD_DIR, f'{prefix}_y{tag}.dat')
        if not os.path.exists(filepath):
            print(f'  missing {prefix}_y{tag}.dat')
            continue
        X, Z, U = parse_tecplot_slice(filepath)
        keep = (X >= 0.0) & (X <= X_RANGE[1] + 10) & (np.abs(Z) <= abs(Z_RANGE[0]) + 5)
        Umax = np.fmax(Umax, griddata((X[keep], Z[keep]), U[keep], (xg, zg),
                                      method='linear'))
    return Umax


def extract_hazard_boundary(xg, zg, Umax, threshold):
    """Extract the hazard zone contour, return [N,2] array of [x,z] points."""
    fig, ax = plt.subplots()
    cs = ax.contour(xg, zg, Umax, levels=[threshold])
    plt.close(fig)
    loops = [p for p in cs.allsegs[0] if len(p) >= MIN_LOOP_POINTS]
    if not loops:
        raise RuntimeError(f'No hazard zone contour found at threshold {threshold}')
    return max(loops, key=len)


def smooth_curve(z, window):
    if window <= 1:
        return z
    pad = window // 2
    z_padded = np.pad(z, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(z_padded, kernel, mode='same')[pad:pad + len(z)]


def smooth_boundary(boundary, window):
    """Moving average along the boundary path, endpoints held fixed."""
    if window <= 1:
        return boundary
    smoothed = np.column_stack([smooth_curve(boundary[:, 0], window),
                                smooth_curve(boundary[:, 1], window)])
    smoothed[0] = boundary[0]
    smoothed[-1] = boundary[-1]
    return smoothed


def split_loop_at_extrema(boundary):
    """Split a closed boundary loop into upper and lower arcs."""
    x = boundary[:, 0]
    i_start = np.argmin(x)
    i_tip = np.argmax(x)
    if i_start < i_tip:
        arc1 = boundary[i_start:i_tip + 1]
        arc2 = np.vstack([boundary[i_tip:], boundary[:i_start + 1]])
    else:
        arc1 = np.vstack([boundary[i_start:], boundary[:i_tip + 1]])
        arc2 = boundary[i_tip:i_start + 1]
    if arc1[0, 0] > arc1[-1, 0]:
        arc1 = arc1[::-1]
    if arc2[0, 0] > arc2[-1, 0]:
        arc2 = arc2[::-1]
    if arc1[:, 1].mean() < arc2[:, 1].mean():
        return arc2, arc1
    return arc1, arc2


def normalize_and_resample(upper, lower):
    """Normalize x by streamwise extent, resample onto common x* grid."""
    L = max(upper[:, 0].max(), lower[:, 0].max())
    x_star = np.linspace(0, 1, N_STATIONS)
    z_upper = np.interp(x_star, upper[:, 0] / L, upper[:, 1])
    z_lower = np.interp(x_star, lower[:, 0] / L, lower[:, 1])

    start_z, tip_z = z_upper[0], z_upper[-1]
    z_upper = smooth_curve(z_upper, SMOOTH_WINDOW)
    z_lower = smooth_curve(z_lower, SMOOTH_WINDOW)
    z_upper[0] = z_lower[0] = start_z
    z_upper[-1] = z_lower[-1] = tip_z
    return x_star, z_upper, z_lower, L


# main

def main():
    threshold = THRESHOLD_RATIO * UREF
    results = {}

    x = np.linspace(*X_RANGE, NX)
    z = np.linspace(*Z_RANGE, NZ)
    xg, zg = np.meshgrid(x, z)

    for name, prefix in MODELS.items():
        print(name)
        Umax = project_over_height(prefix, xg, zg)
        boundary = extract_hazard_boundary(xg, zg, Umax, threshold)
        upper, lower = split_loop_at_extrema(boundary)
        x_star, z_upper, z_lower, L = normalize_and_resample(upper, lower)
        results[name] = dict(boundary=boundary, x_star=x_star,
                             z_upper=z_upper, z_lower=z_lower, L=L)
        print(f'  Xh/D = {L:.2f}, Zh/D = {np.abs(boundary[:, 1]).max():.2f}')

    # mean and std across models
    x_star = next(iter(results.values()))['x_star']
    upper_stack = np.vstack([results[m]['z_upper'] for m in MODELS])
    lower_stack = np.vstack([results[m]['z_lower'] for m in MODELS])
    upper_mean, upper_std = upper_stack.mean(axis=0), upper_stack.std(axis=0)
    lower_mean, lower_std = lower_stack.mean(axis=0), lower_stack.std(axis=0)

    # Maslov reference data
    maslov_data = np.loadtxt(os.path.join(DATA_DIR, 'hazard_zone_maslov.csv'),
                             delimiter=',')
    x_maslov, z_maslov = maslov_data[:, 0], maslov_data[:, 1]

    # Figure 4: raw hazard zone boundaries
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in MODELS:
        b = smooth_boundary(results[name]['boundary'], BOUNDARY_SMOOTH)
        ax.plot(b[:, 0], b[:, 1], linewidth=1.5, label=name, color=COLORS[name])

    ax.plot(x_maslov, z_maslov, 'ko', linewidth=1.5, label='Maslov et al. (2001)')

    ax.set_xlabel(r'$x/D$', fontsize=12)
    ax.set_ylabel(r'$z/D$', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-10, 230)
    ax.set_ylim(-30, 30)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    #fig.savefig(os.path.join(OUTPUT_DIR, 'Figure03_hazard_zones.jpeg'), dpi=1000)

    # height sweep summary
    print('\n%-28s %6s %8s %8s' % ('model', 'y/D', 'Xh/D', 'Zh/D'))
    for name, prefix in MODELS.items():
        for tag, y in sorted(HEIGHT_TAGS.items(), key=lambda kv: kv[1]):
            filepath = os.path.join(HAZARD_DIR, f'{prefix}_y{tag}.dat')
            if not os.path.exists(filepath):
                continue
            X, Z, U = parse_tecplot_slice(filepath)
            keep = (X >= 0.0) & (X <= X_RANGE[1] + 10) & (np.abs(Z) <= abs(Z_RANGE[0]) + 5)
            Ui = griddata((X[keep], Z[keep]), U[keep], (xg, zg), method='linear')
            b = extract_hazard_boundary(xg, zg, Ui, threshold)
            print('%-28s %6.2f %8.2f %8.2f' % (name, y, b[:, 0].max(),
                                                 np.abs(b[:, 1]).max()))

    # save normalized data (read by 05_normalised_hazard_zone.py)
    header = 'x_star,upper_mean,upper_std,lower_mean,lower_std,' + \
             ','.join(f'{m}_upper,{m}_lower' for m in MODELS)
    cols = [x_star, upper_mean, upper_std, lower_mean, lower_std]
    for m in MODELS:
        cols += [results[m]['z_upper'], results[m]['z_lower']]
    out = np.column_stack(cols)
    np.savetxt(os.path.join(DATA_DIR, 'hazard_zone_normalized_data.csv'), out,
               delimiter=',', header=header, comments='')


if __name__ == '__main__':
    main()
    plt.show()
