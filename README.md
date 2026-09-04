# Simulating Aircraft Jet Blast for Improved Airport Safety (CFD)

RANS turbulence model assessment for airport jet blast hazard zone prediction. Compares Spalart-Allmaras, SA with corner flow correction (CFC), and Reynolds Stress Model (RSM) against experimental data from Davis & Winarto (1980) and Maslov et al. (2001) for a cylindrical jet above a ground plane at h/D = 0.5, Re = 1e5.

## Repository Structure

```
git_python_scripts/
    02_u_profiles.py                 # Centerline velocity decay + radial/vertical profiles
    03_hazard_zone_analysis.py       # Hazard zone boundary extraction and normalisation
    04_normalised_hazard_zone.py     # Normalised hazard zone plot (mean +/- 1 std)
    data/
        dw_digitized.csv             # Davis & Winarto (1980) digitised decay data
        maslov_decay_digitised.csv   # Maslov et al. (2001) digitised decay data
        radial_profile_maslov.csv    # Maslov et al. (2001) radial profile
        vertical_profile_maslov.csv  # Maslov et al. (2001) vertical profile
        hazard_zone_maslov.csv       # Maslov et al. (2001) hazard zone boundary
        radial_velocity_profiles/    # Fluent XY exports at x/D = 50 (lateral)
        vertical_velocity_profiles/  # Fluent XY exports at x/D = 50 (vertical)
        Max_u_decay/                 # Fluent maximum velocity reports
        Hazard_zones/                # Tecplot FEPolygon slice exports (36 files)
    output/                          # Generated figures (not tracked)
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- A LaTeX distribution (e.g. TeX Live, MiKTeX) with `amsmath` for rendering plot labels

Install the Python dependencies:
```bash
pip install numpy scipy matplotlib
```

## Usage

Run the scripts from the repository root:

```bash
python 02_u_profiles.py
python 03_hazard_zone_analysis.py
python 04_normalised_hazard_zone.py
```

Script 03 generates `hazard_zone_normalized_data.csv` in the `data/` folder, which script 04 reads. Run them in order.

Figures are saved to `output/`.

## Outputs

| Script | Figure |
|--------|--------|
| `02_u_profiles.py` | `Figure02_combined_profiles.pdf` |
| `03_hazard_zone_analysis.py` | `Figure04_hazard_zones.jpeg` |
| `04_normalised_hazard_zone.py` | `Figure05_hazard_zone_normalized_meanstd.jpeg` |

## References

- Davis, M. R. & Winarto, H. (1980). Jet diffusion from a circular nozzle above a solid plane. *Journal of Fluid Mechanics*, 101(1), 201-221.
- Maslov, V., Mineev, B., Secundov, A., Vorobiev, A. & Birch, S. (2001). An experimental study of three-dimensional wall jets. *39th AIAA Aerospace Sciences Meeting and Exhibit*, Reno, NV.

## Author

Arjun Nehru
MSc Computational Fluid Dynamics, Cranfield University
