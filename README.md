# Quantum Time Simulation

Interactive 3D double-slit simulation in Rust (Macroquad) exploring a phase-perturbed "quantum time" model. The app can run live visualizations, batch analyses, and automated sweeps that export CSV summaries for use in papers and plots.

[example video](https://www.youtube.com/watch?v=Ky9TNk5_orQ)

## Features
- Real-time 3D view with trails and intensity map on the detection screen.
- Batch analysis that fires 1M particles and writes `analysis_results.csv` plus a text summary.
- Automated sweeps for slit width, emission phase distributions, and slit tilt angles.
- Diagnostic mode to probe symmetry at the barrier.
- Python helpers to render publication-ready plots from the exported CSVs.

## Quickstart
1) Install Rust (stable toolchain).  
2) Run the simulator: `cargo run --release`  
3) Optional plotting dependencies: `python -m pip install -r requirements.txt`

## Controls (while the window is focused)
- `Space` — Reset simulation state.
- `A` — Run single analysis (1M particles) and write `analysis_results.csv` + `analysis_summary.txt`.
- `S` — Sweep slit width from 0.2–2.0; writes `analysis_results_width_*.csv` and `sweep_results.csv`.
- `P` — Sweep emission phase distributions (uniform/sinusoidal/Gaussian π); writes `analysis_results_phase_*.csv` and `phase_sweep_results.csv`.
- `T` — Sweep slit tilt angles (0°, 15°, -15°, 30°); writes `angle_sweep_results.csv`.
- `M` — Mirror transverse motion.
- `D` — Diagnostic symmetry check at the barrier (no screen hits).

During analyses/sweeps the simulation pauses normal motion, fires particles in batches, and reports metrics (contrast, FWHM, symmetry, centroids, tilt) in the HUD and summary files.

## Plotting the results
- Line profile of the 1M-particle run: `python plot_analysis.py` → `interference_pattern.png`
- Sweep curves (FWHM/contrast/symmetry vs slit width): `python plot_sweep.py` → `sweep_plot.png`

Both scripts read the CSVs produced by the simulator; rerun the relevant sweep/analysis if the files are missing.

## Repository layout
- `src/main.rs` — Macroquad simulator.
- `analysis_results*.csv`, `analysis_summary*.txt` — Saved outputs from analyses and sweeps.
- `plot_analysis.py`, `plot_sweep.py` — Plot helpers for paper figures.

## License
- Code: Apache-2.0 (see `LICENSE`).
- Generated data/plots (CSV, PNG): CC BY 4.0 (see `LICENSE-CC-BY-4.0.txt`).

For citations, tag a release or commit so your paper can reference an immutable snapshot.
