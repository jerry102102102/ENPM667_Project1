# Constraint-following vibration control simulations

This repo mirrors the simulations from *Constraint-following vibration control for robot follow-up support system under force-stiffness interdependence* ([paper link](https://www.sciencedirect.com/science/article/abs/pii/S0307904X25004895)). In short, the code recreates the paper's stiffness studies and the controller comparison between the proposed CFC strategy and the baseline FLC.

## Repository structure
- `src/stiffness_analysis/`: scripts that reproduce the three stiffness-related studies (robot compliance, gas-spring module, workpiece plate).
- `src/control_performance/`: equivalent-model dynamics, controller implementations, and the sweeps/plots used to compare CFC vs. FLC.
- `data/` and `figs/`: auto-created folders that collect CSV/NPZ dumps and Matplotlib figures.
- `pyproject.toml`, `uv.lock`: uv-managed project metadata; `main.py` is just a placeholder entry point.

## Environment setup with uv
1. Install [uv](https://docs.astral.sh/uv/) (one-time). On macOS or Linux, you can run:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   On Windows, download the installer from the uv docs or use `pipx install uv`.
2. From the repo root, create the virtual environment and install dependencies:
   ```bash
   uv sync
   ```
3. Launch scripts inside that environment with `uv run`, for example:
   ```bash
   # Run a stiffness study
   uv run python src/stiffness_analysis/kr_robot.py

   # Run the FLC gain sweep
   uv run python src/control_performance/flc_sim.py
   ```

## Simulation workflows
### 1. Stiffness analyses (Sec. 4.1 of the paper)
- `kr_robot.py`: evaluates the task-space stiffness of the UR5 stand-in robot and builds a workspace heatmap similar to Fig. 10.
- `ks_support.py`: recreates the gas-spring stiffness formula, tabulates a single operating point, and sweeps the (pressure, extension) surface.
- `kw_workpiece.py`: computes the equivalent plate stiffness map, compares the center deflection to the FEA numbers, and exports the distribution plot.

### 2. Control performance (Sec. 4.2)
- `flc.py` / `flc_sim.py`: defines the feedback-linearization controller, runs a grid search over `(k_p, k_d)`, and drops the best traces plus the error/pressure surfaces.
- `cfc.py` / `cfc_sim.py`: implements the constraint-following controller on the equivalent Eq.(21) plant and sweeps `(kappa_1, kappa_2)` to recreate the error/cost surfaces.
- `cfc_flc_system.py`: contains a more literal Eq.(3) implementation along with both controllers, useful when you want to include the force-stiffness interdependence explicitly.
- `cfc_flc_comparison.py`: runs both controllers on the same equivalent plant to mimic Fig. 12–13, writing comparison plots for error and pressure.

Each run stores its numeric metrics in `data/` (CSV or NPZ) and plots in `figs/`, making it easy to cross-check against the original figures.

## Reference
Y. Yang, B. Wu, C. Zhang, et al., *Constraint-following vibration control for robot follow-up support system under force-stiffness interdependence*, **Applied Mathematical Modelling**, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0307904X25004895
