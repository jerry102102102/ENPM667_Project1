# cfc_flc_system.py
# Paper-faithful 2-DOF vibration model and controllers for Sec. 4.2
# - Dynamics: Eq. (3)
# - Separated form for control design: Eq. (21)
# - Workpiece stiffness kw(xw, yw): Eq. (13)
# - Gas-spring stiffness ks(p): Eq. (14)
# - CFC: Eqs. (31), (35)–(36)
# - FLC: Eq. (37)

import numpy as np

# ---------- Workpiece stiffness kw(xw, yw) — Eq. (13) ----------
def kw_plate(xw: float, yw: float, D: float = 800.0, a: float = 0.5, b: float = 0.5) -> float:
    """
    Equivalent local stiffness of a clamped rectangular thin plate at (xw, yw).
    Matches Eq. (13) of the paper. Units: [N/m].
    """
    c1 = (1.0 + np.cos(np.pi * xw / a))
    c2 = (1.0 + np.cos(np.pi * yw / b))
    denom = (c1 * c1) * (c2 * c2)
    denom = np.maximum(denom, 1e-12)  # guard
    num = D * (np.pi**4) * (3 * a**4 + 2 * a**2 * b**2 + 3 * b**4)
    return num / (a**3 * b**3 * denom)

# ---------- Default parameters (Sec. 4.2) ----------
def default_params():
    p = {}

    # Masses (kg): mw=13.5, mr=2000, ms1=5, ms2=5; ns=7 support modules
    p["mw"] = 13.5
    p["mr"] = 2000.0
    p["ms1"] = 5.0
    p["ms2"] = 5.0
    p["ns"] = 7

    # Robot normal stiffness kr (within Fig. 10 range 1e6–1e7 N/m).
    # Use a representative constant for Sec. 4.2 simulations.
    p["kr"] = 3.0e6  # [N/m]

    # Rayleigh damping coefficients (Sec. 4.2 sets α=β=1)
    p["alpha"] = 1.0
    p["beta"] = 1.0

    # Gas-spring parameters (Sec. 4.1/4.2): γ=1.4, s = π·1e-4 m^2, l = 0.01 m
    p["gamma"] = 1.4
    p["s_area"] = np.pi * 1e-4
    p["l_ext"] = 0.01

    # Pressure bound ±1 MPa
    p["p_max"] = 1.0e6

    # CFC parameters (Sec. 4.2): κ1=100, κ2=1e6, P=100 I
    p["kappa1"] = 100.0
    p["kappa2"] = 1.0e6
    p["P_gain"] = 100.0

    # Plate geometry/stiffness used in kw(xw, yw)
    p["D"] = 800.0
    p["a"] = 0.5
    p["b"] = 0.5

    # FLC sweep grids (around reported optimum kd≈13888.89, kp≈55555.56)
    p["kp_grid"] = np.linspace(2.0e4, 1.0e5, 36)
    p["kd_grid"] = np.linspace(5.0e3, 2.0e4, 36)
    return p

# ---------- Task trajectory and excitation (Sec. 4.2) ----------
def trajectory(t: float):
    """
    Returns desired workpiece displacement and its derivatives, the moving contact
    point (xw, yw), and external axial force Fa(t).
    - z̄w = 1e-3 * sin(t), t ∈ [0, 3π]
    - xw = 0.05 t - 0.4, yw = 0
    - Fa = 100 * sin(2π t) [N]
    """
    zbar = 1e-3 * np.sin(t)
    dzbar = 1e-3 * np.cos(t)
    ddzbar = -1e-3 * np.sin(t)
    xw = 0.05 * t - 0.4
    yw = 0.0
    Fa = 100.0 * np.sin(2 * np.pi * t)
    return zbar, dzbar, ddzbar, xw, yw, Fa

# ---------- Gas-spring stiffness ks(p) — Eq. (14) ----------
def ks_gas(p: float, params) -> float:
    gamma = params["gamma"]
    s = params["s_area"]
    l = params["l_ext"]
    return gamma * s * p / l

# ---------- Separated terms for control design (Eq. (21)) ----------
def separated_terms(z: np.ndarray, dz: np.ndarray, t: float, params):
    """
    Builds M, Ctilde, Ktilde, Ftilde (no p) and the input vector f(z, dz) such that:
        M zdd + Ctilde dz + Ktilde z + Ftilde = f(z, dz) * p(t)
    matching Eq. (21).
    """
    mw, mr, ms1, ms2 = params["mw"], params["mr"], params["ms1"], params["ms2"]
    ns, kr = params["ns"], params["kr"]
    alpha, beta = params["alpha"], params["beta"]

    zbar, dzbar, ddzbar, xw, yw, Fa = trajectory(t)
    kw = kw_plate(xw, yw, params["D"], params["a"], params["b"])

    # Mass and "p-independent" stiffness/damping (tilded terms)
    M = np.diag([mw + ms1, mr + ms2])
    Ktilde = np.diag([kw, kr])
    Ctilde = alpha * M + beta * Ktilde
    Ftilde = np.array([[-Fa], [0.0]])

    # Input vector f(z, dz) (2x1) multiplying the scalar pressure p(t)
    gamma = params["gamma"]
    s = params["s_area"]
    l = params["l_ext"]

    # Matrices used in Eq. (21)
    A2 = np.array([[-1.0, 1.0],
                   [ 1.0,-1.0]])  # multiplies z and dz
    f_vec = (beta * ns * gamma * s / l) * (A2 @ dz) + (ns * gamma * s / l) * (A2 @ z) + np.array([[-ns * s],
                                                                                                  [ ns * s]])
    return M, Ctilde, Ktilde, Ftilde, f_vec, zbar, dzbar, ddzbar

# ---------- Full dynamics for simulation (original Eq. (3)) ----------
def system_mats_with_p(z: np.ndarray, dz: np.ndarray, t: float, p_scalar: float, params):
    """
    Returns M, C(p), K(p), F(p) for the original (non-separated) dynamics in Eq. (3).
    """
    mw, mr, ms1, ms2 = params["mw"], params["mr"], params["ms1"], params["ms2"]
    ns, kr = params["ns"], params["kr"]
    alpha, beta = params["alpha"], params["beta"]

    zbar, dzbar, ddzbar, xw, yw, Fa = trajectory(t)
    kw = kw_plate(xw, yw, params["D"], params["a"], params["b"])
    ks = ks_gas(p_scalar, params)

    M = np.diag([mw + ms1, mr + ms2])
    # Stiffness with gas-spring coupling (ns * ks) — Eq. (3)
    K = np.array([[kw + ns * ks,   -ns * ks],
                  [ -ns * ks,   ns * ks + kr]])
    # Rayleigh damping using current K
    C = alpha * M + beta * K
    # Force vector with pressure-dependent total support force Fs = ns * s * p
    Fs = params["ns"] * params["s_area"] * p_scalar
    F = np.array([[Fa - Fs],
                  [     Fs]])
    return M, C, K, F, zbar, dzbar, ddzbar

# ---------- State derivative for numerical integration ----------
def f_state(state: np.ndarray, t: float, u_pressure: float, params):
    """
    Right-hand side for state x = [zw, zr, dzw, dzr], given pressure input p(t).
    Integrates the original Eq. (3) so the simulation includes the true force–stiffness interdependence.
    """
    z = state[0:2].reshape(2, 1)
    dz = state[2:4].reshape(2, 1)

    # Clip pressure to ±1 MPa
    p_clip = float(np.clip(u_pressure, -params["p_max"], params["p_max"]))
    M, C, K, F, *_ = system_mats_with_p(z, dz, t, p_clip, params)

    ddz = np.linalg.solve(M, -C @ dz - K @ z + F)
    return np.vstack([dz, ddz]).flatten()

# ---------- CFC pressure (Eqs. (31), (35)–(36)) ----------
def cfc_pressure(state: np.ndarray, t: float, params) -> float:
    """
    Pressure input p(t) = p1 + p2 per the constraint-following controller.
    A = [1, 0] imposes the servo constraint on the workpiece channel.
    """
    z = state[0:2].reshape(2, 1)
    dz = state[2:4].reshape(2, 1)

    M, Ctilde, Ktilde, Ftilde, f_vec, zbar, dzbar, ddzbar = separated_terms(z, dz, t, params)

    # Constraint matrices/vectors
    A = np.array([[1.0, 0.0]])                      # only zw constrained
    b = np.array([[ddzbar - params["kappa1"] * (dz[0, 0] - dzbar)],
                  [0.0]])                            # Eq. (24)
    # Minimal-norm nominal term p1 — Eq. (31)
    Minv = np.linalg.inv(M)
    term = b + A @ Minv @ (Ctilde @ dz + Ktilde @ z + Ftilde)
    gamma = (A @ Minv @ f_vec)[0, 0]                # scalar
    p1 = float(term[0, 0] / max(abs(gamma), 1e-12))

    # Feedback term p2 — Eq. (36)
    e = (A @ dz - np.array([[dzbar], [0.0]]))[0, 0]  # constraint-following error (first channel)
    p2 = - params["kappa2"] * gamma * params["P_gain"] * e

    p = np.clip(p1 + p2, -params["p_max"], params["p_max"])
    return float(p)

# ---------- FLC pressure (Eq. (37)) ----------
def flc_pressure(state: np.ndarray, t: float, params, kp: float, kd: float) -> float:
    """
    Feedback linearization pressure (Eq. 37) with numeric safeguards:
    - sensor saturation on z, dz (bounded measurement),
    - ridge on pseudoinverse denom,
    - clipping on intermediate numerics,
    - final pressure clipping to ±p_max.
    """
    # raw states
    z_raw = state[0:2].reshape(2, 1)
    dz_raw = state[2:4].reshape(2, 1)

    # ----- bounded measurements for controller only (not plant) -----
    z = np.clip(z_raw,  -1e-2,  1e-2)    # ±10 mm
    dz = np.clip(dz_raw, -1.0,   1.0)    # ±1 m/s

    M, Ctilde, Ktilde, Ftilde, f_vec, zbar, dzbar, ddzbar = separated_terms(z, dz, t, params)

    Z = np.array([[ddzbar + kd * (dzbar - dz[0, 0]) + kp * (zbar - z[0, 0])],
                  [0.0]])
    rhs = M @ Z + Ctilde @ dz + Ktilde @ z + Ftilde   # 2x1

    # numeric guards
    eps = 1e-9
    f_vec = np.clip(f_vec, -1e6, 1e6)
    rhs   = np.clip(rhs,   -1e6, 1e6)

    denom = float((f_vec.T @ f_vec)[0, 0]) + eps
    num   = float((f_vec.T @ rhs)[0, 0])
    num   = float(np.clip(num, -1e12, 1e12))

    p = num / denom
    return float(np.clip(p, -params["p_max"], params["p_max"]))
