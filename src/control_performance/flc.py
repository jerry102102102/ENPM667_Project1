"""
FLC controller and 2-DOF plant (paper-consistent magnitudes, Sec. 4.2).

State x = [z_w, z_s, dz_w, dz_s]^T.
Input is chamber pressure p [Pa], mapped to force via F = p * s_area.

We adopt the "equivalent transformed model" of Eq.(21):
  - The input map f has two nonzero rows: f = s_area * [alpha, beta]^T.
  - Paper uses alpha = beta = 1.

Key implementation detail:
  To avoid a "flat eRMS surface", we do a 2-by-2 algebraic solve using BOTH
  rows of the dynamics at each step, treating [z̈_s, p] as unknowns rather
  than using a single-row pseudoinverse. This ensures the virtual command on
  the workpiece channel actually propagates through the coupled dynamics.

All units are SI. Comments aim to be clear for reproducibility.
"""

from __future__ import annotations
import numpy as np


# ---------- Spatially varying workpiece stiffness k_w(xw, yw) (Sec. 4.1 / Eq. 13) ----------
def kw_plate(xw: float, yw: float, *, D: float, a: float, b: float) -> float:
    """
    Plate-equivalent local normal stiffness k_w(xw, yw) from the paper (Sec. 4.1).
    Units: D [N·m], a,b,xw,yw [m] -> k_w [N/m].

    Analytical surrogate consistent with basis-function intuition:
        k_w ∝ 1 / [(1+cos(pi*xw/a))^2 * (1+cos(pi*yw/b))^2]
    Stiffness grows near supported edges and is lower near the center.
    """
    c1 = 1.0 + np.cos(np.pi * xw / a)
    c2 = 1.0 + np.cos(np.pi * yw / b)
    denom = max((c1 * c1) * (c2 * c2), 1e-12)
    num = D * (np.pi**4) * (3 * a**4 + 2 * a**2 * b**2 + 3 * b**4)
    return float(num / (a**3 * b**3 * denom))


# ---------- Paper-consistent parameters ----------
def default_params() -> dict:
    """
    Parameters aligned with paper Sec. 4.2 (alpha = beta = 1).
    Mass aggregation example: ms = mr + ns*ms1 = 2000 + 7*5 = 2035 kg.
    """
    p: dict = {}
    p["mw"] = 13.5                 # workpiece mass [kg]
    p["ms"] = 2000.0 + 7.0 * 5.0   # equivalent support mass [kg] = 2035

    # Stiffness & damping
    p["kr"] = 3.0e6                # robot normal stiffness [N/m]
    p["kc"] = 1.0e6                # coupling stiffness [N/m]
    p["cw"] = 300.0                # damping at workpiece [N·s/m]
    p["cs"] = 300.0                # damping at support [N·s/m]

    # Pressure-to-force map and input sharing
    p["s_area"] = np.pi * 1e-4     # effective piston/diaphragm area [m^2]
    p["alpha"]  = 1.0              # Eq.(21) share to z_w row
    p["beta"]   = 1.0              # Eq.(21) share to z_s row
    p["p_max"]  = 1.0e6            # ±1 MPa saturation [Pa]

    # Plate parameters for k_w(xw,yw) (Sec. 4.1)
    p["D"] = 800.0                 # [N·m]
    p["a"] = 0.5                   # [m]
    p["b"] = 0.5                   # [m]

    # Integration window (paper uses t in [0, 3π])
    p["t0"] = 0.0
    p["t1"] = 3.0 * np.pi
    p["dt"] = 1e-3
    return p


# ---------- Reference & excitation (Sec. 4.2) ----------
def trajectory(t: float):
    """
    Returns (zbar, dzbar, ddzbar, xw, yw, Fa)

      zbar_w(t) = 1e-3 * sin(t), t ∈ [0, 3π]
      xw = 0.05*t - 0.4,  yw = 0
      Fa = 100 * sin(2π t)  [N]  (external excitation on the workpiece channel)
    """
    zbar   = 1e-3 * np.sin(t)
    dzbar  = 1e-3 * np.cos(t)
    ddzbar = -1e-3 * np.sin(t)
    xw = 0.05 * t - 0.4
    yw = 0.0
    Fa = 100.0 * np.sin(2.0 * np.pi * t)
    return zbar, dzbar, ddzbar, xw, yw, Fa


# ---------- System matrices ----------
def system_mats(z: np.ndarray, dz: np.ndarray, t: float, params: dict):
    """
    Two-DOF equivalent model in the spirit of Eq.(21):

        M qdd + C qd + K q = Fext + f p,  with q = [z_w, z_s]^T.

    K includes the position/time-varying workpiece stiffness k_w(xw,yw).
    f has two nonzero rows with alpha=beta=1 (transformed input sharing).
    """
    mw, ms = params["mw"], params["ms"]
    cw, cs = params["cw"], params["cs"]
    kc, kr = params["kc"], params["kr"]
    s_area  = params["s_area"]
    alpha   = params["alpha"]
    beta    = params["beta"]
    D, a, b = params["D"], params["a"], params["b"]

    # Reference & machining location (single call to be consistent)
    zbar, dzbar, ddzbar, xw, yw, Fa = trajectory(t)

    # Spatial stiffness at the machining point
    kw = kw_plate(xw, yw, D=D, a=a, b=b)

    # Matrices
    M = np.array([[mw, 0.0],
                  [0.0, ms]], dtype=float)
    C = np.array([[cw, 0.0],
                  [0.0, cs]], dtype=float)

    # Workpiece-to-ground: kw + kr + kc; support-to-ground: kc; coupling: -kc
    K = np.array([[kw + kr + kc, -kc],
                  [-kc,           kc]], dtype=float)

    # Input map (two rows)
    f = s_area * np.array([[alpha],
                           [beta]], dtype=float)

    # External excitation on the workpiece only
    Fext = np.array([[Fa],
                     [0.0]], dtype=float)

    return M, C, K, f, Fext, zbar, dzbar, ddzbar


# ---------- Continuous-time dynamics (for RK4 integrator) ----------
def rk4_step(f_fun, x, t, h, *args):
    k1 = f_fun(x, t, *args)
    k2 = f_fun(x + 0.5 * h * k1, t + 0.5 * h, *args)
    k3 = f_fun(x + 0.5 * h * k2, t + 0.5 * h, *args)
    k4 = f_fun(x + h * k3,       t + h,       *args)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def f_state(state: np.ndarray, t: float, u_pressure: float, params: dict) -> np.ndarray:
    """
    Continuous dynamics with pressure input:
      x = [z_w, z_s, dz_w, dz_s]^T;  u_pressure = p [Pa].
    """
    z  = state[0:2].reshape(2, 1)
    dz = state[2:4].reshape(2, 1)
    M, C, K, f, Fext, *_ = system_mats(z, dz, t, params)

    # Apply hard saturation at ±p_max
    p = float(np.clip(u_pressure, -params["p_max"], params["p_max"]))

    ddz = np.linalg.solve(M, -C @ dz - K @ z + Fext + f * p)
    return np.vstack([dz, ddz]).flatten()


# ---------- FLC controller (Eq. 37; two-row 2x2 solve) ----------
def flc_pressure(state: np.ndarray, t: float, params: dict, *, kp: float, kd: float) -> float:
    """
    Compute FLC pressure p(t) using a 2x2 solve on BOTH dynamic rows.

    Virtual command on the workpiece channel:
        Z1 = z̈_ref + kd (ż_ref - ż_w) + kp (z_ref - z_w)

    From M qdd + C qd + K q = Fext + f p, build two scalar equations
    (row 1 and row 2), with unknowns u = [z̈_s, p]. Solve A u = b in
    least-squares (2x2 here), then clip p to ±p_max.
    """
    z  = state[0:2].reshape(2, 1)
    dz = state[2:4].reshape(2, 1)
    M, C, K, f, Fext, zbar, dzbar, ddzbar = system_mats(z, dz, t, params)

    # Desired (virtual) acceleration on workpiece
    Z1 = ddzbar + kd * (dzbar - float(dz[0, 0])) + kp * (zbar - float(z[0, 0]))

    # Shorthands
    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]
    f1, f2   = float(f[0, 0]), float(f[1, 0])

    C1 = (C[0, :] @ dz).item()
    K1 = (K[0, :] @ z ).item()
    C2 = (C[1, :] @ dz).item()
    K2 = (K[1, :] @ z ).item()
    F1 = float(Fext[0, 0])
    F2 = float(Fext[1, 0])

    # Build A u = b, u = [z̈_s, p]
    # From: M*[...] + C*dz + K*z - Fext - f*p = 0
    # -> Row1: M11*Z1 + M12*z̈_s + C1 + K1 - F1 - f1*p = 0
    # -> Row2: M21*Z1 + M22*z̈_s + C2 + K2 - F2 - f2*p = 0
    A = np.array([[M12, -f1],
                  [M22, -f2]], dtype=float)
    b = np.array([
        [-(M11 * Z1 + C1 + K1 - F1)],
        [-(M21 * Z1 + C2 + K2 - F2)]
    ], dtype=float)

    # Robust solve (tiny Tikhonov if near-singular)
    AtA = A.T @ A
    if np.linalg.cond(AtA) > 1e12:
        AtA += 1e-12 * np.eye(2)
    u = np.linalg.solve(AtA, A.T @ b)

    # Extract pressure and apply saturation
    _zdd_s = float(u[0, 0])  # not used further, kept for completeness
    p_cmd  = float(np.clip(u[1, 0], -params["p_max"], params["p_max"]))
    return p_cmd


__all__ = [
    "default_params",
    "trajectory",
    "kw_plate",
    "system_mats",
    "rk4_step",
    "f_state",
    "flc_pressure",
]
