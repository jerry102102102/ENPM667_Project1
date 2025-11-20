# cfc.py
# Equivalent model (Eq.21) + CFC controller (Eq.31/35/36)
# Units: SI. Pressure saturation ±1 MPa.

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


# ----------------------------- Parameters ------------------------------------
def default_params() -> Dict[str, float]:
    p: Dict[str, float] = {}

    # Masses
    p["mw"] = 13.5          # workpiece mass [kg]
    p["mr"] = 2000.0        # robot-side equivalent mass [kg]

    # Support modules & chamber section
    p["ns"]     = 7
    p["s_area"] = float(np.pi * 1e-4)  # [m^2]

    # Robot normal stiffness (can be pose-dependent; here a representative constant)
    p["kr0"] = 3.0e6        # [N/m]

    # Plate-equivalent stiffness model parameters (used by Eq.13 along path)
    p["D_plate"] = 800.0    # [N·m]
    p["a_half"]  = 0.5      # [m]
    p["b_half"]  = 0.5      # [m]

    # Rayleigh damping coefficients (set to None => auto-fit to ~300 N·s/m at t=t0)
    p["alpha_ray"] = None   # [1/s]
    p["beta_ray"]  = None   # [s]

    # Pressure saturation
    p["p_max"] = 1.0e6      # ±1 MPa

    # Simulation window
    p["t0"] = 0.0
    p["t1"] = 3.0 * np.pi
    p["dt"] = 5e-4          # smaller step for robustness

    # internal flag for one-time Rayleigh fit
    p["_rayleigh_inited"] = False
    return p


# ------------------------- Reference & Excitation -----------------------------
def trajectory(t: float) -> Tuple[float, float, float, float, float, float]:
    zbar   = 1e-3 * np.sin(t)
    dzbar  = 1e-3 * np.cos(t)
    ddzbar = -1e-3 * np.sin(t)
    xw     = 0.05 * t - 0.4
    yw     = 0.0
    Fa     = 100.0 * np.sin(2.0 * np.pi * t)
    return float(zbar), float(dzbar), float(ddzbar), float(xw), float(yw), float(Fa)


# ------------------------- Time-varying stiffness -----------------------------
def kw_time(t: float, p: Dict[str, float]) -> float:
    D = p["D_plate"]; a = p["a_half"]; b = p["b_half"]
    _, _, _, xw, yw, _ = trajectory(t)
    c1 = 1.0 + np.cos(np.pi * xw / a)
    c2 = 1.0 + np.cos(np.pi * yw / b)
    denom = (c1 * c1) * (c2 * c2)
    denom = max(denom, 1e-12)
    num = D * (np.pi ** 4) * (3 * a**4 + 2 * a**2 * b**2 + 3 * b**4)
    kw = num / (a**3 * b**3 * denom)
    return float(kw)


def kr_time(_: float, p: Dict[str, float]) -> float:
    return float(p["kr0"])


# ---------------------- Rayleigh coefficients (auto-fit) ----------------------
def _fit_rayleigh_to_targets(p: Dict[str, float], t_ref: float, target_cw: float = 300.0, target_cr: float = 300.0):
    """
    Solve [ [mw, kw],[mr, kr] ] [alpha, beta]^T = [target_cw, target_cr]^T at t=t_ref.
    Gives α[1/s], β[s] that yield ≈300 N·s/m on both channels near t_ref.
    """
    mw, mr = p["mw"], p["mr"]
    kw0 = kw_time(t_ref, p)
    kr0 = kr_time(t_ref, p)
    A = np.array([[mw, kw0],
                  [mr, kr0]], dtype=float)
    b = np.array([target_cw, target_cr], dtype=float)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    p["alpha_ray"], p["beta_ray"] = float(sol[0]), float(sol[1])
    p["_rayleigh_inited"] = True


# ------------------------------ Model pieces ----------------------------------
def mats_eq21(z: np.ndarray, dz: np.ndarray, t: float, p: Dict[str, float]):
    """
    Build M, C~, K~, F~, f for Eq.(21) with Rayleigh damping.
    f is a constant input map: f = n_s * s_area * [-1, +1]^T.
    """
    if not p["_rayleigh_inited"] or p["alpha_ray"] is None or p["beta_ray"] is None:
        _fit_rayleigh_to_targets(p, p["t0"], target_cw=300.0, target_cr=300.0)

    mw, mr = p["mw"], p["mr"]
    kw = kw_time(t, p)
    kr = kr_time(t, p)
    _, _, _, _, _, Fa = trajectory(t)

    # Mass
    M = np.array([[mw, 0.0],
                  [0.0, mr]], dtype=float)

    # Stiffness (time-varying diagonal)
    Kt = np.array([[kw, 0.0],
                   [0.0, kr]], dtype=float)

    # Rayleigh damping
    Ct = p["alpha_ray"] * M + p["beta_ray"] * Kt

    # External force (sign as in the paper)
    Ft = np.array([[-Fa],
                   [ 0.0]], dtype=float)

    # Constant input map
    f_vec = p["ns"] * p["s_area"] * np.array([[-1.0],
                                              [+1.0]], dtype=float)

    return M, Ct, Kt, Ft, f_vec


# ------------------------------ Integrator ------------------------------------
def rk4_step(f, x, t, h, *args):
    k1 = f(x, t, *args)
    k2 = f(x + 0.5*h*k1, t + 0.5*h, *args)
    k3 = f(x + 0.5*h*k2, t + 0.5*h, *args)
    k4 = f(x + h*k3,     t + h,     *args)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# --------------------------- State time derivative ----------------------------
def f_state_eq21(state: np.ndarray, t: float, p_cmd: float, p: Dict[str, float]) -> np.ndarray:
    """
    x = [z_w, z_r, dz_w, dz_r]^T
    M ddz + C~ dz + K~ z = F~ + f p
    """
    z  = state[0:2].reshape(2,1)
    dz = state[2:4].reshape(2,1)

    M, Ct, Kt, Ft, f_vec = mats_eq21(z, dz, t, p)
    p_sat = float(np.clip(p_cmd, -p["p_max"], p["p_max"]))

    rhs = -Ct @ dz - Kt @ z + Ft + f_vec * p_sat
    ddz = np.linalg.solve(M, rhs)
    return np.vstack([dz, ddz]).flatten()


# ---------------------------- CFC (Eq.31/35/36) -------------------------------
def cfc_pressure(state: np.ndarray, t: float, p: Dict[str, float],
                 kappa1: float = 100.0, kappa2: float = 2.0e5, P_gain: float = 100.0) -> float:
    """
    p = p1 + p2
      gamma = A M^{-1} f
      b     = ddz_ref + kappa1 (dz_ref - dz_w)
      p1    = [ b + A M^{-1}(C~ dz + K~ z - F~) ] / gamma
      d     = dz_ref + kappa1 (z_ref - z_w)
      e     = A dz - d
      p2    = - kappa2 * gamma * P * e
    """
    z  = state[0:2].reshape(2,1)
    dz = state[2:4].reshape(2,1)

    M, Ct, Kt, Ft, f_vec = mats_eq21(z, dz, t, p)
    zref, dzref, ddzref, *_ = trajectory(t)

    # Workpiece constraint row
    A = np.array([[1.0, 0.0]], dtype=float)

    Minv = np.linalg.inv(M)
    gamma = float((A @ Minv @ f_vec)[0, 0])
    if not np.isfinite(gamma) or abs(gamma) < 1e-10:
        gamma = np.sign(gamma) * 1e-10 if gamma != 0.0 else 1e-10

    # p1 term
    b = float(ddzref + kappa1 * (dzref - float(dz[0, 0])))
    aux = float((A @ Minv @ (Ct @ dz + Kt @ z - Ft))[0, 0])
    p1 = (b + aux) / gamma

    # p2 term
    d = float(dzref + kappa1 * (zref - float(z[0, 0])))
    e = float((A @ dz)[0, 0] - d)
    p2 = - kappa2 * gamma * P_gain * e

    p_cmd = float(np.clip(p1 + p2, -p["p_max"], p["p_max"]))
    return p_cmd


# ----------------------------- Simulation helper ------------------------------
def simulate_cfc(p: Dict[str, float], kappa1: float, kappa2: float, P_gain: float = 100.0) -> dict:
    """
    Run one CFC simulation over [t0, t1]. Returns metrics & histories.
    If numerical instability occurs, metrics are NaN.
    """
    t0, t1, dt = p["t0"], p["t1"], p["dt"]
    N = int(np.round((t1 - t0) / dt)) + 1

    x = np.zeros(4, dtype=float)  # [z_w, z_r, dz_w, dz_r]
    t = t0

    t_hist  = np.empty(N, dtype=float)
    e_hist  = np.empty(N, dtype=float)
    p_hist  = np.empty(N, dtype=float)
    zw_hist = np.empty(N, dtype=float)
    zref_hist = np.empty(N, dtype=float)

    valid = True

    for i in range(N):
        zref, *_ = trajectory(t)
        e_hist[i]    = x[0] - zref
        zref_hist[i] = zref
        zw_hist[i]   = x[0]
        t_hist[i]    = t

        try:
            u = cfc_pressure(x, t, p, kappa1=kappa1, kappa2=kappa2, P_gain=P_gain)
            p_hist[i] = u
            x = rk4_step(lambda s, tt, uu: f_state_eq21(s, tt, uu, p), x, t, dt, u)
        except Exception:
            valid = False
            break

        if (not np.all(np.isfinite(x))) or (np.max(np.abs(x[:2])) > 0.5):
            valid = False
            break

        t += dt

    if not valid:
        return dict(
            e_rms=np.nan, e_sum=np.nan, p_cost=np.nan, sat_ratio=np.nan,
            t=t_hist, e=e_hist, p=p_hist, zw=zw_hist, zref=zref_hist
        )

    e_rms   = float(np.sqrt(np.mean(e_hist**2)))
    e_sum   = float(np.trapz(np.abs(e_hist), t_hist))
    p_cost  = float(np.trapz(np.abs(p_hist), t_hist))
    sat_ratio = float(np.mean(np.abs(p_hist) >= 0.999 * p["p_max"]))

    return dict(
        e_rms=e_rms, e_sum=e_sum, p_cost=p_cost, sat_ratio=sat_ratio,
        t=t_hist, e=e_hist, p=p_hist, zw=zw_hist, zref=zref_hist
    )
