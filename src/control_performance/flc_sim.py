"""
Grid-search the FLC gains (kp, kd) over paper-like ranges and visualize e_RMS.

Time window: t in [0, 3π], dt = 1e-3.
Reference: zbar_w = 1e-3 * sin(t).
External excitation on workpiece: Fa = 100 * sin(2π t) [N].
Pressure limits: ±1 MPa.

We use the two-row-solve FLC (see flc.py) so that the virtual command Z1
actually feeds the control channel through the coupled dynamics.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from flc import default_params, rk4_step, f_state, flc_pressure, trajectory


def simulate_flc(kp: float, kd: float, params: dict) -> dict:
    """
    Run one simulation with given (kp, kd) and return metrics.
      - e_rms: RMS of position error e = z_w - zbar_w
      - sat_ratio: fraction of time steps where |p| hits saturation
      - p_rms: RMS of pressure [Pa]
      - p_peak: max |p| [Pa]
    """
    t0, t1, dt = params["t0"], params["t1"], params["dt"]
    N = int(np.round((t1 - t0) / dt)) + 1

    x = np.zeros(4, dtype=float)  # [z_w, z_s, dz_w, dz_s]
    t = t0

    e_hist = np.empty(N, dtype=float)
    p_hist = np.empty(N, dtype=float)

    for i in range(N):
        zbar, *_ = trajectory(t)
        e_hist[i] = x[0] - zbar

        p_cmd = flc_pressure(x, t, params, kp=kp, kd=kd)
        p_hist[i] = p_cmd

        x = rk4_step(lambda s, tt, uu: f_state(s, tt, uu, params), x, t, dt, p_cmd)
        t += dt

    e_rms = float(np.sqrt(np.mean(e_hist**2)))
    p_max = params["p_max"]
    sat_ratio = float(np.mean(np.abs(p_hist) >= 0.999 * p_max))

    return {
        "e_rms": e_rms,
        "sat_ratio": sat_ratio,
        "p_rms": float(np.sqrt(np.mean(p_hist**2))),
        "p_peak": float(np.max(np.abs(p_hist))),
        "e_hist": e_hist,
        "p_hist": p_hist,
    }


def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    p = default_params()

    # Paper-like coarse ranges (cf. Fig. 11): kp∈[2e4,1e5], kd∈[5e3,1.5e4]
    kp_vals = np.linspace(1.0e3, 1.0e4, 13)   # 13 points: 20k .. 100k
    kd_vals = np.linspace(100, 2000, 11)   # 11 points: 5k .. 15k

    Z = np.zeros((len(kd_vals), len(kp_vals)))  # e_RMS grid
    SAT = np.zeros_like(Z)                      # saturation ratio

    # Coarse grid (progress bar)
    for i, kd in enumerate(tqdm(kd_vals, desc="FLC coarse grid")):
        for j, kp in enumerate(kp_vals):
            out = simulate_flc(kp, kd, p)
            Z[i, j] = out["e_rms"]
            SAT[i, j] = out["sat_ratio"]

    # Best (minimum e_RMS)
    idx = np.unravel_index(np.argmin(Z), Z.shape)
    kd_opt, kp_opt = kd_vals[idx[0]], kp_vals[idx[1]]
    best = Z[idx]
    print(f"[FLC coarse] best ≈ eRMS={best/1e-4:.3f} (×1e-4 m) at kp={kp_opt:.0f}, kd={kd_opt:.0f}")
    print(f"[diag] grid median sat-ratio = {np.median(SAT):.3f}")

    # Plot 3D surface of e_RMS
    kd_mesh, kp_mesh = np.meshgrid(kd_vals, kp_vals, indexing="ij")
    fig = plt.figure(figsize=(9.6, 7.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(kp_mesh, kd_mesh, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.scatter([kp_opt], [kd_opt], [best], color="k", s=40)
    ax.set_xlabel(r"$k_p$")
    ax.set_ylabel(r"$k_d$")
    ax.set_zlabel(r"$e_{\mathrm{RMS}}$ (m)")
    ax.set_title("FLC parameter sweep (coarse)")
    cbar = fig.colorbar(surf, shrink=0.75, pad=0.08)
    cbar.set_label(r"$e_{\mathrm{RMS}}$ (m)")
    fig.tight_layout()
    fig.savefig("figs/flc_opt_surface_coarse.png")
    plt.close(fig)

    # Re-simulate the best point and dump a couple of time-series plots (optional)
    out_best = simulate_flc(kp_opt, kd_opt, p)

    # Error time-series
    t_axis = np.linspace(p["t0"], p["t1"], out_best["e_hist"].size)
    fig_e = plt.figure(figsize=(9.6, 5.4), dpi=140)
    ax_e = fig_e.add_subplot(111)
    ax_e.plot(t_axis, out_best["e_hist"], lw=1.2)
    ax_e.set_xlabel("Time (s)")
    ax_e.set_ylabel(r"$e(t)=z_w-\bar{z}_w$ (m)")
    ax_e.set_title(f"FLC error @ kp={kp_opt:.0f}, kd={kd_opt:.0f}  (eRMS={out_best['e_rms']:.2e} m)")
    ax_e.grid(alpha=0.25, linestyle=":")
    fig_e.tight_layout()
    fig_e.savefig("figs/flc_best_error_time.png")
    plt.close(fig_e)

    # Pressure time-series (in MPa)
    fig_p = plt.figure(figsize=(9.6, 5.4), dpi=140)
    ax_p = fig_p.add_subplot(111)
    ax_p.plot(t_axis, out_best["p_hist"] / 1e6, lw=1.2)
    ax_p.set_xlabel("Time (s)")
    ax_p.set_ylabel(r"$p(t)$ (MPa)")
    ax_p.set_title(f"FLC pressure @ kp={kp_opt:.0f}, kd={kd_opt:.0f}")
    ax_p.margins(x=0)
    ax_p.grid(alpha=0.25, linestyle=":")
    fig_p.tight_layout()
    fig_p.savefig("figs/flc_best_pressure_time.png")
    plt.close(fig_p)

    # Save raw grid and saturation map for inspection
    np.savez("data/flc_grid_coarse.npz", kp_vals=kp_vals, kd_vals=kd_vals, Z=Z, SAT=SAT,
             kp_opt=kp_opt, kd_opt=kd_opt, e_best=best)


if __name__ == "__main__":
    main()
