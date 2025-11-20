# cfc_flc_compare.py
# CFC vs FLC on the SAME Eq.(21) plant (from cfc.py) to mirror Fig.12/13.
# Outputs:
#   figs/cfc_flc_error_time.png
#   figs/cfc_flc_error_bar.png
#   figs/cfc_flc_pressure_time.png
#   figs/cfc_flc_cost_bar.png

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

# Use the Eq.(21) plant from cfc.py for BOTH controllers
from cfc import (
    default_params, trajectory, mats_eq21,
    rk4_step, f_state_eq21, cfc_pressure
)

# ------------ FLC on the Eq.(21) plant (paper Eq.37) -----------------
def flc_pressure_eq21(state: np.ndarray, t: float, p: dict, kp: float, kd: float) -> float:
    # === states & model mats ===
    z  = state[0:2].reshape(2, 1)
    dz = state[2:4].reshape(2, 1)

    M, Ct, Kt, Ft, f_vec = mats_eq21(z, dz, t, p)   # 2x2, 2x2, 2x2, 2x1, 2x1
    zref, dzref, ddzref, *_ = trajectory(t)

    # === virtual accel on workpiece (Eq.37 form) ===
    # e = z_w - z_ref  =>  if ddz_w = Z1, then e'' + kd e' + kp e = 0
    Z1 = ddzref + kd*(dzref - float(dz[0,0])) + kp*(zref - float(z[0,0]))

    # Expand into scalars a la Eq.(37): solve A u = b with u = [z̈_s, p]
    M11, M12 = float(M[0,0]), float(M[0,1])
    M21, M22 = float(M[1,0]), float(M[1,1])
    f1,  f2  = float(f_vec[0,0]), float(f_vec[1,0])

    C1 = float((Ct[0,:] @ dz).item());  K1 = float((Kt[0,:] @ z ).item());  F1 = float(Ft[0,0])
    C2 = float((Ct[1,:] @ dz).item());  K2 = float((Kt[1,:] @ z ).item());  F2 = float(Ft[1,0])

    A = np.array([[M12, -f1],
                  [M22, -f2]], dtype=float)
    b = np.array([[-(M11*Z1 + C1 + K1 - F1)],
                  [-(M21*Z1 + C2 + K2 - F2)]], dtype=float)

    # Numerically safe solve (add a tiny Tikhonov ridge if A^T A is ill-conditioned)
    AtA = A.T @ A
    if np.linalg.cond(AtA) > 1e12:
        AtA += 1e-12*np.eye(2)
    u = np.linalg.solve(AtA, A.T @ b)   # 2x1
    p_cmd = float(u[1,0])

    # If the tentative command exceeds ±p_max, rescale Z1 once and re-solve to stay physical
    p_abs = abs(p_cmd)
    if p_abs > p["p_max"]:
        # p is linear in Z1, so shrinking Z1 pulls p right back into range
        scale = (p["p_max"] - 1e3) / (p_abs + 1e-9)   # keep a small 1e3 Pa buffer
        Z1 *= max(0.0, min(1.0, scale))
        b = np.array([[-(M11*Z1 + C1 + K1 - F1)],
                      [-(M21*Z1 + C2 + K2 - F2)]], dtype=float)
        u = np.linalg.solve(AtA, A.T @ b)
        p_cmd = float(u[1,0])

    # Final clamp for good measure
    return float(np.clip(p_cmd, -p["p_max"], p["p_max"]))



# ------------------------ Generic simulator on Eq.(21) ------------------------
def simulate_on_eq21(controller_fn, params: dict, **ctrl_kwargs) -> dict:
    """
    Runs one simulation on the Eq.(21) plant using f_state_eq21().
    Returns histories and integral metrics (e_sum, p_cost).
    """
    t0, t1, dt = params["t0"], params["t1"], params["dt"]
    N = int(np.round((t1 - t0) / dt)) + 1
    x = np.zeros(4, dtype=float)  # [z_w, z_r, dz_w, dz_r]
    t = t0

    t_hist   = np.empty(N, dtype=float)
    zref_hist= np.empty(N, dtype=float)
    e_hist   = np.empty(N, dtype=float)
    p_hist   = np.empty(N, dtype=float)

    for i in range(N):
        zref, *_ = trajectory(t)
        zref_hist[i] = zref
        t_hist[i]    = t
        e_hist[i]    = x[0] - zref

        p_cmd = controller_fn(x, t, params, **ctrl_kwargs)
        p_hist[i] = p_cmd

        x = rk4_step(lambda s, tt, uu: f_state_eq21(s, tt, uu, params), x, t, dt, p_cmd)
        t += dt

    e_sum  = float(np.trapz(np.abs(e_hist), t_hist))
    p_cost = float(np.trapz(np.abs(p_hist), t_hist))

    return dict(t=t_hist, zref=zref_hist, e=e_hist, p=p_hist,
                e_sum=e_sum, p_cost=p_cost)


# ---------------------------------- Main --------------------------------------
def main():
    os.makedirs("figs", exist_ok=True)
    p = default_params()   # Eq.(21) plant params from cfc.py

    # --- Controller settings (paper-like) ---
    # CFC (typical values consistent with Fig.14/15 sweep)
    k1, k2 = 150.0, 1.0e8       # P_gain is fixed to 100.0 inside cfc_pressure

    # FLC (stable/reasonable on Eq.21; adjust if you want to match your grid search)
    kp, kd = 3.6e3, 110

    # Run both on the SAME plant (Eq.21)
    out_cfc = simulate_on_eq21(lambda s, t, pp, **kw: cfc_pressure(s, t, pp, kappa1=k1, kappa2=k2),
                               p)
    out_flc = simulate_on_eq21(lambda s, t, pp, **kw: flc_pressure_eq21(s, t, pp, kp=kp, kd=kd),
                               p)

    # --- Fig.12a-like: error time histories ---
    fig1 = plt.figure(figsize=(8.8, 5.2), dpi=140)
    ax1 = fig1.add_subplot(111)
    ax1.plot(out_cfc["t"], out_cfc["e"], "-",  label="CFC")
    ax1.plot(out_flc["t"], out_flc["e"], "--", label="FLC")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"$e(t)=z_w-\bar{z}_w$ (m)")
    ax1.set_title("Control error (vibration displacement)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("figs/cfc_flc_error_time.png")
    plt.close(fig1)

    # --- Fig.12b-like: e_sum bar chart ---
    fig2 = plt.figure(figsize=(5.2, 5.2), dpi=140)
    ax2 = fig2.add_subplot(111)
    vals = [out_cfc["e_sum"], out_flc["e_sum"]]
    labels = ["CFC", "FLC"]
    bars = ax2.bar([0, 1], vals, tick_label=labels)
    ax2.set_ylabel(r"$e_{\mathrm{sum}}$ (m·s)")
    ax2.set_title("Total position error")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2.0, v*1.02, f"{v:.2e}", ha="center")
    fig2.tight_layout()
    fig2.savefig("figs/cfc_flc_error_bar.png")
    plt.close(fig2)

    # --- Fig.13a-like: PURE LINE Control pressure (MPa), no fill, no inset ---
    fig3 = plt.figure(figsize=(8.8, 5.2), dpi=140)
    ax3 = fig3.add_subplot(111)

    # Convert Pa -> MPa for display
    p_cfc_mpa = out_cfc["p"] / 1e6
    p_flc_mpa = out_flc["p"] / 1e6

    ax3.plot(out_cfc["t"], p_cfc_mpa, color="tab:red",  lw=1.8, label="CFC")
    ax3.plot(out_flc["t"], p_flc_mpa, color="tab:blue", lw=1.8, ls="--", label="FLC")

    ax3.set_xlabel("Time(s)")
    ax3.set_ylabel(r"$p(t)$ (MPa)")
    ax3.set_title("Control pressure")
    ax3.margins(x=0)
    # Keep a sane y-range that shows saturation clearly but avoids "filled" look
    ymin = min(p_cfc_mpa.min(), p_flc_mpa.min())
    ymax = max(p_cfc_mpa.max(), p_flc_mpa.max())
    ax3.set_ylim(min(-1.05, ymin - 0.05), max(1.05,  ymax + 0.05))
    ax3.grid(alpha=0.25, linestyle=":")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig("figs/cfc_flc_pressure_time.png")
    plt.close(fig3)

    # --- Fig.13b-like: p_cost bar chart ---
    fig4 = plt.figure(figsize=(5.2, 5.2), dpi=140)
    ax4 = fig4.add_subplot(111)
    vals = [out_cfc["p_cost"], out_flc["p_cost"]]
    labels = ["CFC", "FLC"]
    bars = ax4.bar([0, 1], vals, tick_label=labels)
    ax4.set_ylabel(r"$p_{\mathrm{cost}}$ (Pa·s)")
    ax4.set_title("Total control cost")
    for b, v in zip(bars, vals):
        ax4.text(b.get_x() + b.get_width()/2.0, v*1.02, f"{v:.2e}", ha="center")
    fig4.tight_layout()
    fig4.savefig("figs/cfc_flc_cost_bar.png")
    plt.close(fig4)

    # Print summary
    print(f"[CFC] e_sum={out_cfc['e_sum']:.3e}, p_cost={out_cfc['p_cost']:.3e}")
    print(f"[FLC] e_sum={out_flc['e_sum']:.3e}, p_cost={out_flc['p_cost']:.3e}")

if __name__ == "__main__":
    main()
