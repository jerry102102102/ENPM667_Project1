# ks_support.py
# Gas-spring equivalent stiffness ks(p,l) & p-l map
# Produces: data/support_single.csv, figs/ks_surface.png (or contour)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR_DATA = "data"
OUT_DIR_FIGS = "figs"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

# ----- Parameters (from paper 4.1) -----
l = 0.01                       # rod extension [m]
s = np.pi * 1e-4               # rodless chamber area [m^2]
p = 1.0e5                      # gas pressure [Pa]
gamma = 1.4                    # adiabatic coefficient
F = 5.0                        # axial force [N]

# References from paper (for comparison only)
paper_disp_fea = 1.14e-3
paper_disp_analytic = 1.12e-3

def ks_pl(p: float, l: float, s: float, gamma: float) -> float:
    """ Eq. (14): ks = gamma * (s / l) * p """
    return gamma * (s / max(l, 1e-12)) * p

def main():
    ks_val = ks_pl(p, l, s, gamma)
    delta = F / ks_val

    df = pd.DataFrame([{
        "p_Pa": p, "l_m": l, "s_m2": s, "gamma": gamma,
        "ks_N_per_m": ks_val, "delta_m": delta,
        "paper_fea_m": paper_disp_fea, "paper_analytic_m": paper_disp_analytic,
        "accuracy_vs_fea_%": 100.0 * (1.0 - abs(delta - paper_disp_fea)/paper_disp_fea)
    }])
    df.to_csv(os.path.join(OUT_DIR_DATA, "support_single.csv"), index=False)

    # p-l map for ks
    p_grid = np.linspace(1.0e5, 5.0e5, 41)
    l_grid = np.linspace(0.005, 0.015, 41)
    P, L = np.meshgrid(p_grid, l_grid)
    KS = gamma * (s / np.maximum(L, 1e-12)) * P

    plt.figure(figsize=(6.8, 5.4), dpi=160)
    cs = plt.contourf(P*1e-5, L, KS, levels=30)  # show p in 1e5 Pa
    cbar = plt.colorbar(cs)
    cbar.set_label(r'$k_s\,[\mathrm{N/m}]$')
    plt.xlabel(r'$p\,(10^5 \mathrm{Pa})$')
    plt.ylabel(r'$l\,(m)$')
    plt.title('Gas-spring stiffness map')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FIGS, "ks_contour.png"))
    plt.close()

    print("[ks_support] ks=%.3e N/m, delta=%.3e m; FEA(ref)=%.3e m" %
          (ks_val, delta, paper_disp_fea))
    print("[ks_support] figs/ks_contour.png & data/support_single.csv written.")

if __name__ == "__main__":
    main()
